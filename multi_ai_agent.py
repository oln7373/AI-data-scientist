"""Multi-agent AI data scientist pipeline with MCP tool integration.

Provides:
    /mcp_add_demo      Simple MCP tool demo (arithmetic via MCP).
    /mcp_agent_add     LLM tool-calling via allowlist-enforced MCP.
    /mcp_tools         List tools from the MCP server.
    /multi_ai_agent    AutoGen multi-agent data scientist pipeline with
                       true tool-calling via MCP-backed AutoGen tools and
                       optional email delivery via the MCP send_email tool.

Design:
    - MCP allowlist enforcement is applied on the backend for all tool calls.
    - Email side-effects go through MCP (send_email tool), not direct code.
    - MCP tools are exposed to AutoGen as real callable tools whose
      implementations bounce from AutoGen worker threads back to the FastAPI
      event loop using anyio.from_thread.run(...).
"""

import json
import os
import re

import anyio
import autogen
import httpx
import requests
import structlog
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import get_config
from email_utils import (
    compose_email_payload,
    extract_email_and_clean_prompt,
    make_absolute_image_url,
)
from response_extractor import extract_relevant_output

load_dotenv()

logger = structlog.get_logger(__name__)

router = APIRouter()

_cfg = get_config()

ALLOWED_TOOLS = {t.strip() for t in os.getenv("ALLOWED_TOOLS", "").split(",") if t.strip()}

if not ALLOWED_TOOLS:
    raise RuntimeError("ALLOWED_TOOLS not configured in .env")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

if LLM_PROVIDER == "openai":
    LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    LLM_BASE_URL = "https://api.openai.com/v1"
else:
    OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
    LLM_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    LLM_BASE_URL = f"http://localhost:{OLLAMA_PORT}/v1"
    LLM_API_KEY = "ollama"

LLM_BASE_URL = os.getenv("LLM_BASE_URL") or LLM_BASE_URL
LLM_URL = f"{LLM_BASE_URL}/chat/completions"

DATA_URL = _cfg.data.dataset_url
IMAGE_DIR = _cfg.data.image_dir
AGENT_CSV = _cfg.data.dataset_filename
SERVER_CSV_PATH = os.path.join(IMAGE_DIR, AGENT_CSV)

os.makedirs(IMAGE_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    """Request body for all agent query endpoints."""

    question: str


@router.post("/mcp_add_demo")
async def mcp_add_demo(request: QueryRequest, http_req: Request) -> dict:
    """Demonstrate direct MCP tool invocation for addition.

    Parses two numbers from the prompt and calls the MCP add_numbers tool
    directly (no LLM involved).

    Args:
        request: Body containing a question with an expression like ``5 + 7``.
        http_req: FastAPI request used to access the MCP client from app state.

    Returns:
        The MCP tool result dict.

    Raises:
        HTTPException: 400 if the expression cannot be parsed or the tool is
            not on the allowlist. 503 if the MCP client is unavailable.
    """
    mcp_http = getattr(http_req.app.state, "mcp_http", None)
    if mcp_http is None:
        raise HTTPException(status_code=503, detail="MCP HTTP client not available")

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*\+\s*(-?\d+(?:\.\d+)?)", request.question)
    if not m:
        raise HTTPException(
            status_code=400,
            detail="Provide a question containing something like '5 + 7'",
        )

    a = float(m.group(1))
    b = float(m.group(2))

    if "add_numbers" not in ALLOWED_TOOLS:
        raise HTTPException(status_code=400, detail="Tool not allowed by policy: add_numbers")

    return await mcp_http.tool_call("add_numbers", {"a": a, "b": b})


@router.post("/mcp_agent_add")
async def mcp_agent_add(request: QueryRequest, http_req: Request) -> dict:
    """LLM-driven tool-calling demonstration using MCP-backed tools.

    Sends the user prompt to the configured LLM, which emits OpenAI-style
    tool_calls. The backend validates the requested tool against the allowlist
    before invoking it via MCP.

    Args:
        request: Body containing the user's question.
        http_req: FastAPI request used to access the MCP client from app state.

    Returns:
        Dict containing llm_used, tool call details, and MCP response.

    Raises:
        HTTPException: 400/502/503 on policy violations or LLM errors.
    """
    mcp_http = getattr(http_req.app.state, "mcp_http", None)
    if mcp_http is None:
        raise HTTPException(status_code=503, detail="MCP HTTP client not available")

    system = (
        "You are a tool-using agent.\n"
        "Available tools:\n"
        "- add_numbers(a, b)\n"
        "- ping(message)\n"
        "Rules:\n"
        "- If the user asks to add numbers → call add_numbers.\n"
        "- If the user asks for a health check or connectivity test → call ping.\n"
        "Use tool calling when possible.\n"
    )

    _headers = {"Content-Type": "application/json"}
    if LLM_PROVIDER == "openai":
        _headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    async with httpx.AsyncClient(trust_env=False, timeout=_cfg.llm.extraction_timeout_seconds) as client:
        resp = await client.post(
            LLM_URL,
            headers=_headers,
            json={
                "model": LLM_MODEL,
                "temperature": _cfg.llm.temperature_agent,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": request.question},
                ],
            },
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"LLM error {resp.status_code}: {resp.text[:1000]}",
        )

    data = resp.json()

    try:
        msg = data["choices"][0]["message"]
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected LLM response shape: {e}. Raw={str(data)[:1000]}",
        ) from e

    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        try:
            fn = tool_calls[0]["function"]["name"]
            arg_str = tool_calls[0]["function"].get("arguments", "{}")
            args = json.loads(arg_str) if isinstance(arg_str, str) else (arg_str or {})
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to parse tool_calls: {e}. Raw={str(tool_calls)[:1000]}",
            ) from e

        if fn not in ALLOWED_TOOLS:
            raise HTTPException(status_code=400, detail=f"Tool not allowed by policy: {fn}")

        tool_resp = await mcp_http.tool_call(fn, args)
        logger.info("mcp_tool_called", tool=fn, args=args)

        return {
            "llm_used": {"provider": LLM_PROVIDER, "url": LLM_URL, "model": LLM_MODEL},
            "llm_finish_reason": data["choices"][0].get("finish_reason"),
            "llm_tool_call": {"tool": fn, "args": args},
            "mcp_response": tool_resp,
            "allowed_tools": sorted(ALLOWED_TOOLS),
        }

    content = (msg.get("content") or "").strip()
    if not content:
        raise HTTPException(
            status_code=502,
            detail=f"LLM returned neither tool_calls nor content. Raw response: {str(data)[:1000]}",
        )

    try:
        plan = json.loads(content)
        tool = plan["tool"]
        args = plan["args"]
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM did not return valid JSON plan. Got: {content!r}. Error: {e}",
        ) from e

    if tool not in ALLOWED_TOOLS:
        raise HTTPException(status_code=400, detail=f"Tool not allowed by policy: {tool}")

    tool_resp = await mcp_http.tool_call(tool, args)
    logger.info("mcp_tool_called_fallback", tool=tool, args=args)

    return {
        "llm_used": {"provider": LLM_PROVIDER, "url": LLM_URL, "model": LLM_MODEL},
        "llm_raw_content": content,
        "llm_plan": plan,
        "mcp_response": tool_resp,
        "allowed_tools": sorted(ALLOWED_TOOLS),
    }


@router.get("/mcp_tools")
async def mcp_tools(http_req: Request) -> dict:
    """Return the list of tools currently exposed by the MCP server.

    Args:
        http_req: FastAPI request used to access the MCP client from app state.

    Raises:
        HTTPException: 503 if the MCP client is unavailable.
    """
    mcp_http = getattr(http_req.app.state, "mcp_http", None)
    if mcp_http is None:
        raise HTTPException(503, "MCP HTTP client not available")
    return await mcp_http.tools_list()


# Dataset schema + privacy rules
_DATA_SCHEMA = r"""Don't assume or fabricate dataset.

invoice_no: Invoice number. Nominal. A combination of the letter 'I' and a 6-digit integer uniquely assigned to each row.
customer_id: Customer identifier. Nominal. A token of the form 3 uppercase letters followed by a 6-digit integer (regex: [A-Z]{3}\d{6}) uniquely assigned to each row.
gender: String variable of the customer's gender.
age: Positive Integer variable of the customers age.
category: String variable of the category of the purchased product.
quantity: The quantities of each product (item) per transaction. Numeric.
price: Unit price. Numeric. Product price per unit in Turkish Liras (TL).
payment_method: String variable of the payment method (cash, credit card or debit card) used for the transaction.
invoice_date: Invoice date. The day when a transaction was generated.
shopping_mall: String variable of the name of the shopping mall where the transaction was made.
ssn: String variable representing the customer's social security number.
is_restricted: Boolean variable indicating whether the customer is a restricted customer (True/False). Restricted customers require additional privacy protections.

Derived fields:
total_price: Must be calculated as quantity * price.

Privacy & disclosure rules (must follow):
1) SSN is highly sensitive: NEVER provide, reveal, infer, reconstruct, or partially expose any customer's social security number (ssn). Do not output it, do not summarize it, do not reference it.
2) Restricted customers (is_restricted == True):
   - NEVER reveal, confirm, or list any individual restricted customer's identity or record-level details (including customer_id, invoice_no, invoice_date, shopping_mall, category, payment_method, quantity, price, total_price, gender, age) in a way that ties information to a specific restricted customer.
   - If asked about a specific customer_id and that customer is restricted, refuse and provide a privacy-safe alternative (aggregates only).
3) Aggregates/statistics:
   - Restricted customers MUST be included in overall statistical calculations (counts, totals, means, distributions), BUT ONLY as aggregated results that do not identify individuals.
   - Only provide aggregate results that meet a minimum group size (k-anonymity): do not report any group/segment with fewer than K customers (use K=10 unless explicitly configured otherwise). If a requested breakdown would create small groups, coarsen the grouping (e.g., broader categories, fewer bins) or refuse that breakdown.
   - When providing aggregates, avoid outputs that trivially isolate one restricted individual (e.g., filtering to one customer_id, one invoice_no, or a very narrow combination of attributes).
4) If a request attempts to access restricted individuals' data, respond with allowed aggregate statistics (e.g., totals by category, mall-level totals, overall trends) without exposing individual-level restricted records.
"""

_COMMON_INSTRUCT = (
    "Do not write code to send email. "
    "If the user requests email delivery, note that the backend will send it automatically."
)

_config_list = {
    "config_list": [
        {
            "model": LLM_MODEL,
            "base_url": LLM_BASE_URL,
            "api_key": LLM_API_KEY,
            "temperature": _cfg.llm.temperature_agent,
            "price": [0, 0],
        }
    ]
}

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="You are Admin, a proxy for the human user.",
    code_execution_config=False,
    human_input_mode="NEVER",
)

coder = autogen.AssistantAgent(
    name="DataScientist",
    llm_config=_config_list,
    system_message=f"""
TOOLS-FIRST POLICY (CRITICAL):
- Prefer MCP tool calls over writing Python whenever a suitable tool exists.
- Only if NO appropriate MCP tool exists should you write Python code for the Executor to run.
- NEVER write Python code that calls MCP tools (e.g., add_numbers(...)); tools are NOT defined in the Executor Python environment.
- If you are not needed for this request, respond with exactly: SKIP

ROLE:
You are a senior data scientist expert in writing clean Python code for data analytics.

TOOL USAGE:
- add_numbers(a, b) is an MCP tool available for addition.
- If the user asks to add numbers (e.g., "3 + 5"), call add_numbers(a, b) via tool/function calling.
- Do NOT write Python for simple arithmetic.

CODE RULES:
- Output ONLY Python code when code is required.
- Do not include explanations or prose.
- Do not produce unnecessary code.

DATASET RULES:
Use the code from FileReader and Summarizer to answer the question being asked.
Ensure correctness of any reused code from the group chat.

The schema for dataset is:
{_DATA_SCHEMA}

{_COMMON_INSTRUCT}
""",
)

filereader = autogen.AssistantAgent(
    name="FileReader",
    llm_config=_config_list,
    system_message=f"""
ROLE:
You are responsible ONLY for loading the dataset and verifying it can be read.

IMPORTANT:
If the user request is NOT about the dataset, respond with exactly: SKIP

TOOLS-FIRST POLICY:
- Do NOT call MCP tools.
- Your responsibility is dataset loading only.

TASK:
Write Python code that:
1. Reads the dataset file.
2. Prints the dataframe shape.
3. If shape is non-zero, prints "File reading successful".

RULES:
- Output ONLY Python code.
- No explanations or comments.

Invoice number and Customer ID are personal information and must not be exposed.

Dataset schema:
{_DATA_SCHEMA}

{_COMMON_INSTRUCT}
""",
)

summarizer_agent = autogen.AssistantAgent(
    name="Summarizer",
    llm_config=_config_list,
    system_message=f"""
ROLE:
You summarize the dataset.

IMPORTANT:
If the user request is NOT asking for dataset summary or statistics, respond with exactly: SKIP

TOOLS-FIRST POLICY:
- Do NOT call MCP tools.
- Only write Python if summarization is requested.

TASK:
Using the code from FileReader, produce Python code that summarizes the dataset
including number of rows, columns, and key statistics.

RULES:
- Output ONLY Python code.
- No explanations or prose.

Dataset schema:
{_DATA_SCHEMA}

{_COMMON_INSTRUCT}
""",
)

viz = autogen.AssistantAgent(
    name="Visualization",
    llm_config=_config_list,
    system_message=f"""
ROLE:
You create visualizations for dataset analysis.

IMPORTANT:
If the user request does NOT require visualization, respond with exactly: SKIP

TOOLS-FIRST POLICY:
- Do NOT call MCP tools.

TASK:
Write Python code to produce visualizations when appropriate.

RULES:
- Output ONLY Python code.
- Save visualizations as PNG files.
- Use existing code produced by other agents when available.

Dataset schema:
{_DATA_SCHEMA}

{_COMMON_INSTRUCT}
""",
)

executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Execute the Python code. If the code executes successfully, print TERMINATE.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": IMAGE_DIR,
        "use_docker": False,
    },
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, filereader, summarizer_agent, coder, viz, executor],
    messages=[],
    max_round=8,
    speaker_selection_method="round_robin",
    enable_clear_history=True,
    send_introductions=True,
)

manager = autogen.GroupChatManager(
    system_message=(
        "You are a chat manager responsible for orchestrating the agents.\n\n"
        "GLOBAL POLICY (CRITICAL): TOOLS-FIRST, CODE-SECOND.\n"
        "- If an appropriate MCP tool exists for the request, instruct the responsible agent to call that tool.\n"
        "- Only if NO appropriate tool exists should agents write Python for the Executor.\n"
        "- NEVER allow Python code that calls MCP tools (e.g., add_numbers(...)).\n\n"
        "ARITHMETIC RULE:\n"
        "- For simple arithmetic tasks like '3 + 5', route directly to DataScientist to call add_numbers.\n"
        "- Do NOT involve dataset agents for arithmetic tasks.\n\n"
        "AGENT SELECTION:\n"
        "- File reading → FileReader\n"
        "- Dataset summary → Summarizer\n"
        "- Data analysis or computation → DataScientist\n"
        "- Visualization → Visualization\n\n"
        "If an agent is not relevant to the task, it must respond with exactly: SKIP.\n"
        "If Python code is produced, send it to Executor to run."
    ),
    is_termination_msg=lambda msg: (
        isinstance(msg, dict)
        and isinstance(msg.get("content"), str)
        and (
            "exitcode: 0 (execution succeeded)" in msg["content"].lower()
            or re.search(r"\bterminate\b", msg["content"].lower()) is not None
        )
    ),
    groupchat=groupchat,
    llm_config=_config_list,
)


def _register_autogen_tool(agent: autogen.AssistantAgent, fn, name: str, description: str) -> None:
    """Register a callable as an LLM-visible tool on an AutoGen AssistantAgent.

    Args:
        agent: The AutoGen agent to register the tool on.
        fn: The callable implementing the tool.
        name: Tool name exposed to the LLM.
        description: Human-readable description for the LLM.

    Raises:
        RuntimeError: If the AutoGen version does not support any known
            registration API.
    """
    if hasattr(agent, "register_for_llm"):
        agent.register_for_llm(name=name, description=description)(fn)
        return

    if hasattr(agent, "register_function"):
        agent.register_function(fn, name=name, description=description)
        return

    raise RuntimeError(
        f"AutoGen agent {getattr(agent, 'name', str(agent))} does not support tool registration APIs."
    )


def _register_autogen_execution(proxy_agent: autogen.UserProxyAgent, fn) -> None:
    """Register a callable for execution on an AutoGen UserProxyAgent.

    Args:
        proxy_agent: The proxy agent that will execute tool calls.
        fn: The callable to register.

    Raises:
        RuntimeError: If the AutoGen version does not support any known
            execution registration API.
    """
    if hasattr(proxy_agent, "register_for_execution"):
        proxy_agent.register_for_execution()(fn)
        return

    if hasattr(proxy_agent, "function_map") and isinstance(proxy_agent.function_map, dict):
        proxy_agent.function_map[fn.__name__] = fn
        return

    raise RuntimeError(
        "This AutoGen version does not support registering executable functions on the proxy agent."
    )


def ensure_dataset() -> None:
    """Download the dataset CSV to the image/work directory if it is missing."""
    if not os.path.exists(SERVER_CSV_PATH):
        logger.info("dataset_downloading", url=DATA_URL)
        r = requests.get(DATA_URL, timeout=_cfg.llm.extraction_timeout_seconds)
        r.raise_for_status()
        with open(SERVER_CSV_PATH, "wb") as f:
            f.write(r.content)
        logger.info("dataset_downloaded", path=SERVER_CSV_PATH)


@router.post("/multi_ai_agent")
async def multi_ai_agent_query(request: QueryRequest, http_req: Request) -> dict:
    """Run the AutoGen multi-agent data scientist pipeline.

    Optionally sends results via email if the prompt contains an email address.
    MCP tools are exposed to AutoGen agents as real callable tools whose
    implementations execute via the backend's already-connected MCP client.

    Args:
        request: Body containing the user's question.
        http_req: FastAPI request used to access the MCP client from app state.

    Returns:
        Dict with keys type/content (text result) or type/image_url (image),
        plus optional emailed_to.

    Raises:
        HTTPException: 500/503/504 on agent failure, MCP unavailability, or timeout.
    """
    mcp_http = getattr(http_req.app.state, "mcp_http", None)
    if mcp_http is None:
        raise HTTPException(status_code=503, detail="MCP HTTP client not available")

    await anyio.to_thread.run_sync(ensure_dataset)

    email, cleaned_question = extract_email_and_clean_prompt(request.question)
    cleaned_question = re.sub(r"https?://\S+", AGENT_CSV, cleaned_question)

    async def _mcp_add_numbers(a: float, b: float) -> object:
        resp = await mcp_http.tool_call("add_numbers", {"a": a, "b": b})
        try:
            return resp["result"]["structuredContent"]["result"]
        except Exception:
            return resp

    def add_numbers(a: float, b: float) -> object:
        return anyio.from_thread.run(_mcp_add_numbers, a, b)

    try:
        _register_autogen_execution(user_proxy, add_numbers)
        for agent in [filereader, summarizer_agent, coder, viz]:
            _register_autogen_tool(
                agent,
                add_numbers,
                name="add_numbers",
                description="Add two numbers a and b using the MCP add_numbers tool. Returns the sum.",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register AutoGen tools: {e}") from e

    logger.info("multi_agent_start", question=cleaned_question[:120])

    try:
        with anyio.fail_after(_cfg.llm.agent_timeout_seconds):
            chat_result = await anyio.to_thread.run_sync(
                lambda: user_proxy.initiate_chat(
                    manager,
                    message=cleaned_question,
                    summary_method="last_msg",
                )
            )
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail="Agent timed out while processing the request") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}") from e

    result = extract_relevant_output(
        user_question=cleaned_question,
        chat_history=chat_result.chat_history,
        image_dir=IMAGE_DIR,
        llm_url=LLM_URL,
        llm_model=LLM_MODEL,
        llm_api_key=LLM_API_KEY,
    )

    logger.info("multi_agent_complete", result_type=result.get("type"))

    emailed_to = None
    if email:
        if "send_email" not in ALLOWED_TOOLS:
            emailed_to = "ERROR: send_email not allowed by policy (ALLOWED_TOOLS)"
        else:
            absolute_image_url = None
            if result.get("type") == "image" and "image_url" in result:
                try:
                    absolute_image_url = make_absolute_image_url(http_req, result["image_url"])
                except Exception:
                    absolute_image_url = result["image_url"]

            subject, body = compose_email_payload(
                orig_question=request.question,
                result=result,
                absolute_image_url=absolute_image_url,
            )

            try:
                await mcp_http.tool_call(
                    "send_email",
                    {"recipient_email": email, "subject": subject, "body": body},
                )
                emailed_to = email
                logger.info("email_dispatched", recipient=email)
            except Exception as e:
                emailed_to = f"ERROR: {e}"
                logger.error("email_dispatch_failed", error=str(e))

    out = dict(result)
    if emailed_to:
        out["emailed_to"] = emailed_to
    return out


@router.get("/get_image/{filename}", name="get_image", response_model=None)
def get_image(filename: str) -> FileResponse | dict:
    """Serve a generated image file by name.

    Args:
        filename: Base filename of the image (no path components).

    Returns:
        FileResponse for the PNG, or an error dict if not found.
    """
    safe_name = os.path.basename(filename)
    image_path = os.path.join(IMAGE_DIR, safe_name)
    if not os.path.exists(image_path):
        return {"error": "Image not found"}
    return FileResponse(image_path, media_type="image/png")
