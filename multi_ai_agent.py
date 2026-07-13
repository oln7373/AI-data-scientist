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
from llm_client import (
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    BedrockAutoGenClient,
    get_async_client,
    get_autogen_llm_config,
)
from response_extractor import extract_relevant_output

load_dotenv()

logger = structlog.get_logger(__name__)

router = APIRouter()

_cfg = get_config()

ALLOWED_TOOLS = {t.strip() for t in os.getenv("ALLOWED_TOOLS", "").split(",") if t.strip()}

if not ALLOWED_TOOLS:
    raise RuntimeError("ALLOWED_TOOLS not configured in .env")

DATA_URL = _cfg.data.dataset_url
IMAGE_DIR = _cfg.data.image_dir
AGENT_CSV = _cfg.data.dataset_filename
SERVER_CSV_PATH = os.path.join(IMAGE_DIR, AGENT_CSV)

os.makedirs(IMAGE_DIR, exist_ok=True)

# Canonical descriptions for every MCP tool the system knows about.
# Only those present in ALLOWED_TOOLS are registered with AutoGen agents.
_MCP_TOOL_CATALOG: dict[str, str] = {
    "add_numbers": (
        "add_numbers(a, b) — Add two numbers together and return their sum. "
        "Use whenever the user asks to add, sum, or compute arithmetic."
    ),
    "ping": (
        "ping(message='hello') — Send a health-check ping to the MCP server. "
        "Returns 'pong: <message>'. Use for connectivity tests or when the user asks to ping."
    ),
    "select_data": (
        "select_data() — Return a sample list of customer IDs from the shopping dataset via MCP. "
        "Use when the user asks to select, fetch, or retrieve customer IDs or sample data."
    ),
    "compose_email": (
        "compose_email(recipient_email, subject, body) — Compose an email draft without sending it. "
        "Use to prepare the email content before calling send_email."
    ),
    "send_email": (
        "send_email(recipient_email, subject, body) — Send a plain-text email via MCP. "
        "Use when the user requests email delivery."
    ),
}

_available_tools_lines = "\n".join(
    f"- {desc}"
    for name, desc in _MCP_TOOL_CATALOG.items()
    if name in ALLOWED_TOOLS
)
_available_tools_text = (
    _available_tools_lines
    if _available_tools_lines
    else "- (No MCP tools are currently enabled in ALLOWED_TOOLS)"
)


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

    async_client = get_async_client()
    try:
        completion = await async_client.chat.completions.create(
            model=LLM_MODEL,
            temperature=_cfg.llm.temperature_agent,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": request.question},
            ],
            timeout=_cfg.llm.extraction_timeout_seconds,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}") from e

    msg = completion.choices[0].message
    tool_calls = msg.tool_calls or []

    if tool_calls:
        try:
            tc = tool_calls[0]
            fn = tc.function.name
            arg_str = tc.function.arguments
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
            "llm_used": {"base_url": LLM_BASE_URL, "model": LLM_MODEL},
            "llm_finish_reason": completion.choices[0].finish_reason,
            "llm_tool_call": {"tool": fn, "args": args},
            "mcp_response": tool_resp,
            "allowed_tools": sorted(ALLOWED_TOOLS),
        }

    content = (msg.content or "").strip()
    if not content:
        raise HTTPException(
            status_code=502,
            detail="LLM returned neither tool_calls nor content.",
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
        "llm_used": {"base_url": LLM_BASE_URL, "model": LLM_MODEL},
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

"""

_COMMON_INSTRUCT = (
    "Do not write code to send email. "
    "If the user requests email delivery, note that the backend will send it automatically."
)

if LLM_PROVIDER == "bedrock":
    autogen.AssistantAgent.register_model_client(model_client_cls=BedrockAutoGenClient)

_config_list = get_autogen_llm_config()

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
ROLE:
You are a senior data scientist and the primary MCP tool-caller in this group.

TOOLS-FIRST POLICY (CRITICAL):
- You MUST call an MCP tool via function/tool calling whenever a suitable tool exists for the request.
- Do NOT write Python code to do what an MCP tool can already do.
- Do NOT call MCP tools from Python code — they are not importable; use the tool-calling interface directly.
- Only write Python code when NO appropriate MCP tool exists (e.g., data analysis, visualisation, statistics).
- If you are not needed for this request, respond with exactly: SKIP

AVAILABLE MCP TOOLS (call these via tool/function calling, not Python):
{_available_tools_text}

WHEN TO CALL TOOLS vs WRITE CODE:
- Arithmetic / addition → call add_numbers
- Connectivity / health check / ping → call ping
- Retrieve customer IDs / sample data → call select_data
- Compose an email draft → call compose_email
- Send an email → call send_email
- Data analysis, statistics, charts on the dataset → write Python code

CODE RULES (only when writing Python):
- Output ONLY Python code, no explanations or prose.
- Do not produce unnecessary code.

DATASET RULES:
Use code from FileReader and Summarizer to answer the question being asked.
Ensure correctness of any reused code from the group chat.

The schema for the dataset is:
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
If the user request is NOT about reading or loading the dataset, respond with exactly: SKIP

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
If the user request is NOT asking for a dataset summary or statistics, respond with exactly: SKIP

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
If the user request does NOT require a chart or visualization, respond with exactly: SKIP

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
    speaker_selection_method="auto",
    enable_clear_history=True,
    send_introductions=True,
)

manager = autogen.GroupChatManager(
    system_message=(
        "You are a chat manager responsible for selecting the next speaker.\n\n"
        "ROUTING RULES — follow in order:\n"
        f"1. MCP TOOL TASKS: Any request that matches an available MCP tool must go to DataScientist.\n"
        f"   Available tools: {', '.join(t for t in _MCP_TOOL_CATALOG if t in ALLOWED_TOOLS)}.\n"
        "   Examples: add numbers → DataScientist, ping / health check → DataScientist, "
        "   retrieve customer IDs → DataScientist, compose or send email → DataScientist.\n"
        "2. DATASET LOADING: Reading / loading the CSV file → FileReader.\n"
        "3. DATASET SUMMARY: Summarising dataset statistics → Summarizer.\n"
        "4. CHARTS / VISUALISATIONS: Any plot or graph → Visualization.\n"
        "5. DATA ANALYSIS / COMPUTATION: Complex analytics requiring Python → DataScientist.\n"
        "6. CODE EXECUTION: After any agent produces Python code → Executor.\n\n"
        "KEY RULES:\n"
        "- Route MCP tool requests to DataScientist FIRST and IMMEDIATELY — do not involve other agents.\n"
        "- DataScientist calls tools via the tool-calling interface, NOT by writing Python.\n"
        "- Agents that are irrelevant to the current step respond with exactly: SKIP.\n"
        "- After Executor runs code successfully or prints TERMINATE, end the conversation."
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

    async def _mcp_invoke(tool_name: str, args: dict) -> object:
        """Route a tool call through the already-connected MCP client."""
        resp = await mcp_http.tool_call(tool_name, args)
        try:
            sc = resp["result"]["structuredContent"]
            for key in ("result", "output", "answer"):
                if key in sc:
                    return sc[key]
            return sc
        except Exception:
            return resp

    # Define a strongly-typed sync bridge for every known MCP tool.
    # Sync wrappers are required because AutoGen worker threads call tools
    # synchronously; anyio.from_thread.run bounces back to the event loop.

    def add_numbers(a: float, b: float) -> object:
        """Add two numbers via MCP and return their sum."""
        return anyio.from_thread.run(_mcp_invoke, "add_numbers", {"a": a, "b": b})

    def ping(message: str = "hello") -> object:
        """Ping the MCP server for a health or connectivity check."""
        return anyio.from_thread.run(_mcp_invoke, "ping", {"message": message})

    def select_data() -> object:
        """Retrieve a sample list of customer IDs from the shopping dataset via MCP."""
        return anyio.from_thread.run(_mcp_invoke, "select_data", {})

    def compose_email(recipient_email: str, subject: str, body: str) -> object:
        """Compose an email draft (does not send it)."""
        return anyio.from_thread.run(
            _mcp_invoke,
            "compose_email",
            {"recipient_email": recipient_email, "subject": subject, "body": body},
        )

    def send_email_tool(recipient_email: str, subject: str, body: str) -> object:
        """Send a plain-text email via MCP."""
        return anyio.from_thread.run(
            _mcp_invoke,
            "send_email",
            {"recipient_email": recipient_email, "subject": subject, "body": body},
        )

    _tool_fn_map: dict[str, tuple] = {
        "add_numbers": (add_numbers, _MCP_TOOL_CATALOG["add_numbers"]),
        "ping": (ping, _MCP_TOOL_CATALOG["ping"]),
        "select_data": (select_data, _MCP_TOOL_CATALOG["select_data"]),
        "compose_email": (compose_email, _MCP_TOOL_CATALOG["compose_email"]),
        "send_email": (send_email_tool, _MCP_TOOL_CATALOG["send_email"]),
    }

    try:
        for tool_name, (tool_fn, tool_desc) in _tool_fn_map.items():
            if tool_name not in ALLOWED_TOOLS:
                continue
            _register_autogen_execution(user_proxy, tool_fn)
            _register_autogen_tool(coder, tool_fn, name=tool_name, description=tool_desc)
            logger.info("autogen_tool_registered", tool=tool_name)
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
