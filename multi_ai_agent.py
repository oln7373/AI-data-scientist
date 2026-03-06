"""
multi_ai_agent.py

This file provides:
1) /mcp_add_demo      (simple MCP tool demo)
2) /mcp_agent_add     (LLM tool-calling via Ollama -> allowlist -> MCP)
3) /mcp_tools         (list tools from MCP server)
4) /multi_ai_agent    (RESTORED AutoGen multi-agent "data scientist" pipeline,
                       with TRUE tool-calling via MCP-backed AutoGen tools,
                       and OPTIONAL email sending via MCP send_email tool)

Design:
- Keep the old AutoGen pipeline behavior (code-gen + Executor).
- Keep MCP allowlist enforcement on the backend.
- Email side-effects go through MCP (send_email tool), not direct Gmail code.
- For /multi_ai_agent: expose MCP tools to AutoGen as real callable tools.
  The tool implementation bounces from AutoGen's worker thread back to the
  FastAPI event loop using anyio.from_thread.run(...)
"""

import os
import re
import json
import httpx
import requests
import anyio

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from response_extractor import extract_relevant_output

# AutoGen (old multi-agent pipeline)
import autogen

# Email helpers (parsing + formatting)
from email_utils import (
    extract_email_and_clean_prompt,
    compose_email_payload,
    make_absolute_image_url,
)

load_dotenv()

router = APIRouter()

# -----------------------------
# Backend policy + Ollama config
# -----------------------------
ALLOWED_TOOLS = set(
    t.strip()
    for t in os.getenv("ALLOWED_TOOLS", "").split(",")
    if t.strip()
)

if not ALLOWED_TOOLS:
    raise RuntimeError("ALLOWED_TOOLS not configured in .env")

OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_URL = f"http://localhost:{OLLAMA_PORT}/v1/chat/completions"


class QueryRequest(BaseModel):
    question: str


# -----------------------------
# MCP demo endpoints (keep)
# -----------------------------
@router.post("/mcp_add_demo")
async def mcp_add_demo(request: QueryRequest, http_req: Request):
    mcp_http = getattr(http_req.app.state, "mcp_http", None)
    if mcp_http is None:
        raise HTTPException(status_code=503, detail="MCP HTTP client not available")

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*\+\s*(-?\d+(?:\.\d+)?)", request.question)
    if not m:
        raise HTTPException(status_code=400, detail="Provide a question containing something like '5 + 7'")

    a = float(m.group(1))
    b = float(m.group(2))

    # Allowlist check
    if "add_numbers" not in ALLOWED_TOOLS:
        raise HTTPException(status_code=400, detail="Tool not allowed by policy: add_numbers")

    return await mcp_http.tool_call("add_numbers", {"a": a, "b": b})


@router.post("/mcp_agent_add")
async def mcp_agent_add(request: QueryRequest, http_req: Request):
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

    async with httpx.AsyncClient(trust_env=False, timeout=60.0) as client:
        resp = await client.post(
            OLLAMA_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": OLLAMA_MODEL,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": request.question},
                ],
            },
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama error {resp.status_code}: {resp.text[:1000]}")

    data = resp.json()

    try:
        msg = data["choices"][0]["message"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Unexpected Ollama response shape: {e}. Raw={str(data)[:1000]}")

    # ---- Preferred: OpenAI-style tool_calls ----
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        try:
            fn = tool_calls[0]["function"]["name"]
            arg_str = tool_calls[0]["function"].get("arguments", "{}")
            args = json.loads(arg_str) if isinstance(arg_str, str) else (arg_str or {})
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to parse tool_calls: {e}. Raw={str(tool_calls)[:1000]}")

        if fn not in ALLOWED_TOOLS:
            raise HTTPException(status_code=400, detail=f"Tool not allowed by policy: {fn}")

        tool_resp = await mcp_http.tool_call(fn, args)

        return {
            "ollama_used": {"url": OLLAMA_URL, "model": OLLAMA_MODEL},
            "llm_finish_reason": data["choices"][0].get("finish_reason"),
            "llm_tool_call": {"tool": fn, "args": args},
            "mcp_response": tool_resp,
            "allowed_tools": sorted(ALLOWED_TOOLS),
        }

    # ---- Fallback: JSON in content ----
    content = (msg.get("content") or "").strip()
    if not content:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned neither tool_calls nor content. Raw response: {str(data)[:1000]}",
        )

    try:
        plan = json.loads(content)
        tool = plan["tool"]
        args = plan["args"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM did not return valid JSON plan. Got: {content!r}. Error: {e}")

    if tool not in ALLOWED_TOOLS:
        raise HTTPException(status_code=400, detail=f"Tool not allowed by policy: {tool}")

    tool_resp = await mcp_http.tool_call(tool, args)

    return {
        "ollama_used": {"url": OLLAMA_URL, "model": OLLAMA_MODEL},
        "llm_raw_content": content,
        "llm_plan": plan,
        "mcp_response": tool_resp,
        "allowed_tools": sorted(ALLOWED_TOOLS),
    }


@router.get("/mcp_tools")
async def mcp_tools(http_req: Request):
    mcp_http = getattr(http_req.app.state, "mcp_http", None)
    if mcp_http is None:
        raise HTTPException(503, "MCP HTTP client not available")
    return await mcp_http.tools_list()


# ============================================================
# RESTORED: Multi-agent "Agentic AI Data Scientist" pipeline
# ============================================================

# -----------------------------
# Dataset plumbing (old behavior)
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/oln7373/AI-data-scientist/refs/heads/main/customer_shopping_data.csv"

IMAGE_DIR = "data"
os.makedirs(IMAGE_DIR, exist_ok=True)

AGENT_CSV = "customer_shopping_data.csv"
SERVER_CSV_PATH = os.path.join(IMAGE_DIR, AGENT_CSV)


def ensure_dataset():
    """
    Download dataset to data/customer_shopping_data.csv so that
    AutoGen's Executor (work_dir=data) can read it by filename.
    """
    if not os.path.exists(SERVER_CSV_PATH):
        r = requests.get(DATA_URL, timeout=30)
        r.raise_for_status()
        with open(SERVER_CSV_PATH, "wb") as f:
            f.write(r.content)


# Dataset schema + privacy rules (kept from your old pipeline)
data_schema = r"""Don't assume or fabricate dataset.

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
4) If a request attempts to access restricted individuals’ data, respond with allowed aggregate statistics (e.g., totals by category, mall-level totals, overall trends) without exposing individual-level restricted records.
"""

common_instruct = (
    "Do not write code to send email. "
    "If the user requests email delivery, note that the backend will send it automatically."
)

# -----------------------------
# AutoGen: LLM config (Ollama OpenAI-compatible endpoint)
# -----------------------------
config_list = {
    "config_list": [
        {
            "model": OLLAMA_MODEL,
            "base_url": f"http://localhost:{OLLAMA_PORT}/v1",
            "api_key": "ollama",
            "temperature": 0.0,
            "price": [0, 0],
        }
    ]
}

# -----------------------------
# AutoGen agents (tools-first policy)
# -----------------------------
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="You are Admin, a proxy for the human user.",
    code_execution_config=False,
    human_input_mode="NEVER",
)

coder = autogen.AssistantAgent(
    name="DataScientist",
    llm_config=config_list,
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
{data_schema}

{common_instruct}
""",
)

filereader = autogen.AssistantAgent(
    name="FileReader",
    llm_config=config_list,
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
{data_schema}

{common_instruct}
""",
)

summarizer_agent = autogen.AssistantAgent(
    name="Summarizer",
    llm_config=config_list,
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
{data_schema}

{common_instruct}
""",
)

viz = autogen.AssistantAgent(
    name="Visualization",
    llm_config=config_list,
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
{data_schema}

{common_instruct}
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
    llm_config=config_list,
)




def _register_autogen_tool(agent, fn, name: str, description: str):
    """
    AutoGen APIs differ across versions. This tries the common patterns.
    """
    # register_for_llm pattern
    if hasattr(agent, "register_for_llm"):
        agent.register_for_llm(name=name, description=description)(fn)
        return

    # Some versions may support a generic register_function on agent
    if hasattr(agent, "register_function"):
        agent.register_function(fn, name=name, description=description)
        return

    raise RuntimeError(f"AutoGen agent {getattr(agent, 'name', str(agent))} does not support tool registration APIs.")


def _register_autogen_execution(proxy_agent, fn):
    """
    Register a function for execution. API differs across versions.
    """
    if hasattr(proxy_agent, "register_for_execution"):
        proxy_agent.register_for_execution()(fn)
        return

    # Some versions expose a function map dict
    if hasattr(proxy_agent, "function_map") and isinstance(proxy_agent.function_map, dict):
        proxy_agent.function_map[fn.__name__] = fn
        return

    # Last resort: do nothing (LLM may still "call" but it won't execute)
    raise RuntimeError("This AutoGen version does not support registering executable functions on the proxy agent.")


@router.post("/multi_ai_agent")
async def multi_ai_agent_query(request: QueryRequest, http_req: Request):
    """
    Restored old multi-agent pipeline.

    Optional behavior:
    - If user includes an email address in the prompt, we will email the result
      via MCP send_email tool (ONLY if allowed by ALLOWED_TOOLS).

    Tool-calling behavior:
    - Exposes MCP tools (currently: add_numbers) to AutoGen as real tools.
    - Tool implementation runs via the backend's already-connected MCP client.
    """
    # MCP client must exist
    mcp_http = getattr(http_req.app.state, "mcp_http", None)
    if mcp_http is None:
        raise HTTPException(status_code=503, detail="MCP HTTP client not available")

    # Ensure dataset without blocking loop
    await anyio.to_thread.run_sync(ensure_dataset)

    # Extract email + clean question
    email, cleaned_question = extract_email_and_clean_prompt(request.question)

    # Rewrite any URL into agent-visible filename
    cleaned_question = re.sub(r"https?://\S+", AGENT_CSV, cleaned_question)

    # ----------------------------
    # MCP-backed AutoGen tools
    # ----------------------------
    # async implementation that uses the already-connected MCP client
    async def _mcp_add_numbers(a: float, b: float):
        resp = await mcp_http.tool_call("add_numbers", {"a": a, "b": b})

        # Try to return a clean scalar for the LLM
        try:
            return resp["result"]["structuredContent"]["result"]
        except Exception:
            return resp

    # sync wrapper callable from AutoGen's worker thread
    def add_numbers(a: float, b: float):
        return anyio.from_thread.run(_mcp_add_numbers, a, b)

    # Register tool for THIS request lifecycle
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
        raise HTTPException(status_code=500, detail=f"Failed to register AutoGen tools: {e}")

    # Run AutoGen (blocking) in a thread with timeout
    try:
        with anyio.fail_after(180):
            chat_result = await anyio.to_thread.run_sync(
                lambda: user_proxy.initiate_chat(
                    manager,
                    message=cleaned_question,
                    summary_method="last_msg",
                )
            )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Agent timed out while processing the request")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}")

    # Decide what to return
    result = extract_relevant_output(
        user_question=cleaned_question,
        chat_history=chat_result.chat_history,
        image_dir=IMAGE_DIR,
        ollama_url=OLLAMA_URL,
        ollama_model=OLLAMA_MODEL,
    )

    # If email present, send via MCP (side-effect)
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
            except Exception as e:
                emailed_to = f"ERROR: {e}"

    out = dict(result)
    if emailed_to:
        out["emailed_to"] = emailed_to
    return out


@router.get("/get_image/{filename}", name="get_image")
def get_image(filename: str):
    safe_name = os.path.basename(filename)
    image_path = os.path.join(IMAGE_DIR, safe_name)
    if not os.path.exists(image_path):
        return {"error": "Image not found"}
    return FileResponse(image_path, media_type="image/png")