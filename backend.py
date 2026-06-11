"""FastAPI application entry point.

Starts the API server, registers routers, manages the MCP client
connection lifecycle, and serves PDF reports.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import structlog
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse

from config import configure_logging, get_config
from mcp_http_client import MCPHttpClient
from multi_ai_agent import router as multi_ai_agent_router

# from summarizer import summarizer_router
# from rag import rag_router

load_dotenv()
configure_logging()

logger = structlog.get_logger(__name__)

_cfg = get_config()

mcp_port = os.getenv("MCP_PORT", "8005")
mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"
mcp_http = MCPHttpClient(mcp_url)

_REPORTS_DIR = Path("reports")
_REPORTS_DIR.mkdir(exist_ok=True)

_REPORTS: dict[str, str] = {
    "Transformer, LLM, RAG and Multi-Model": "Report1.pdf",
    "Agentic AI: A complete guide": "Report2.pdf",
}

_PITCH_DECK_FILE = "genAI-insights-hub.pdf"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MCP client connection for the application lifetime.

    MCP connection failure is non-fatal: the backend starts normally and
    logs a warning. Only MCP-dependent features (e.g. email sending) will
    be unavailable until the MCP server is running.
    """
    logger.info("mcp_client_connecting", url=mcp_url)
    try:
        with anyio.fail_after(_cfg.mcp.initialize_timeout_seconds):
            await mcp_http.initialize()
        app.state.mcp_http = mcp_http
        logger.info("mcp_client_ready")
    except Exception as exc:
        logger.warning(
            "mcp_client_unavailable",
            error=str(exc),
            detail="MCP-dependent features (email) will be unavailable.",
        )
        app.state.mcp_http = None
    yield
    logger.info("mcp_client_shutdown")


app = FastAPI(root_path="/api", lifespan=lifespan)

app.include_router(multi_ai_agent_router)
# app.include_router(summarizer_router)
# app.include_router(rag_router)


@app.get("/list_reports")
def list_reports() -> dict:
    """Return the names of all available PDF reports.

    Returns:
        A dict with key ``reports`` containing a list of report names.
    """
    return {"reports": list(_REPORTS.keys())}


@app.get("/view_report/{report_name}", response_model=None)
def view_report(report_name: str) -> FileResponse | dict:
    """Serve a PDF report by name.

    Args:
        report_name: The human-readable report name from list_reports.

    Returns:
        A FileResponse streaming the PDF, or an error dict if not found.
    """
    if report_name not in _REPORTS:
        return {"error": "Report not found"}
    file_path = _REPORTS_DIR / _REPORTS[report_name]
    return FileResponse(file_path, media_type="application/pdf")


@app.get("/view_pitch_deck", response_model=None)
def view_pitch_deck() -> FileResponse | dict:
    """Serve the pitch deck PDF.

    Returns:
        A FileResponse streaming the PDF, or an error dict if not found.
    """
    file_path = _REPORTS_DIR / _PITCH_DECK_FILE
    if not file_path.exists():
        return {"error": "Pitch Deck not found"}
    return FileResponse(file_path, media_type="application/pdf")
