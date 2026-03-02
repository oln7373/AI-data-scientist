from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv

from multi_ai_agent import router as multi_ai_agent_router
from summarizer import summarizer_router
from rag import rag_router

from mcp_manager import MCPManager

load_dotenv()

mcp_port = os.getenv("MCP_PORT", "8005")
mcp_url = f"http://127.0.0.1:{mcp_port}/mcp"
mcp = MCPManager(mcp_url)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await mcp.start()
    app.state.mcp = mcp
    try:
        yield
    finally:
        await mcp.stop()

app = FastAPI(root_path="/api", lifespan=lifespan)

app.include_router(multi_ai_agent_router)
app.include_router(summarizer_router)
app.include_router(rag_router)

# everything below this can stay the same:
os.makedirs("reports", exist_ok=True)

reports = {
    "Transformer, LLM, RAG and Multi-Model": "Report1.pdf",
    "Agentic AI: A complete guide": "Report2.pdf"
}

pitch_deck_file = "genAI-insights-hub.pdf"

@app.get("/list_reports")
def list_reports():
    return {"reports": list(reports.keys())}

@app.get("/view_report/{report_name}")
def view_report(report_name: str):
    if report_name not in reports:
        return {"error": "Report not found"}
    file_path = os.path.join("reports", reports[report_name])
    return FileResponse(file_path, media_type="application/pdf")

@app.get("/view_pitch_deck")
def view_pitch_deck():
    file_path = os.path.join("reports", pitch_deck_file)
    if not os.path.exists(file_path):
        return {"error": "Pitch Deck not found"}
    return FileResponse(file_path, media_type="application/pdf")
