from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
from multi_ai_agent import router as multi_ai_agent_router  # ✅ Import Multi-AI-Agent API
from summarizer import summarizer_router  # ✅ Import Summarizer API
from rag import rag_router  # Import the RAG router from rag.py


app = FastAPI(root_path="/api")

# ✅ Include Multi-AI-Agent Routes
app.include_router(multi_ai_agent_router)

# ✅ Include Summarizer Routes
app.include_router(summarizer_router)

# Include the RAG router
app.include_router(rag_router)


# Ensure 'reports' directory exists
os.makedirs("reports", exist_ok=True)

# Define available reports
reports = {
    "Transformer, LLM, RAG and Multi-Model": "Report1.pdf",
    "Agentic AI: A complete guide": "Report2.pdf"
}

pitch_deck_file = "genAI-insights-hub.pdf"


@app.get("/list_reports")
def list_reports():
    """Returns a list of available reports with user-friendly names."""
    return {"reports": list(reports.keys())}

@app.get("/view_report/{report_name}")
def view_report(report_name: str):
    """Serves the selected report as a browser-viewable PDF."""
    if report_name not in reports:
        return {"error": "Report not found"}
    
    file_path = os.path.join("reports", reports[report_name])
    return FileResponse(file_path, media_type="application/pdf")

@app.get("/view_pitch_deck")
def view_pitch_deck():
    """Serves the Pitch Deck PDF."""
    file_path = os.path.join("reports", pitch_deck_file)
    if not os.path.exists(file_path):
        return {"error": "Pitch Deck not found"}
    return FileResponse(file_path, media_type="application/pdf")
