from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# ----------------------------
# FastAPI router
# ----------------------------
rag_router = APIRouter()

# ----------------------------
# LlamaIndex settings
# ----------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5"
)

_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

if _LLM_PROVIDER == "openai":
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    Settings.llm = LlamaOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        api_base=os.getenv("LLM_BASE_URL") or None,
        temperature=0.3,
    )
else:
    from llama_index.llms.ollama import Ollama
    Settings.llm = Ollama(
        model="gpt-oss:20b",
        options={"temperature": 0.3},
        request_timeout=460,
    )

# ----------------------------
# Load or build vector index
# ----------------------------
DATA_DIR = "data"
INDEX_DIR = "storage"

if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
    # Fast path: load persisted index
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
else:
    # Slow path: build index once
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=INDEX_DIR)


query_engine = index.as_query_engine()


# ----------------------------
# Request model
# ----------------------------
class QueryRequest(BaseModel):
    question: str

# ----------------------------
# RAG endpoint
# ----------------------------
@rag_router.post("/rag-query")
def rag_info(request: QueryRequest):
    try:
        response = query_engine.query(request.question)
        return {"response": response.response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


