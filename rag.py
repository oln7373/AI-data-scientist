from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

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


