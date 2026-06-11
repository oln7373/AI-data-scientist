"""Retrieval-Augmented Generation (RAG) pipeline API."""

import os

import structlog
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel

from config import get_config

load_dotenv()

logger = structlog.get_logger(__name__)

rag_router = APIRouter()

_cfg = get_config()

Settings.embed_model = HuggingFaceEmbedding(model_name=_cfg.rag.embed_model)

_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

if _LLM_PROVIDER == "openai":
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    Settings.llm = LlamaOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        api_base=os.getenv("LLM_BASE_URL") or None,
        temperature=_cfg.llm.temperature_rag,
    )
else:
    from llama_index.llms.ollama import Ollama
    Settings.llm = Ollama(
        model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
        options={"temperature": _cfg.llm.temperature_rag},
        request_timeout=_cfg.llm.rag_request_timeout_seconds,
    )

_index_dir = _cfg.rag.index_dir
_data_dir = _cfg.rag.data_dir

if os.path.exists(_index_dir) and os.listdir(_index_dir):
    logger.info("rag_index_loading", index_dir=_index_dir)
    storage_context = StorageContext.from_defaults(persist_dir=_index_dir)
    index = load_index_from_storage(storage_context)
else:
    logger.info("rag_index_building", data_dir=_data_dir)
    documents = SimpleDirectoryReader(_data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=_index_dir)
    logger.info("rag_index_persisted", index_dir=_index_dir)

query_engine = index.as_query_engine()


class QueryRequest(BaseModel):
    """Request body for the RAG query endpoint."""

    question: str


@rag_router.post("/rag-query")
def rag_info(request: QueryRequest) -> dict:
    """Answer a question using the RAG pipeline.

    Args:
        request: Body containing the question string.

    Returns:
        Dict with key ``response`` containing the generated answer.

    Raises:
        HTTPException: 500 on any retrieval or generation failure.
    """
    try:
        response = query_engine.query(request.question)
        return {"response": response.response}
    except Exception as e:
        logger.error("rag_query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
