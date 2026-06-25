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
from llm_client import (
    AWS_REGION,
    BEDROCK_BASE_URL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    _BedRockSigV4Auth,
)

load_dotenv()

logger = structlog.get_logger(__name__)

rag_router = APIRouter()

_cfg = get_config()

Settings.embed_model = HuggingFaceEmbedding(model_name=_cfg.rag.embed_model)

from llama_index.llms.openai import OpenAI as LlamaOpenAI

if LLM_PROVIDER == "bedrock":
    import httpx as _httpx

    try:
        Settings.llm = LlamaOpenAI(
            model=LLM_MODEL,
            api_key="bedrock",
            api_base=BEDROCK_BASE_URL,
            temperature=_cfg.llm.temperature_rag,
            http_client=_httpx.Client(auth=_BedRockSigV4Auth(AWS_REGION), trust_env=False),
        )
        logger.info("rag_llm_bedrock", model=LLM_MODEL, region=AWS_REGION)
    except TypeError:
        logger.warning(
            "rag_bedrock_http_client_unsupported",
            detail="Installed llama-index-llms-openai does not support http_client; "
            "RAG queries will fail auth against Bedrock. "
            "Install llama-index-llms-bedrock-converse for full support.",
        )
        Settings.llm = LlamaOpenAI(
            model=LLM_MODEL,
            api_key="bedrock",
            api_base=BEDROCK_BASE_URL,
            temperature=_cfg.llm.temperature_rag,
        )
else:
    Settings.llm = LlamaOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        api_base=LLM_BASE_URL,
        temperature=_cfg.llm.temperature_rag,
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
