"""PDF summarisation API using the configured LLM backend."""

import io
import os
from datetime import datetime

import numpy as np
import structlog
from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader
from rouge_score import rouge_scorer
import sacrebleu
import bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import get_config

load_dotenv()

logger = structlog.get_logger(__name__)

_cfg = get_config()

from openai import OpenAI as _OpenAIClient

_ollama_port = os.getenv("OLLAMA_PORT", "11434")
_openai_client = _OpenAIClient(
    api_key=(
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "ollama"
    ),
    base_url=(
        os.getenv("LLM_BASE_URL")
        or f"http://localhost:{_ollama_port}/v1"
    ),
)
_openai_model = (
    os.getenv("LLM_MODEL")
    or os.getenv("OPENAI_MODEL")
    or os.getenv("OLLAMA_MODEL")
    or "llama3.2"
)

summarizer_router = APIRouter()


def read_pdf_from_bytes(file_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes.

    Args:
        file_bytes: Raw PDF content.

    Returns:
        Concatenated text extracted from all pages.

    Raises:
        Exception: Re-raised from PdfReader on corrupt or unreadable input.
    """
    try:
        file_stream = io.BytesIO(file_bytes)
        reader = PdfReader(file_stream)
        text = "\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )
        logger.info("pdf_extracted", char_count=len(text))
        return text
    except Exception as e:
        logger.error("pdf_extraction_failed", error=str(e))
        raise


def chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split text into word-count-bounded chunks.

    Args:
        text: Input text to split.
        chunk_size: Maximum number of words per chunk.

    Returns:
        List of text chunks.
    """
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def generate_summary_prompt(text: str, summary_type: str) -> str:
    """Build a summarisation prompt for the given summary style.

    Args:
        text: The text to summarise.
        summary_type: One of abstractive, extractive, long, short.

    Returns:
        A formatted prompt string for the LLM.
    """
    prompt_templates = {
        "abstractive": f"Create an abstractive summary of the following text:\n{text}",
        "extractive": f"Extract the most important key sentences from the following text:\n{text}",
        "long": f"Generate a detailed and comprehensive summary of the following text:\n{text}",
        "short": f"Create a concise summary of the following text in a few sentences:\n{text}",
    }
    return prompt_templates.get(summary_type, prompt_templates["abstractive"])


def summarize_text_ollama(text: str, summary_type: str) -> str:
    """Generate a summary for a text chunk using the configured LLM.

    Args:
        text: Text chunk to summarise.
        summary_type: Summary style (abstractive, extractive, long, short).

    Returns:
        Generated summary string, or empty string on failure.
    """
    try:
        prompt = generate_summary_prompt(text, summary_type)
        response = _openai_client.chat.completions.create(
            model=_openai_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("summarization_failed", error=str(e))
        return ""


@summarizer_router.post("/summarize")
async def summarize_pdf(
    file: UploadFile = File(...),
    temperature: float = _cfg.llm.temperature_summarizer,
    max_tokens: int = 2048,
    summary_type: str = Query(
        "abstractive", enum=["abstractive", "extractive", "long", "short"]
    ),
) -> dict:
    """Summarise an uploaded PDF document.

    Args:
        file: Uploaded PDF file.
        temperature: LLM temperature (unused directly; forwarded to future LLM calls).
        max_tokens: Maximum tokens per chunk pass (controls chunk word count).
        summary_type: Summary style to apply.

    Returns:
        Dict with key ``summary`` containing the generated text.
    """
    try:
        logger.info("summarize_request", filename=file.filename)

        content = await file.read()
        logger.info("pdf_received", size_bytes=len(content))

        text = read_pdf_from_bytes(content)
        logger.info("pdf_text_extracted", char_count=len(text))

        # Temporary: persist received PDF for corruption debugging.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = f"saved_pdfs/{timestamp}.pdf"
        os.makedirs("saved_pdfs", exist_ok=True)
        with open(saved_path, "wb") as f:
            f.write(content)

        chunks = chunk_text(text, max_tokens)
        logger.info("chunks_created", chunk_count=len(chunks))

        chunk_summaries = [summarize_text_ollama(chunk, summary_type) for chunk in chunks]
        logger.info("chunks_summarized", summary_count=len(chunk_summaries))

        final_summary = " ".join(chunk_summaries)
        return {"summary": final_summary}

    except Exception as e:
        logger.error("summarize_endpoint_failed", error=str(e))
        return {"error": str(e)}


class SummaryEvaluationRequest(BaseModel):
    """Request body for the summary evaluation endpoint."""

    reference: str
    generated: str


def evaluate_summary(reference: str, generated: str) -> dict[str, float]:
    """Compute ROUGE, BLEU, BERTScore, and cosine similarity for a summary pair.

    Args:
        reference: Ground-truth reference summary.
        generated: Model-generated summary to evaluate.

    Returns:
        Dict mapping metric names to float scores.
    """
    scores: dict[str, float] = {}

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = rouge.score(reference, generated)
    scores["ROUGE-1"] = float(rouge_scores["rouge1"].fmeasure)
    scores["ROUGE-2"] = float(rouge_scores["rouge2"].fmeasure)
    scores["ROUGE-L"] = float(rouge_scores["rougeL"].fmeasure)

    scores["BLEU"] = float(sacrebleu.corpus_bleu([generated], [[reference]]).score)

    _, _, f1 = bert_score.score([generated], [reference], lang="en")
    scores["BERTScore"] = float(f1.mean().item())

    model = SentenceTransformer("all-MiniLM-L6-v2")
    ref_emb = model.encode([reference], convert_to_tensor=True).cpu().numpy()
    gen_emb = model.encode([generated], convert_to_tensor=True).cpu().numpy()
    scores["Cosine Similarity"] = float(
        cosine_similarity(ref_emb.reshape(1, -1), gen_emb.reshape(1, -1))[0][0]
    )

    return scores


@summarizer_router.post("/evaluate_summary")
async def evaluate_summaries(request: SummaryEvaluationRequest) -> dict:
    """Evaluate a generated summary against a reference using multiple metrics.

    Args:
        request: Body containing reference and generated summary strings.

    Returns:
        Dict with key ``scores`` mapping metric names to float values.

    Raises:
        HTTPException: 400 if either summary field is empty.
    """
    if not request.reference or not request.generated:
        raise HTTPException(
            status_code=400,
            detail="Both reference and generated summaries are required.",
        )

    scores = evaluate_summary(request.reference, request.generated)
    return {"scores": scores}
