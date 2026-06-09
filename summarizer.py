import os
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

if _LLM_PROVIDER == "openai":
    from openai import OpenAI as _OpenAIClient
    _openai_client = _OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL") or None,
    )
    _openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
else:
    import ollama
import io
from rouge_score import rouge_scorer
import sacrebleu
import bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

summarizer_router = APIRouter()

def read_pdf_from_bytes(file_bytes):
    """Extract text from a PDF file given its bytes."""
    try:
        file_stream = io.BytesIO(file_bytes)  
        reader = PdfReader(file_stream)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        print(f"Extracted {len(text)} characters from PDF.")
        
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        raise

def chunk_text(text, chunk_size=1024):
    """Split text into smaller chunks to fit model input limits."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def generate_summary_prompt(text, summary_type):
    """Generate a dynamic prompt based on the selected summary type."""
    prompt_templates = {
        "abstractive": f"Create an abstractive summary of the following text:\n{text}",
        "extractive": f"Extract the most important key sentences from the following text:\n{text}",
        "long": f"Generate a detailed and comprehensive summary of the following text:\n{text}",
        "short": f"Create a concise summary of the following text in a few sentences:\n{text}",
    }
    return prompt_templates.get(summary_type, prompt_templates["abstractive"])  # Default to abstractive

def summarize_text_ollama(text, summary_type):
    """Generate a summary for a given text chunk using the configured LLM."""
    try:
        prompt = generate_summary_prompt(text, summary_type)
        if _LLM_PROVIDER == "openai":
            response = _openai_client.chat.completions.create(
                model=_openai_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        else:
            response = ollama.chat(model='gpt-oss:20b', messages=[{'role': 'user', 'content': prompt}])
            return response['message']['content']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return ""

@summarizer_router.post("/summarize")
async def summarize_pdf(
    file: UploadFile = File(...),
    temperature: float = 0.7,
    max_tokens: int = 2048,
    summary_type: str = Query("abstractive", enum=["abstractive", "extractive", "long", "short"])
):
    try:
        print(f"Received file: {file.filename}")

        content = await file.read()
        print(f"File Size: {len(content)} bytes")

        text = read_pdf_from_bytes(content)
        print(f"Text Extraction Successful: {len(text)} characters")

        ### This is temporary functionality to check if pdf is curropted during transfer from UI to backend. Remove this. 
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = f"saved_pdfs/{timestamp}.pdf"
        
        os.makedirs("saved_pdfs", exist_ok=True)
        with open(saved_path, "wb") as f:
            f.write(content)
            
        ###
        chunks = chunk_text(text, max_tokens)
        print(f"Number of chunks created: {len(chunks)}")

        # Summarize each chunk separately using selected summary type
        chunk_summaries = [summarize_text_ollama(chunk, summary_type) for chunk in chunks]
        print(f"Generated {len(chunk_summaries)} chunk summaries")

        final_summary = " ".join(chunk_summaries)

        return {"summary": final_summary}


    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}






# -------------------- Summary Evaluation --------------------
class SummaryEvaluationRequest(BaseModel):
    reference: str
    generated: str

def evaluate_summary(reference, generated):
    """Evaluate the summary using multiple metrics."""
    scores = {}

    # ROUGE Score
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(reference, generated)
    scores['ROUGE-1'] = float(rouge_scores['rouge1'].fmeasure)
    scores['ROUGE-2'] = float(rouge_scores['rouge2'].fmeasure)
    scores['ROUGE-L'] = float(rouge_scores['rougeL'].fmeasure)

    # BLEU Score
    scores['BLEU'] = float(sacrebleu.corpus_bleu([generated], [[reference]]).score)

    # BERTScore
    P, R, F1 = bert_score.score([generated], [reference], lang="en")
    scores['BERTScore'] = float(F1.mean().item())  # Convert from tensor to Python float

    # Cosine Similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ref_emb = model.encode([reference], convert_to_tensor=True).cpu().numpy()
    gen_emb = model.encode([generated], convert_to_tensor=True).cpu().numpy()
    scores['Cosine Similarity'] = float(cosine_similarity(ref_emb.reshape(1, -1), gen_emb.reshape(1, -1))[0][0])  # Convert numpy float to Python float

    return scores

@summarizer_router.post("/evaluate_summary")
async def evaluate_summaries(request: SummaryEvaluationRequest):
    if not request.reference or not request.generated:
        raise HTTPException(status_code=400, detail="Both reference and generated summaries are required.")

    scores = evaluate_summary(request.reference, request.generated)
    return {"scores": scores}
