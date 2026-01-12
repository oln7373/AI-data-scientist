from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Create an APIRouter instance
rag_router = APIRouter()

# Load documents and set up the RAG pipeline at startup
#documents = SimpleDirectoryReader("data").load_data()
documents = SimpleDirectoryReader("data").load_data()
#print (documents)


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.1",
                      options={"temperature": 0.3},
                      request_timeout=460)


index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Define request model
class QueryRequest(BaseModel):
    question: str

@rag_router.post("/rag-query")
def rag_info(request: QueryRequest):
    try:
        response = query_engine.query(request.question)
        print ("response: ",response.response)
        return {"response": response.response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

