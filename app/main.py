# FastAPI - REST API Key
from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import create_rag_pipeline

app = FastAPI()
 
qa_chain = create_rag_pipeline()
 
class Query(BaseModel):
    query: str
 
@app.get("/")
def home():
    return {"message": "RAG API Running"}
 
@app.post("/ask")
def ask(q : Query):
    result = qa_chain.invoke({"question": q.query})
    response = result['answer']
 
    sources = result.get('source_documents', [])
    return {"response": response, "sources": sources}
 