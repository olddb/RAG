import io

from pypdf import PdfReader
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from embeddings import EMBED_MODEL, embed
from retrieval import get_top_k
from store import STORE_PATH, load_store, save_chunks
from generation import generate_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload files

def extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text, size=500, overlap=100):
    chunks = []
    chunk_index = 0
    for i in range(0, len(text), size - overlap):
        end = min(i + size, len(text))
        chunk_content = text[i:end]
        
        if not chunk_content.strip():
            continue
        
        line_start = text[:i].count("\n")
        line_end = text[:end].count("\n")
        
        chunks.append({
            "text": chunk_content,
            "start_pos": i,
            "end_pos": end,
            "line_start": line_start,
            "line_end": line_end,
            "chunk_index": chunk_index,
        })
        chunk_index += 1
    
    return chunks

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    text = extract_text(pdf_bytes)
    text_chunks = chunk_text(text)
    stored = []
    for chunk in text_chunks:
        vector = await embed(chunk["text"])
        stored.append({
            "text": chunk["text"],
            "embedding": vector,
            "start_pos": chunk["start_pos"],
            "end_pos": chunk["end_pos"],
            "line_start": chunk["line_start"],
            "line_end": chunk["line_end"],
            "chunk_index": chunk["chunk_index"],
        })
    save_chunks(stored, source_filename=file.filename or "upload.bin", embedding_model=EMBED_MODEL)
    return {
        "filename": file.filename,
        "chunk_count": len(stored),
        "saved_to": str(STORE_PATH),
        "embedding_model": EMBED_MODEL,
    }

# Store API

@app.get("/store")
async def store_info():
    data = load_store()
    if data is None:
        return {"exists": False, "path": str(STORE_PATH)}
    return {
        "exists": True,
        "path": str(STORE_PATH),
        "source_filename": data["source_filename"],
        "embedding_model": data["embedding_model"],
        "updated_at": data["updated_at"],
        "chunk_count": len(data["chunks"]),
    }

# Query API

class QueryBody(BaseModel):
    question: str
    top_k: int = 3


@app.post("/query")
async def query(body: QueryBody):
    data = load_store()
    if data is None:
        return {"error": "No data stored"}
    query_vec = await embed(body.question)
    top = get_top_k(query_vec, data["chunks"], k=body.top_k)
    retrieved = [{
        "score": score,
        "text": chunk["text"],
        "line_start": chunk.get("line_start", 0),
        "line_end": chunk.get("line_end", 0),
        "chunk_index": chunk.get("chunk_index", -1),
        "source_filename": data["source_filename"],
    } for score, chunk in top]
    
    answer = await generate_answer(body.question, retrieved)
    
    return {
        "question": body.question,
        "retrieved": retrieved,
        "answer": answer,
        "source_filename": data["source_filename"],
    }

# Base API

@app.get("/")
async def root():
    return {"message": "Hello World"}
