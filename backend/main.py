import io

from pypdf import PdfReader
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from embeddings import EMBED_MODEL, embed
from retrieval import get_top_k
from store import STORE_PATH, load_store, save_chunks

app = FastAPI()

# Upload files

def extract_text(pdf_bytes: bytes) -> str:
    # UploadFile is not a full file-like object for pypdf (seek signature differs)
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text, size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i+size])
    return chunks

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    text = extract_text(pdf_bytes)
    text_chunks = chunk_text(text)
    # One embedding per chunk via Ollama (nomic-embed-text)
    stored = []
    for piece in text_chunks:
        if not piece.strip():
            continue
        vector = await embed(piece)
        stored.append({"text": piece, "embedding": vector})
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


@app.post("/query")
async def query(body: QueryBody):
    data = load_store()
    if data is None:
        return {"error": "No data stored"}
    query_vec = await embed(body.question)
    top = get_top_k(query_vec, data["chunks"], k=3)
    # Omit raw embeddings from the HTTP response (large); keep scores + text for debugging
    retrieved = [{"score": score, "text": chunk["text"]} for score, chunk in top]
    return {"question": body.question, "retrieved": retrieved}

# Base API

@app.get("/")
async def root():
    return {"message": "Hello World"}
