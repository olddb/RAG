# RAG — Retrieval-Augmented Generation

Minimalist FastAPI backend for uploading PDFs, indexing them via Ollama embeddings, and querying documents using cosine similarity retrieval.

## Architecture

```
POST /upload   →  extract text → chunk → embed (Ollama) → store.json
POST /query    →  embed question → cosine similarity → top-3 chunks + generate answer
GET  /store    →  store metadata (without vectors)
```

Embeddings are generated locally by Ollama (`nomic-embed-text` by default). No calls to external APIs are made.

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- Required models available locally:

```bash
ollama pull nomic-embed-text
ollama pull mistral
```

## Installation

From the project root:

```bash
cd RAG/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Starting the server

```bash
source venv/bin/activate
fastapi dev main.py
```

The API is available at http://127.0.0.1:8000 and interactive documentation at http://127.0.0.1:8000/docs.

## Endpoints

### `POST /upload`

Upload a PDF file, split it into 500-character chunks (100-character overlap), and generate embeddings for each chunk via Ollama.

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@document.pdf"
```

Response:

```json
{
  "filename": "document.pdf",
  "chunk_count": 42,
  "saved_to": "/path/to/store.json",
  "embedding_model": "nomic-embed-text"
}
```

### `POST /query`

Ask a question in natural language. The backend embeds the question and returns the 3 most relevant chunks by cosine similarity, plus a generated answer using Mistral.

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this document?"}'
```

Response:

```json
{
  "question": "What is the main topic of this document?",
  "retrieved": [
    {
      "score": 0.91,
      "text": "...",
      "line_start": 10,
      "line_end": 15,
      "chunk_index": 2,
      "source_filename": "document.pdf"
    },
    { "score": 0.87, "text": "..." },
    { "score": 0.83, "text": "..." }
  ],
  "answer": "Based on the provided documents, the main topic is...",
  "source_filename": "document.pdf"
}
```

### `GET /store`

Returns metadata about the last upload without loading the embeddings.

```bash
curl http://127.0.0.1:8000/store
```

Response:

```json
{
  "exists": true,
  "path": "/path/to/store.json",
  "source_filename": "document.pdf",
  "embedding_model": "nomic-embed-text",
  "updated_at": "2026-03-22T10:00:00+00:00",
  "chunk_count": 42
}
```

## Environment Variables

| Variable                   | Default                  | Description                              |
|----------------------------|--------------------------|------------------------------------------|
| `OLLAMA_BASE`              | `http://127.0.0.1:11434` | Ollama server URL                        |
| `OLLAMA_EMBED_MODEL`       | `nomic-embed-text`       | Embedding model to use                   |
| `OLLAMA_GENERATION_MODEL`  | `mistral`                | Generation model to use for answers      |

## Storage

After each `POST /upload`, chunks and embeddings are saved to `backend/store.json` (the file is overwritten with each new upload — one document at a time).

## How it works

1. **Upload**: PDF text is extracted and split into overlapping chunks with line number metadata
2. **Embedding**: Each chunk is embedded using Ollama's `nomic-embed-text` model
3. **Query**: Question is embedded and compared against stored chunks using cosine similarity
4. **Generation**: Top-3 chunks are provided as context to Mistral, which generates an answer
5. **Response**: Returns the question, retrieved chunks with scores/metadata, and the generated answer
