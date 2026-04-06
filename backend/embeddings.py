import os

import httpx

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBEDDINGS_URL = f"{OLLAMA_BASE.rstrip('/')}/api/embeddings"


async def embed(text: str) -> list[float]:
    """
    Calls Ollama embeddings API (same model for chunks and questions later).
    Requires: ollama serve + ollama pull nomic-embed-text
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            EMBEDDINGS_URL,
            json={"model": EMBED_MODEL, "prompt": text},
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
