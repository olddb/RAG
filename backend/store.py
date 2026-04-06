import json
from datetime import datetime, timezone
from pathlib import Path

STORE_PATH = Path(__file__).resolve().parent / "store.json"


def save_chunks(
    chunks: list[dict],
    source_filename: str,
    embedding_model: str,
) -> None:
    payload = {
        "source_filename": source_filename,
        "embedding_model": embedding_model,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "chunks": chunks,
    }
    text = json.dumps(payload, ensure_ascii=False)
    STORE_PATH.write_text(text, encoding="utf-8")


def load_store() -> dict | None:
    if not STORE_PATH.exists():
        return None
    raw = STORE_PATH.read_text(encoding="utf-8")
    return json.loads(raw)
