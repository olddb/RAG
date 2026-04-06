"""
Retrieval: rank text chunks by cosine similarity between embeddings.

Embeddings live in the same vector space (same model for query and chunks),
so a higher cosine score means "more semantically aligned" in that space.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """
    Cosine similarity between two vectors (same length).

    Geometric idea: measure the angle between two arrows from the origin.
    - 1.0  -> same direction (very similar in embedding space)
    - 0.0  -> orthogonal (unrelated)
    - -1.0 -> opposite direction (rare for typical text embeddings)

    Formula: dot(a, b) / (||a|| * ||b||)
    This is equivalent to the cosine of the angle between a and b.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same length for cosine similarity")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # Avoid division by zero if a model ever returns a zero vector
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def get_top_k(
    query_vec: list[float],
    chunks: list[dict],
    k: int = 3,
) -> list[tuple[float, dict]]:
    """
    Score each chunk against the query embedding and return the top-k chunks.

    Each chunk dict is expected to have an "embedding" key (list of floats)
    and usually a "text" key. Other keys are preserved.

    Returns:
        List of (score, chunk) pairs, sorted from highest score to lowest.
        Length is min(k, number of chunks). Caller can format JSON or strip
        heavy fields (e.g. embeddings) before sending to the client.
    """
    if k < 1:
        return []
    if not chunks:
        return []

    scored: list[tuple[float, dict]] = []
    for chunk in chunks:
        emb = chunk.get("embedding")
        if emb is None:
            continue
        score = cosine_similarity(query_vec, emb)
        scored.append((score, chunk))

    # Highest cosine first: best semantic match for the query
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:k]
