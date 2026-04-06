import os
import httpx

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")
GENERATION_MODEL = os.environ.get("OLLAMA_GENERATION_MODEL", "mistral")
GENERATION_URL = f"{OLLAMA_BASE.rstrip('/')}/api/chat"


def build_context_string(retrieved_chunks: list[dict]) -> str:
    """
    Build a formatted context string from retrieved chunks.
    
    Each chunk includes filename, line numbers, and similarity score.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        filename = chunk.get("source_filename", "document")
        line_start = chunk.get("line_start", 0)
        line_end = chunk.get("line_end", 0)
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        
        header = f"[Chunk {i} - {filename} - Lines {line_start}-{line_end} (Score: {score:.0%})]"
        context_parts.append(f"{header}\n{text}")
    
    return "\n\n".join(context_parts)


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build the complete prompt for Mistral with context and question.
    
    Clear instructions to ensure the model only uses the provided context.
    """
    if not retrieved_chunks:
        return f"""You are a helpful assistant. Answer the following question.

Question: {question}

Important: No context was provided."""
    
    context = build_context_string(retrieved_chunks)
    
    prompt = f"""You are a helpful assistant. Answer the following question based ONLY on the context provided below.

Important instructions:
- Use ONLY the information from the context
- If the answer is not in the context, respond: "I could not find an answer to your question in the provided documents."
- Be concise and relevant

Context:
{context}

Question: {question}

Answer:"""
    
    return prompt


async def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    model: str = GENERATION_MODEL,
) -> str:
    """
    Generate an answer using Mistral via Ollama.
    
    Args:
        question: The user's question
        retrieved_chunks: List of relevant chunks with metadata
        model: The Ollama model to use (default: mistral)
    
    Returns:
        The generated answer from the model
    
    Raises:
        Exception if Ollama is unavailable or generation fails
    """
    prompt = build_prompt(question, retrieved_chunks)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                GENERATION_URL,
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "Error: no response from model").strip()
    except httpx.TimeoutException:
        return "Error: generation exceeded timeout (300s)."
    except httpx.ConnectError:
        return "Error: unable to connect to Ollama. Make sure the server is running."
    except Exception as e:
        return f"Error during generation: {str(e)}"
