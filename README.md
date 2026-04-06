# RAG — Retrieval-Augmented Generation (backend)

Backend FastAPI minimaliste qui permet d'uploader un PDF, de l'indexer via des embeddings Ollama, puis d'interroger le document par similarité cosinus.

## Architecture

```
POST /upload   →  extract text → chunk → embed (Ollama) → store.json
POST /query    →  embed question → cosine similarity → top-3 chunks
GET  /store    →  metadata du store (sans les vecteurs)
```

Les embeddings sont produits localement par Ollama (`nomic-embed-text` par défaut). Aucun appel vers un service cloud n'est effectué.

## Prérequis

- Python 3.12+
- [Ollama](https://ollama.com) installé et en cours d'exécution (`ollama serve`)
- Le modèle d'embeddings disponible localement :

```bash
ollama pull nomic-embed-text
```

## Installation

Depuis la racine du projet :

```bash
cd rag/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Lancer le serveur

```bash
source venv/bin/activate
fastapi dev main.py
```

L'API est disponible sur http://127.0.0.1:8000 et la documentation interactive sur http://127.0.0.1:8000/docs.

## Endpoints

### `POST /upload`

Uploade un fichier PDF, le découpe en chunks de 500 caractères (overlap 100), et génère un embedding par chunk via Ollama.

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@document.pdf"
```

Réponse :

```json
{
  "filename": "document.pdf",
  "chunk_count": 42,
  "saved_to": "/path/to/store.json",
  "embedding_model": "nomic-embed-text"
}
```

### `POST /query`

Pose une question en langage naturel. Le backend embed la question avec le même modèle, puis retourne les 3 chunks les plus proches par similarité cosinus.

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this document?"}'
```

Réponse :

```json
{
  "question": "What is the main topic of this document?",
  "retrieved": [
    { "score": 0.91, "text": "..." },
    { "score": 0.87, "text": "..." },
    { "score": 0.83, "text": "..." }
  ]
}
```

### `GET /store`

Retourne les métadonnées du dernier upload sans charger les vecteurs.

```bash
curl http://127.0.0.1:8000/store
```

Réponse :

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

## Variables d'environnement

| Variable             | Défaut                      | Description                              |
|----------------------|-----------------------------|------------------------------------------|
| `OLLAMA_BASE`        | `http://127.0.0.1:11434`    | URL du serveur Ollama                    |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text`          | Modèle d'embeddings à utiliser           |

## Stockage

Après chaque `POST /upload`, les chunks et leurs embeddings sont écrits dans `backend/store.json` (le fichier est écrasé à chaque nouvel upload — un seul document à la fois).
