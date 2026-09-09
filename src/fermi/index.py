"""
index.py -- Stage 3: Embed Chunks and Upsert into Chroma

Uses a local sentence-transformers model (bge-small-en-v1.5) as Chroma's
embedding function -- no API key needed, deterministic weights, works offline.

Design decisions:
- get_or_create_collection: idempotent -- safe to call on every run.
- index_chunks: uses upsert so re-running never duplicates documents.
- is_indexed: cheap count check -- skips embedding on re-runs to protect time
  and, importantly, to keep the pipeline fast when only chat/eval changes.
- Chroma metadata stores floats (start_sec, end_sec) which it handles natively.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .chunk import Chunk

COLLECTION_NAME = "podcast_chunks"


def get_or_create_collection(
    persist_dir: Path,
    embed_model: str,
) -> chromadb.Collection:
    """Return a persisted Chroma collection using a local bge-small embedding function.

    Creates the collection (and downloads the embedding model on first run) if
    it does not already exist. Subsequent calls return the existing collection.

    Args:
        persist_dir:  Path where Chroma persists its index (data/chroma_db/).
        embed_model:  HuggingFace model name, e.g. "BAAI/bge-small-en-v1.5".
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    ef = SentenceTransformerEmbeddingFunction(model_name=embed_model)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[Chunk], collection: chromadb.Collection) -> None:
    """Embed and upsert chunks into the collection.

    Idempotent -- calling with the same chunks again updates in place.
    Chunk.chunk_id is used as the Chroma document ID.

    Args:
        chunks:      Chunks to index (typically all episodes' chunks).
        collection:  Chroma collection from get_or_create_collection().
    """
    if not chunks:
        return

    # Chroma metadata values must be scalars (str / int / float / bool).
    # Segments list is JSON-serialized so it round-trips through Chroma intact.
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "episode_id": c.episode_id,
                "episode_title": c.episode_title,
                "start_sec": c.start_sec,
                "end_sec": c.end_sec,
                "segments_json": json.dumps(c.segments),
            }
            for c in chunks
        ],
    )


def is_indexed(collection: chromadb.Collection) -> bool:
    """Return True if the collection already contains documents.

    Used by run_pipeline.py to skip re-embedding on every re-run -- the most
    expensive step after ASR is never repeated unless the index is deleted.
    """
    return collection.count() > 0
