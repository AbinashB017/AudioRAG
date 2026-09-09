"""
retrieve.py -- Stage 4: Query -> Top-K Chunks

Plain dense cosine similarity over the full Chroma index (all 5 episodes).
No filtering, reranking, or BM25 in Phase 1 -- this is the simplest thing
that works. Phase 4 will add hybrid retrieval if eval shows it's needed.
"""

from __future__ import annotations

import json
import logging

import chromadb
from openai import OpenAI

from .chunk import Chunk
from .config import load_config


def _decompose_query(query: str) -> list[str]:
    """Use a cheap LLM call to split compound queries into sub-topics."""
    cfg = load_config()
    client = OpenAI(
        api_key=cfg.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "Fermi"},
    )
    
    prompt = f"""You are a query analyzer for a podcast search engine.
Determine if the following user query is a compound query asking to compare or explain TWO distinct concepts/entities that likely require retrieving separate documents.
If it is single-topic or a broad question, return a JSON list with just the original query.
If it is compound, return a JSON list of 2 distinct search queries (one for each topic).

Query: "{query}"

Return ONLY a valid JSON list of strings, nothing else. Example:
["shannon entropy", "self attention"]
"""
    try:
        response = client.chat.completions.create(
            model=cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        # Clean markdown formatting if present
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        queries = json.loads(content)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries
    except Exception as e:
        logging.warning(f"Query decomposition failed: {e}")
        pass
    
    return [query]


def retrieve(
    query: str,
    collection: chromadb.Collection,
    top_k: int = 5,
) -> list[Chunk]:
    """Embed query and return the top_k most similar Chunks from the index.

    Uses Chroma's built-in embedding function (same bge-small model used at
    index time) so query and document embeddings are always in the same space.
    If the query is compound, decomposes it and retrieves top_k for each sub-topic.

    Args:
        query:       Natural-language question from the user.
        collection:  Chroma collection from index.get_or_create_collection().
        top_k:       Number of results to return (default 5).

    Returns:
        List of Chunk objects ordered by cosine similarity (best first).
        Returns fewer than top_k results if the collection has fewer documents.
    """
    queries = _decompose_query(query)
    all_chunks = []
    seen_ids = set()
    
    for q in queries:
        n = min(top_k, collection.count())
        if n == 0:
            continue

        results = collection.query(query_texts=[q], n_results=n)

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for chunk_id, doc, meta in zip(ids, documents, metadatas):
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        episode_id=meta["episode_id"],
                        episode_title=meta["episode_title"],
                        text=doc,
                        start_sec=float(meta["start_sec"]),
                        end_sec=float(meta["end_sec"]),
                        segments=json.loads(meta.get("segments_json", "[]")),
                    )
                )

    return all_chunks
