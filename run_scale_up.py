#!/usr/bin/env python3
"""
run_scale_up.py -- Phase 1B: Scale up to 16 episodes

1. Transcribes remaining episodes using transcribe_episode() (which uses cache)
2. Checks Chroma DB for already-indexed episode IDs
3. Chunks only the NEW transcripts
4. Embeds and appends the new chunks to Chroma
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, 'src')
from fermi.config import load_config
from fermi.transcribe import transcribe_episode, episode_id_from_path, episode_title_from_path
from fermi.chunk import make_chunks, Segment
from fermi.index import get_or_create_collection, index_chunks

BOLD, DIM, CYAN, GREEN, YELLOW, RED, RESET = (
    "\033[1m", "\033[2m", "\033[96m", "\033[92m", "\033[93m", "\033[91m", "\033[0m"
)

def main():
    cfg = load_config()
    audio_dir = Path("podcasts")
    
    # 1. Discover all 16 episodes
    episodes = sorted(audio_dir.glob("*.mp3"), key=lambda p: episode_id_from_path(p))
    print(f"{CYAN}--- Step 1: Transcribing All Episodes ---{RESET}")
    for ep in episodes:
        ep_id = episode_id_from_path(ep)
        title = episode_title_from_path(ep)
        print(f"\n{CYAN}> {ep_id}: {title}{RESET}")
        # transcribe_episode automatically skips hitting ASR if the transcript exists
        transcribe_episode(ep, cfg.transcript_dir, cfg)

    # 2. Connect to Chroma and figure out what is already indexed
    print(f"\n{CYAN}--- Step 2: Determining Incremental Chunks ---{RESET}")
    collection = get_or_create_collection(cfg.chroma_dir, cfg.embed_model)
    
    # Extract distinct episode IDs from the metadata in Chroma
    try:
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        indexed_eps = {m["episode_id"] for m in metadatas if m and "episode_id" in m}
    except Exception as e:
        print(f"{YELLOW}Warning: Could not read metadata from Chroma. Assuming empty. ({e}){RESET}")
        indexed_eps = set()

    print(f"Already indexed episodes in Chroma: {sorted(indexed_eps)}")
    
    # 3. Chunk ONLY the new episodes
    print(f"\n{CYAN}--- Step 3: Chunking NEW Episodes ---{RESET}")
    new_chunks = []
    for ep in episodes:
        ep_id = episode_id_from_path(ep)
        title = episode_title_from_path(ep)
        
        if ep_id in indexed_eps:
            print(f"  {GREEN}[SKIP] {ep_id} already embedded in Chroma{RESET}")
            continue
            
        cache_path = cfg.transcript_dir / f"{ep_id}.json"
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        segments = [Segment(**s) for s in raw["segments"]]
        
        chunks = make_chunks(
            segments,
            episode_id=ep_id,
            episode_title=title,
            target_words=cfg.target_words,
            overlap_words=cfg.overlap_words,
        )
        new_chunks.extend(chunks)
        print(f"  {ep_id}: {len(segments)} segments -> {len(chunks)} chunks")

    # 4. Embed and append to Chroma
    print(f"\n{CYAN}--- Step 4: Embedding and Indexing ---{RESET}")
    if not new_chunks:
        print(f"  {GREEN}[OK] No new chunks to index. Everything is up to date!{RESET}")
    else:
        print(f"  Embedding and indexing {len(new_chunks)} NEW chunks...")
        t0 = time.time()
        index_chunks(new_chunks, collection)
        elapsed = time.time() - t0
        print(f"  {GREEN}[OK] Indexed {len(new_chunks)} chunks in {elapsed:.1f}s{RESET}")
        
    print(f"\n{BOLD}Total chunks now in Chroma: {collection.count()}{RESET}")

if __name__ == "__main__":
    main()
