#!/usr/bin/env python3
"""
generate_summaries.py -- Generate Episode Summaries for the Recommendation Feature

Iterates over all cached transcripts and generates a structured summary for each
using the LLM. Caches the summaries to data/episode_summaries/ to be used
in real-time retrieval by the chat interface.

Idempotent: skips episodes that already have a summary generated.
"""

import sys
import json
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, 'src')
from fermi.config import load_config
from fermi.summarize import generate_episode_summary
from fermi.transcribe import episode_id_from_path, episode_title_from_path

CYAN, GREEN, YELLOW, RESET = "\033[96m", "\033[92m", "\033[93m", "\033[0m"

def main():
    cfg = load_config()
    summary_dir = Path("data/episode_summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # We iterate over the original audio files to get the titles properly,
    # and then look for the transcript.
    audio_dir = Path("podcasts")
    episodes = sorted(audio_dir.glob("*.mp3"), key=lambda p: episode_id_from_path(p))
    
    if not episodes:
        print("No podcasts found.")
        sys.exit(1)
        
    print(f"{CYAN}--- Generating Episode Summaries ---{RESET}")
    for ep in episodes:
        ep_id = episode_id_from_path(ep)
        title = episode_title_from_path(ep)
        
        cache_path = summary_dir / f"{ep_id}.json"
        if cache_path.exists():
            print(f"  {GREEN}[SKIP]{RESET} {ep_id} summary already exists.")
            continue
            
        transcript_path = cfg.transcript_dir / f"{ep_id}.json"
        if not transcript_path.exists():
            print(f"  {YELLOW}[WARN]{RESET} Transcript for {ep_id} not found. Skipping summary.")
            continue
            
        print(f"  {CYAN}[GEN]{RESET} Summarizing {ep_id}: {title}...")
        try:
            raw = json.loads(transcript_path.read_text(encoding="utf-8"))
            transcript = raw.get("segments", [])
            
            summary = generate_episode_summary(transcript, ep_id, title, cfg)
            
            # Save to cache
            cache_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
            print(f"    {GREEN}Done.{RESET}")
        except Exception as e:
            print(f"    {YELLOW}Error summarizing {ep_id}: {e}{RESET}")

if __name__ == "__main__":
    main()
