"""
run_eval.py -- Phase 3 Evaluation Runner
=========================================
Executes all cases in eval/cases.json against the live pipeline and writes
one raw JSON file per case to eval/results/<run_id>/.

Usage:
    python run_eval.py                          # run all cases
    python run_eval.py --case EP04-FACT-01      # run single case
    python run_eval.py --run-id my_label        # tag the output directory

Output per case (eval/results/<run_id>/<case_id>.json):
    - case metadata (id, category, description, criteria)
    - all turns with: query, retrieved_chunks (text + score), final Answer
    - timing per turn
    - model used
    - raw LLM response preserved

Nothing is summarized away. Every retrieved chunk and every answer token is
written to disk exactly as returned.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fermi.config import load_config
from fermi.generate import Answer, generate_answer
from fermi.index import get_or_create_collection
from fermi.retrieve import retrieve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _answer_to_dict(answer: Answer) -> dict:
    return {
        "text": answer.text,
        "refused": answer.refused,
        "citations": [
            {
                "ref": c.ref,
                "episode_id": c.episode_id,
                "episode_title": c.episode_title,
                "chunk_id": c.chunk_id,
                "start_sec": c.start_sec,
                "end_sec": c.end_sec,
                "excerpt": c.excerpt,
            }
            for c in answer.citations
        ],
        # Phase 4: chunk_ids retrieved but not used because refused=True
        "related_but_insufficient": getattr(answer, "related_but_insufficient", []),
    }


def _chunks_to_list(chunks, scores=None) -> list[dict]:
    """Serialize retrieved chunks, including cosine scores where available."""
    out = []
    for i, c in enumerate(chunks):
        out.append({
            "rank": i + 1,
            "chunk_id": c.chunk_id,
            "episode_id": c.episode_id,
            "episode_title": c.episode_title,
            "start_sec": c.start_sec,
            "end_sec": c.end_sec,
            "n_segments": len(c.segments),
            "text_preview": c.text[:300],   # first 300 chars -- readable in results
            "text_full": c.text,             # full text -- for completeness
        })
    return out


def run_case(case: dict, collection, cfg) -> dict:
    """Run all turns in one eval case. Returns the full result dict."""
    turns = case["turns"]
    history: list[dict] = []
    turn_results = []

    for turn_idx, turn in enumerate(turns):
        query = turn["content"]
        turn_start = time.monotonic()

        # Retrieve
        chunks = retrieve(query, collection, top_k=cfg.top_k)

        # Generate (pass accumulated history for multi-turn)
        answer = generate_answer(query, chunks, history, cfg)

        elapsed = time.monotonic() - turn_start

        turn_results.append({
            "turn_index": turn_idx,
            "query": query,
            "retrieved_chunks": _chunks_to_list(chunks),
            "answer": _answer_to_dict(answer),
            "elapsed_sec": round(elapsed, 2),
        })

        # Append this turn to history for subsequent turns
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer.text})

    return {
        "case_id": case["id"],
        "category": case["category"],
        "description": case["description"],
        "episode_hint": case.get("episode_hint", ""),
        "criteria": case.get("criteria", {}),
        "model": cfg.llm_model,
        "embed_model": cfg.embed_model,
        "top_k": cfg.top_k,
        "turns": turn_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fermi Podcast Companion -- Eval Runner")
    parser.add_argument("--case", default=None, help="Run only this case ID")
    parser.add_argument("--run-id", default=None,
                        help="Label for output directory (default: timestamp)")
    args = parser.parse_args()

    cfg = load_config()
    print(f"Model: {cfg.llm_model}")
    print(f"Embed: {cfg.embed_model}")
    print(f"Top-K: {cfg.top_k}")

    # Load eval cases
    cases_path = Path("eval/cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"ERROR: case '{args.case}' not found in {cases_path}")
            sys.exit(1)

    # Output directory
    run_id = args.run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    out_dir = Path("eval/results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun ID : {run_id}")
    print(f"Cases  : {len(cases)}")
    print(f"Output : {out_dir}")
    print("-" * 60)

    # Load collection once
    print("Loading Chroma collection...")
    collection = get_or_create_collection(cfg.chroma_dir, cfg.embed_model)
    print(f"Collection: {collection.count()} chunks indexed\n")

    # Run cases
    results_index = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": cfg.llm_model,
        "embed_model": cfg.embed_model,
        "top_k": cfg.top_k,
        "cases": [],
    }

    total_start = time.monotonic()
    for i, case in enumerate(cases, 1):
        cid = case["id"]
        print(f"[{i:02d}/{len(cases):02d}] {cid}  ({case['category']})")
        sys.stdout.flush()

        try:
            result = run_case(case, collection, cfg)
            status = "ok"
        except Exception as e:
            result = {
                "case_id": cid,
                "error": f"{type(e).__name__}: {e}",
            }
            status = "ERROR"

        # Write individual case file
        out_file = out_dir / f"{cid}.json"
        out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        # Summarise to console
        if status == "ok" and result.get("turns"):
            last_turn = result["turns"][-1]
            ans = last_turn["answer"]
            refused_flag = "[REFUSED]" if ans["refused"] else ""
            n_citations = len(ans["citations"])
            elapsed = last_turn["elapsed_sec"]
            preview = ans["text"][:120].replace("\n", " ")
            print(f"         status={status}  refused={ans['refused']}  "
                  f"citations={n_citations}  elapsed={elapsed}s  {refused_flag}")
            print(f"         \"{preview}...\"")
        else:
            print(f"         status={status}  {result.get('error', '')}")

        results_index["cases"].append({
            "case_id": cid,
            "category": case["category"],
            "status": status,
            "file": str(out_file),
        })
        print()

    total_elapsed = round(time.monotonic() - total_start, 1)

    # Write index file
    index_file = out_dir / "_index.json"
    index_file.write_text(json.dumps(results_index, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"Done. {len(cases)} cases in {total_elapsed}s")
    print(f"Results in: {out_dir}")
    print(f"Index:      {index_file}")


if __name__ == "__main__":
    main()
