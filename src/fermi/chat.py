"""
chat.py -- Stage 6: CLI Conversational Interface

REPL loop that:
  1. Reads a question from the user.
  2. Retrieves the top-k most relevant chunks from the Chroma index.
  3. Calls generate_answer() with the chunks + conversation history.
  4. Displays the answer and clickable source citations (episode + timestamp).
  5. Appends the turn to history so follow-ups stay grounded.

History strategy: full history is passed to the LLM so "explain that more
simply" or "go deeper on [X]" can reference the previous answer. Phase 4
will add history truncation if context windows become a concern.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb

from .chunk import Chunk
from .config import Config, load_config
from .generate import Answer, Citation, format_timestamp, generate_answer
from .index import get_or_create_collection
from .retrieve import retrieve

# -- ANSI colours (degrade gracefully on terminals that don't support them) --
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"

_BANNER = f"""\
{_BOLD}{'-' * 60}
 Fermi Podcast Companion
 Ask anything about the Great Papers podcast series.
 Type  quit  or press Ctrl-C to exit.
{'-' * 60}{_RESET}"""


# -- display helpers ----------------------------------------------------------


def _display_answer(answer: Answer) -> None:
    """Print a formatted answer with source citations."""
    print()

    if answer.refused:
        print(f"  {_YELLOW}?  Not covered in the podcast material:{_RESET}")

    # Wrap answer text at ~72 chars for readability
    for line in _wrap(answer.text, width=72, indent="  "):
        print(line)

    if answer.citations:
        print(f"\n  {_DIM}{'-' * 55}{_RESET}")
        print(f"  {_BOLD}Sources:{_RESET}")
        for c in answer.citations:
            ts_start = format_timestamp(c.start_sec)
            ts_end = format_timestamp(c.end_sec)
            print(
                f"  {_CYAN}[{c.ref}]{_RESET} {c.episode_title}  "
                f"{_DIM}{ts_start} -> {ts_end}{_RESET}"
            )
            if c.excerpt:
                for line in _wrap(f'"{c.excerpt}"', width=68, indent="      "):
                    print(f"  {_DIM}{line}{_RESET}")

    print()


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    """Very simple word-wrap (no dependency on textwrap for portability)."""
    words = text.split()
    lines, line = [], []
    budget = width - len(indent)
    cur = 0
    for w in words:
        if cur + len(w) + (1 if line else 0) > budget:
            lines.append(indent + " ".join(line))
            line = [w]
            cur = len(w)
        else:
            cur += len(w) + (1 if line else 0)
            line.append(w)
    if line:
        lines.append(indent + " ".join(line))
    return lines or [indent]


# -- main chat loop ------------------------------------------------------------


def run_chat(config: Config | None = None) -> None:
    """Start the conversational REPL.

    Loads the Chroma index from config.chroma_dir. If the index is empty,
    tells the user to run run_pipeline.py first and exits.

    Conversation history is kept in memory for the session; follow-up
    questions are understood in context of the preceding exchange.
    """
    cfg = config or load_config()

    # -- Load index ------------------------------------------------------------
    try:
        collection = get_or_create_collection(cfg.chroma_dir, cfg.embed_model)
    except Exception as e:
        print(f"Error loading index: {e}", file=sys.stderr)
        sys.exit(1)

    doc_count = collection.count()
    if doc_count == 0:
        print(
            "Index is empty. Run the pipeline first:\n"
            "  python run_pipeline.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(_BANNER)
    print(f"  {_DIM}Index: {doc_count} chunks across all episodes{_RESET}\n")

    history: list[dict] = []

    while True:
        # -- Prompt ------------------------------------------------------------
        try:
            query = input(f"{_BOLD}You:{_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q", "bye"}:
            print("Bye!")
            break

        # -- Retrieve ----------------------------------------------------------
        chunks: list[Chunk] = retrieve(query, collection, top_k=cfg.top_k)
        if not chunks:
            print("  (No relevant chunks found in the index.)\n")
            continue

        # -- Generate ----------------------------------------------------------
        answer: Answer = generate_answer(query, chunks, history, cfg)

        # -- Display -----------------------------------------------------------
        _display_answer(answer)

        # -- Update history (store natural-language text, not JSON) ------------
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer.text})


if __name__ == "__main__":
    run_chat()
