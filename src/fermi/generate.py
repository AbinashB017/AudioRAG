"""
generate.py -- Stage 5: Chunks + History -> Grounded Answer with Citations

Uses OpenRouter (anthropic/claude-sonnet-4-5 for eval, haiku for dev) via the
OpenAI-compatible SDK.

Phase 4 changes (evidence-driven improvements):
  1. Citations are chunk-ID-bound: the LLM outputs only chunk_id + excerpt;
     episode_id, episode_title, start_sec, end_sec are derived programmatically
     from the retrieved Chunk objects. This eliminates citation drift (CROSS-01).
  2. Context block shows an episode inventory header + an explicit cross-episode
     coverage rule -- the LLM must address ALL represented episodes or explain
     why an episode's evidence is irrelevant (CROSS-02).
  3. When refused=True, citations is always [] -- no misleading "close but
     insufficient" citations rendered in the UI. The chunk_ids that were
     retrieved but not sufficient are stored in related_but_insufficient for
     eval visibility (SCOPE-01, SCOPE-02).

System prompt rules:
  1. Answer from provided chunks only -- no general knowledge.
  2. Cite every claim with [N]; chunk_id is the citation anchor.
  3. Address ALL episodes in context for comparison questions.
  4. If refused, set citations=[].
  5. If insufficient context, refused=True + brief explanation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

from .chunk import Chunk
from .config import Config

# -- data models --------------------------------------------------------------


@dataclass
class Citation:
    """One traceable source: episode + segment-refined timestamp + verbatim excerpt."""

    ref: str            # matches inline [N] in Answer.text
    episode_id: str
    episode_title: str
    chunk_id: str
    start_sec: float    # segment-level refined (not chunk-level)
    end_sec: float
    excerpt: str        # verbatim 1-2 sentence snippet from the chunk text
    is_episode_level: bool = False  # True if this is a full-episode recommendation


@dataclass
class Answer:
    """LLM response: natural-language text + traceable citations."""

    text: str
    citations: list[Citation]
    refused: bool       # True -> LLM determined context doesn't support the query
    # chunk_ids that were retrieved but deemed insufficient (only populated when refused=True)
    # preserved for eval visibility; NOT rendered as clickable citations in the UI
    related_but_insufficient: list[str] = field(default_factory=list)


# -- prompt templates ---------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a learning assistant for the Fermi Podcast -- a series that explores landmark scientific papers through in-depth conversation.

Your job is to help learners understand and navigate the podcast content.

STRICT RULES (enforced without exception):
1. Answer using ONLY the transcript excerpts and episode summaries provided. Never add facts from general knowledge.
2. Cite every substantive claim with an inline reference number [N] (e.g. [1], [2]).
3. If answering a factual question, cite the chunk_id of the excerpt it comes from (e.g. "ep01_0000").
4. If the query asks for a recommendation or overview (e.g., "which episode covers X", "what is ep04 about"), you should primarily use the Episode Summaries to answer. For recommendations, cite the episode as a whole using its episode_id (e.g., "ep04") as the chunk_id. Do NOT invent IDs.
5. Cross-episode rule: if the question is comparative or asks 'which of these', your answer MUST explicitly address the evidence from ALL episodes represented in the chunks provided above -- either use it or state why it is not relevant.
6. If the provided context does not contain enough information, set "refused": true, write a brief explanation of what is missing, and set "citations": [].
7. Keep citation excerpts to 1-2 verbatim sentences copied exactly from the chunk text (leave excerpt empty if citing a full episode).

OUTPUT FORMAT -- respond with ONLY a valid JSON object, no markdown, no preamble:
{
  "text": "Your answer here. Every substantive claim has an inline citation like [1] or [2].",
  "citations": [
    {
      "ref": "1",
      "chunk_id": "ep01_0042",
      "excerpt": "Verbatim 1-2 sentence quote copied exactly from that chunk."
    }
  ],
  "refused": false
}

When refused=true, citations MUST be an empty array [].
"""


# -- utilities -----------------------------------------------------------------


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS string for display and audio seeking."""
    m, s = int(seconds // 60), int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _refine_start_sec(
    excerpt: str,
    segments: list[dict],
    fallback_start: float,
) -> float:
    """Refine a chunk-level start_sec to the best-matching constituent segment.

    Given the verbatim excerpt the LLM pulled from a chunk, find which of the
    chunk's constituent Whisper segments has the highest word-overlap with that
    excerpt. Returns the start of that segment, shrinking the seek gap from
    ~15-90s (chunk granularity) down to a few seconds (sentence granularity).

    Falls back to fallback_start if no segments provided or no words match.
    """
    if not segments or not excerpt:
        return fallback_start

    excerpt_words = set(excerpt.lower().split())
    best_start = fallback_start
    best_score = 0

    for seg in segments:
        seg_words = set(seg["text"].lower().split())
        overlap = len(excerpt_words & seg_words)
        if overlap > best_score:
            best_score = overlap
            best_start = seg["start"]

    return best_start


def _get_client(config: Config) -> OpenAI:
    return OpenAI(
        api_key=config.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/fermi-podcast-companion",
            "X-Title": "Fermi Podcast Companion",
        },
    )

def _load_summaries() -> list[dict]:
    """Load all offline-generated episode summaries."""
    from pathlib import Path
    summaries = []
    summary_dir = Path("data/episode_summaries")
    if summary_dir.exists():
        for p in sorted(summary_dir.glob("*.json")):
            try:
                summaries.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
    return summaries


def _build_context_block(chunks: list[Chunk]) -> str:
    """Format retrieved chunks and episode summaries for the LLM context window.

    Phase 4/1B: adds an episode inventory header and ALL 16 episode summaries
    so the LLM can answer recommendation queries.
    """
    parts = []
    
    # 1. Episode Summaries
    summaries = _load_summaries()
    if summaries:
        parts.append("ALL EPISODE SUMMARIES (For recommendations/overviews):")
        for s in summaries:
            parts.append(
                f"- [{s['episode_id']}] \"{s['episode_title']}\"\n"
                f"  Topics: {', '.join(s['topics'])}\n"
                f"  Concepts: {', '.join(s['concepts'])}\n"
                f"  Description: {s['description']}"
            )
        parts.append("\n" + "-"*40 + "\n")

    # Build episode inventory for the header
    episode_counts: dict[str, tuple[str, int]] = {}  # episode_id -> (title, count)
    for c in chunks:
        if c.episode_id not in episode_counts:
            episode_counts[c.episode_id] = (c.episode_title, 0)
        title, count = episode_counts[c.episode_id]
        episode_counts[c.episode_id] = (title, count + 1)

    inventory_lines = [
        f"  {ep_id}: {title!r} ({n} chunk{'s' if n != 1 else ''})"
        for ep_id, (title, n) in episode_counts.items()
    ]
    inventory = "\n".join(inventory_lines)

    parts.append("RETRIEVED TRANSCRIPT EXCERPTS -- your ONLY allowed source for factual queries.\n"
                 "Cite claims using the chunk_id shown for each excerpt.\n\n"
                 f"Episodes represented in chunks below:\n{inventory}\n\n"
                 "IMPORTANT: If the question is comparative or asks 'which of these', "
                 "you MUST address evidence from ALL episodes represented in the chunks provided above.\n")

    for chunk in chunks:
        ts = f"{format_timestamp(chunk.start_sec)} -> {format_timestamp(chunk.end_sec)}"
        parts.append(
            f"\n--- Chunk {chunk.chunk_id} ---\n"
            f"Episode: {chunk.episode_id} | \"{chunk.episode_title}\"\n"
            f"Timestamp: {ts}\n"
            f"{chunk.text}\n"
        )
    return "\n".join(parts)


def _extract_json(text: str) -> dict:
    """Robustly extract a JSON object from LLM response text.

    Handles three common formats:
      - Pure JSON
      - JSON wrapped in ```json ... ```
      - JSON somewhere inside prose
    """
    text = text.strip()

    # Strip markdown code block
    md = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md:
        text = md.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost {...} block
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    if obj:
        return json.loads(obj.group(0))

    raise ValueError(f"Could not extract JSON from LLM response: {text[:300]!r}")


# -- main generation function -------------------------------------------------


def generate_answer(
    query: str,
    chunks: list[Chunk],
    history: list[dict],
    config: Config,
) -> Answer:
    """Call the LLM to produce a grounded answer with traceable citations.

    Phase 4 changes:
    - LLM outputs chunk_id + excerpt only; all other Citation fields derived
      programmatically from the retrieved Chunk objects (Fix 1: citation binding).
    - Context block shows episode inventory + cross-episode coverage instruction
      (Fix 2: cross-episode completeness).
    - When refused=True, citations=[] always; related_but_insufficient records
      the chunk_ids that were retrieved but not used (Fix 3: refusal hygiene).

    Args:
        query:    Current user question.
        chunks:   Retrieved chunks from retrieve() -- the only allowed source.
        history:  Previous turns as [{"role": "user"|"assistant", "content": str}].
                  Enables follow-ups like "explain that more simply" to stay grounded.
        config:   Runtime configuration (controls which LLM model is called).

    Returns:
        Answer with text, citations derived from actual chunk metadata, and
        refused flag. citations is always [] when refused=True.
    """
    client = _get_client(config)

    context_block = _build_context_block(chunks)
    user_message = f"{context_block}\n\nQUESTION: {query}"

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content or ""
        data = _extract_json(raw)

        refused = bool(data.get("refused", False))

        # Fix 1: Build chunk lookup -- all citation metadata comes from here,
        # not from the LLM's output. The LLM only provides chunk_id + excerpt.
        chunk_lookup: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        
        # Build summary lookup for episode-level citations
        summary_lookup: dict[str, dict] = {s["episode_id"]: s for s in _load_summaries()}

        citations: list[Citation] = []
        if not refused:
            for i, c in enumerate(data.get("citations", [])):
                chunk_id = c.get("chunk_id", "")
                excerpt = c.get("excerpt", "")

                # Handle episode-level citations (recommendations)
                if chunk_id in summary_lookup:
                    s = summary_lookup[chunk_id]
                    citations.append(
                        Citation(
                            ref=str(c.get("ref", i + 1)),
                            episode_id=s["episode_id"],
                            episode_title=s["episode_title"],
                            chunk_id=chunk_id,
                            start_sec=0.0,
                            end_sec=0.0,
                            excerpt="",
                            is_episode_level=True,
                        )
                    )
                    continue

                chunk = chunk_lookup.get(chunk_id)
                if chunk is None:
                    # LLM referenced a chunk_id not in the retrieved set -- skip.
                    # This is a grounding failure; dropping it is safer than
                    # fabricating a citation from an unknown chunk.
                    continue

                # Derive all metadata from the actual Chunk object (not from LLM output)
                refined_start = _refine_start_sec(excerpt, chunk.segments, chunk.start_sec)
                citations.append(
                    Citation(
                        ref=str(c.get("ref", i + 1)),
                        episode_id=chunk.episode_id,
                        episode_title=chunk.episode_title,
                        chunk_id=chunk_id,
                        start_sec=refined_start,
                        end_sec=chunk.end_sec,
                        excerpt=excerpt,
                    )
                )

        # Fix 3: When refused, citations is always empty. Preserve retrieved
        # chunk_ids in related_but_insufficient for eval inspection.
        related_but_insufficient: list[str] = []
        if refused:
            related_but_insufficient = [c.chunk_id for c in chunks]

        return Answer(
            text=data.get("text", raw),
            citations=citations,
            refused=refused,
            related_but_insufficient=related_but_insufficient,
        )

    except Exception as e:
        # Graceful fallback: surface the error without crashing the chat loop
        return Answer(
            text=f"[Generation error: {type(e).__name__}: {e}]",
            citations=[],
            refused=False,
        )
