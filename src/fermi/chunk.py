"""
chunk.py -- Stage 2: ASR Segments -> Retrieval Chunks

Strategy: sliding word-count window over the flattened word stream.
- target_words (~200):  each chunk is ~1?1.5 min of spoken audio,
  enough for a coherent idea without burying the retriever in noise.
- overlap_words (~40):  ~15 s of repeated context prevents an answer
  from being split across two non-adjacent chunks.
- Timestamps: each Chunk carries the start of its first word's segment
  and the end of its last word's segment, so citations link to exactly
  the right audio position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .transcribe import Segment


@dataclass
class Chunk:
    """One retrievable unit: a window of words with audio timestamps."""

    chunk_id: str         # "<episode_id>_<index>"  e.g. "ep05_0042"
    episode_id: str       # e.g. "ep05"
    episode_title: str    # human-readable, e.g. "Attention Is All You Need, 2017"
    text: str             # the actual spoken words in this window
    start_sec: float      # audio timestamp of first word in this chunk
    end_sec: float        # audio timestamp of last word in this chunk
    # Constituent Whisper segments -- used for segment-level citation refinement.
    # Each dict: {"start": float, "end": float, "text": str}
    segments: list[dict] = field(default_factory=list)


def make_chunks(
    segments: list[Segment],
    episode_id: str,
    episode_title: str,
    target_words: int = 200,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Slide a word-count window over segments to produce Chunk objects.

    Algorithm:
      1. Flatten all segments into a word list, recording which segment
         each word came from (so we can look up exact timestamps).
      2. Emit a Chunk every (target_words - overlap_words) words.
      3. Each Chunk's start_sec / end_sec comes from the segment boundaries
         of its first and last words respectively.

    Args:
        segments:       Ordered list of Segment objects from transcribe_episode().
        episode_id:     e.g. "ep05"
        episode_title:  Human-readable title for display and citation.
        target_words:   Approximate words per chunk (default 200 ? 1?1.5 min).
        overlap_words:  Words shared with the next chunk to preserve context.

    Returns:
        Ordered list of Chunk objects ready for indexing.
    """
    if not segments:
        return []

    # -- Step 1: flatten segments -> (word, seg_index) list -------------------
    word_stream: list[tuple[str, int]] = []
    for seg_idx, seg in enumerate(segments):
        for word in seg.text.split():
            if word:
                word_stream.append((word, seg_idx))

    if not word_stream:
        return []

    # -- Step 2: slide window -------------------------------------------------
    chunks: list[Chunk] = []
    stride = max(1, target_words - overlap_words)
    pos = 0
    chunk_idx = 0

    while pos < len(word_stream):
        window = word_stream[pos : pos + target_words]
        if not window:
            break

        words = [w for w, _ in window]
        first_seg_idx = window[0][1]
        last_seg_idx = window[-1][1]

        # Capture every unique segment that contributed at least one word to
        # this window -- used later to refine citation timestamps to sentence
        # level rather than chunk level.
        seen_seg_indices = list(dict.fromkeys(idx for _, idx in window))
        chunk_segs = [
            {"start": segments[i].start, "end": segments[i].end, "text": segments[i].text}
            for i in seen_seg_indices
        ]

        chunks.append(
            Chunk(
                chunk_id=f"{episode_id}_{chunk_idx:04d}",
                episode_id=episode_id,
                episode_title=episode_title,
                text=" ".join(words),
                start_sec=segments[first_seg_idx].start,
                end_sec=segments[last_seg_idx].end,
                segments=chunk_segs,
            )
        )

        chunk_idx += 1
        pos += stride

    return chunks
