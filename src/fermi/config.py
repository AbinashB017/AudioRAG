"""
config.py -- Central configuration for Fermi Podcast Companion.

All environment variables and tunable constants flow through here.
No other module reads os.environ directly -- import from here only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    # -- API credentials ------------------------------------------------------
    openrouter_api_key: str

    # -- Model names (controlled by env vars -- switching tiers is one line) ---
    llm_model: str = "anthropic/claude-haiku-4.5"          # LLM_MODEL env var
    asr_model: str = "openai/whisper-large-v3-turbo"       # ASR_MODEL env var
    embed_model: str = "BAAI/bge-small-en-v1.5"            # EMBED_MODEL env var

    # -- Pipeline knobs -------------------------------------------------------
    chunk_minutes: int = 6      # audio split size sent to ASR per request
    target_words: int = 200     # target word count per retrieval chunk
    overlap_words: int = 40     # word overlap between adjacent chunks
    top_k: int = 5              # number of chunks retrieved per query

    # -- Optional tool paths (override PATH lookup) ----------------------------
    ffmpeg_bin: str = "ffmpeg"   # set FFMPEG_BIN env var if ffmpeg isn't in PATH
    ffprobe_bin: str = "ffprobe" # set FFPROBE_BIN env var if not in PATH

    # -- Safety ---------------------------------------------------------------
    budget_usd: float = 10.0    # hard cost ceiling -- used to gate expensive ops

    # -- Filesystem paths (relative to project root) --------------------------
    audio_dir: Path = field(default_factory=lambda: Path("podcasts"))
    transcript_dir: Path = field(default_factory=lambda: Path("data/transcripts"))
    audio_chunk_dir: Path = field(default_factory=lambda: Path("data/audio_chunks"))
    chroma_dir: Path = field(default_factory=lambda: Path("data/chroma_db"))


def load_config(env_path: Path | None = None) -> Config:
    """Load configuration from .env and environment variables.

    Required env var:
        OPENROUTER_API_KEY  -- your OpenRouter API key

    Optional env vars (each overrides its default above):
        LLM_MODEL           -- generation model  (default: anthropic/claude-haiku-4.5)
        ASR_MODEL           -- transcription model (default: openai/whisper-large-v3-turbo)
        EMBED_MODEL         -- local embedding model (default: BAAI/bge-small-en-v1.5)
        CHUNK_MINUTES       -- audio split size, minutes (default: 6)
        TARGET_WORDS        -- words per retrieval chunk (default: 200)
        OVERLAP_WORDS       -- word overlap between chunks (default: 40)
        TOP_K               -- retrieval results per query (default: 5)
        BUDGET_USD          -- hard cost ceiling in USD (default: 10.0)
    """
    load_dotenv(env_path or Path(".env"))

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key, or set it in your shell."
        )

    return Config(
        openrouter_api_key=api_key,
        llm_model=os.environ.get("LLM_MODEL", "anthropic/claude-haiku-4.5"),
        asr_model=os.environ.get("ASR_MODEL", "openai/whisper-large-v3-turbo"),
        embed_model=os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        chunk_minutes=int(os.environ.get("CHUNK_MINUTES", "6")),
        target_words=int(os.environ.get("TARGET_WORDS", "200")),
        overlap_words=int(os.environ.get("OVERLAP_WORDS", "40")),
        top_k=int(os.environ.get("TOP_K", "5")),
        budget_usd=float(os.environ.get("BUDGET_USD", "10.0")),
        ffmpeg_bin=os.environ.get("FFMPEG_BIN", "ffmpeg"),
        ffprobe_bin=os.environ.get("FFPROBE_BIN", "ffprobe"),
    )
