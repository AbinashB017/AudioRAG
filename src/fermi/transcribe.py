"""
transcribe.py -- Stage 1: Audio -> Timestamped Segments via OpenRouter Whisper

Pipeline per episode:
  1. Check cache -> return immediately if data/transcripts/<episode_id>.json exists
  2. Get audio duration via ffprobe
  3. Split into ?chunk_minutes clips via ffmpeg (stream-copy, no re-encoding)
  4. POST each clip to OpenRouter /audio/transcriptions with verbose_json
  5. Apply time_offset to each segment so timestamps are episode-absolute
  6. Stitch all segments into one ordered list
  7. Write cache JSON
  8. Delete temp clip files (only on success -- preserve on failure for inspection)

Key design:
  - ffmpeg stream-copy means no audio re-encoding: fast, lossless, ~26ms boundary imprecision
    (completely acceptable for ASR purposes)
  - time_offset_sec = i * chunk_minutes * 60 makes stitching exact and verifiable
  - Cache prevents re-spend on re-runs (budget protection)
  - Cleanup only on success so a failed run's chunks stay for debugging
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from .config import Config


@dataclass
class Segment:
    """One Whisper output segment with episode-absolute timestamps."""

    start: float  # seconds from episode start (after time_offset applied)
    end: float
    text: str


# -- helpers -----------------------------------------------------------------


def _safe_stem(stem: str) -> str:
    """Filesystem-safe name for temp chunk files, max 50 chars."""
    return re.sub(r"[^\w]", "_", stem)[:50]


def _fmt_time(sec: float) -> str:
    m, s = int(sec // 60), int(sec % 60)
    return f"{m:02d}:{s:02d}"


def check_ffmpeg(config: "Config | None" = None) -> None:
    """Raise RuntimeError with install guidance if ffmpeg/ffprobe are not found.
    
    Checks config.ffmpeg_bin / config.ffprobe_bin first (can be full paths),
    falling back to PATH lookup of 'ffmpeg' / 'ffprobe'.
    """
    from .config import Config as _Config  # local import avoids circular at module level
    bins = {
        "ffmpeg": (config.ffmpeg_bin if config else "ffmpeg"),
        "ffprobe": (config.ffprobe_bin if config else "ffprobe"),
    }
    for tool, exe in bins.items():
        try:
            subprocess.run([exe, "-version"], capture_output=True, check=True)
        except FileNotFoundError:
            raise RuntimeError(
                f"'{exe}' not found.\n"
                f"Install ffmpeg: https://ffmpeg.org/download.html\n"
                f"  Windows: winget install ffmpeg  OR set FFMPEG_BIN=<full path> in .env\n"
                f"  macOS:   brew install ffmpeg\n"
                f"  Linux:   sudo apt install ffmpeg"
            )


def get_audio_duration(audio_path: Path, ffprobe_bin: str = "ffprobe") -> float:
    """Return duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            ffprobe_bin, "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


# -- public pipeline functions ------------------------------------------------


def split_audio(
    audio_path: Path,
    chunk_minutes: int,
    output_dir: Path,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> list[Path]:
    """Split audio into clips of ?chunk_minutes using ffmpeg stream-copy.

    Returns paths in temporal order. The time offset for chunk i is exactly
    i * chunk_minutes * 60 seconds -- used by transcribe_chunk for stitching.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(audio_path.stem)
    chunk_sec = chunk_minutes * 60
    duration = get_audio_duration(audio_path, ffprobe_bin)

    paths: list[Path] = []
    start = 0.0
    idx = 0
    while start < duration - 0.5:  # 0.5 s tolerance for float rounding
        out = output_dir / f"{stem}_c{idx:03d}.mp3"
        subprocess.run(
            [
                ffmpeg_bin, "-y",
                "-i", str(audio_path),
                "-ss", f"{start:.3f}",
                "-t", f"{chunk_sec:.3f}",
                "-acodec", "copy",
                str(out),
            ],
            capture_output=True,
            check=True,
        )
        paths.append(out)
        start += chunk_sec
        idx += 1

    return paths


def transcribe_chunk(
    chunk_path: Path,
    time_offset_sec: float,
    config: Config,
    retries: int = 2,
) -> list[Segment]:
    """POST one audio clip to OpenRouter Whisper; return episode-absolute Segments.

    time_offset_sec is added to every segment timestamp so that stitching
    across chunks produces continuously correct episode-level timestamps.
    Retries up to `retries` times on transient failures.
    """
    last_err: Exception | None = None

    for attempt in range(retries + 1):
        try:
            with open(chunk_path, "rb") as f:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {config.openrouter_api_key}"},
                    files={"file": (chunk_path.name, f, "audio/mpeg")},
                    data={"model": config.asr_model, "response_format": "verbose_json"},
                    timeout=300,
                )
            resp.raise_for_status()
            data = resp.json()

            return [
                Segment(
                    start=float(seg["start"]) + time_offset_sec,
                    end=float(seg["end"]) + time_offset_sec,
                    text=seg["text"].strip(),
                )
                for seg in data.get("segments", [])
                if seg.get("text", "").strip()
            ]

        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = 2 ** attempt
                print(f"    ? retry {attempt + 1}/{retries} in {wait}s ({type(e).__name__}: {e})")
                time.sleep(wait)

    raise RuntimeError(
        f"Transcription failed after {retries + 1} attempts"
    ) from last_err


def transcribe_episode(
    audio_path: Path,
    cache_dir: Path,
    config: Config,
    episode_id: str | None = None,
) -> list[Segment]:
    """Transcribe a full episode; return episode-absolute Segments.

    Cache behaviour:
      - If cache_dir/<episode_id>.json exists, load and return it immediately
        (no API calls, no cost).
      - Otherwise: split -> transcribe each chunk -> stitch -> write cache.

    Temp file cleanup:
      - Chunk files in data/audio_chunks/<episode_id>/ are deleted only after
        the cache write succeeds. On failure they are preserved for inspection.

    Args:
        audio_path:   Path to source MP3 file.
        cache_dir:    Directory for per-episode JSON transcript cache.
        config:       Runtime configuration.
        episode_id:   Override episode ID; if None, derived from filename number.

    Returns:
        Ordered list of Segment objects with episode-absolute timestamps.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    ep_id = episode_id or _episode_id_from_path(audio_path)
    cache_path = cache_dir / f"{ep_id}.json"

    # -- Cache hit ------------------------------------------------------------
    if cache_path.exists():
        print(f"  Cache hit: {cache_path.name} -- skipping ASR")
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return [Segment(**s) for s in raw["segments"]]

    # -- Cache miss -> transcribe ----------------------------------------------
    check_ffmpeg(config)

    duration = get_audio_duration(audio_path, config.ffprobe_bin)
    chunk_sec = config.chunk_minutes * 60
    n_chunks = int(duration / chunk_sec) + 1
    print(
        f"  Duration: {_fmt_time(duration)} -> splitting into {n_chunks} chunk(s) "
        f"of {config.chunk_minutes} min"
    )

    chunk_dir = config.audio_chunk_dir / ep_id
    chunk_paths = split_audio(
        audio_path, config.chunk_minutes, chunk_dir,
        ffmpeg_bin=config.ffmpeg_bin, ffprobe_bin=config.ffprobe_bin,
    )

    all_segments: list[Segment] = []

    try:
        for i, chunk_path in enumerate(chunk_paths):
            offset = i * chunk_sec
            end_time = min(offset + chunk_sec, duration)
            print(
                f"  Chunk {i + 1}/{len(chunk_paths)} "
                f"({_fmt_time(offset)} -> {_fmt_time(end_time)})... ",
                end="",
                flush=True,
            )
            segs = transcribe_chunk(chunk_path, offset, config)
            all_segments.extend(segs)
            print(f"[OK]  ({len(segs)} segments)")

    except Exception:
        print(
            f"\n  [X] Transcription failed. "
            f"Temp chunks preserved at {chunk_dir} for inspection."
        )
        raise

    # -- Write cache -----------------------------------------------------------
    cache_data = {
        "episode_id": ep_id,
        "audio_path": str(audio_path),
        "duration_sec": round(duration, 3),
        "chunk_minutes": config.chunk_minutes,
        "n_chunks": len(chunk_paths),
        "n_segments": len(all_segments),
        "segments": [asdict(s) for s in all_segments],
    }
    cache_path.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
    print(f"  Cached -> {cache_path.name}  ({len(all_segments)} segments)")

    # -- Cleanup: only on success ---------------------------------------------
    for p in chunk_paths:
        if p.exists():
            p.unlink()
    try:
        chunk_dir.rmdir()  # succeeds only if now empty
    except OSError:
        pass
    print(f"  Cleaned {len(chunk_paths)} temp chunk file(s)")

    return all_segments


# -- filename utilities (used by pipeline scripts) ----------------------------


def episode_id_from_path(p: Path) -> str:
    """Derive episode ID from filename, e.g. 'Great Papers 03 - ...' -> 'ep03'."""
    m = re.search(r"(\d+)", p.stem)
    return f"ep{int(m.group(1)):02d}" if m else p.stem[:8]


def episode_title_from_path(p: Path) -> str:
    """Extract human-readable title from filename, fixing underscore apostrophes."""
    stem = p.stem
    title = stem.split(" - ", 1)[1] if " - " in stem else stem
    title = re.sub(r"(\w)_s\b", r"\1's", title)  # "Einstein_s" -> "Einstein's"
    return title.strip()


# keep private alias for internal use
_episode_id_from_path = episode_id_from_path
