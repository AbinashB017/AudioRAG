"""
evaluate.py — Stage 7: Evaluation Harness (Phase 3)

Stub module. Real implementation arrives in Phase 3.
Signatures and data models are defined now so run_pipeline.py and
other modules can import from here without breaking when Phase 3 fills
in the logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .chunk import Chunk
from .generate import Answer


@dataclass
class EvalCase:
    """One evaluation test case."""

    id: str
    query: str
    notes: str = ""   # what a correct answer should cover (for manual grading)
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Output of running one EvalCase through the full retrieve+generate pipeline."""

    case_id: str
    query: str
    answer: Answer
    retrieved_chunks: list[Chunk]
    latency_sec: float


def load_cases(cases_path: Path) -> list[EvalCase]:
    """Load evaluation cases from eval/cases.json.

    Expected format:
        [{"id": "q01", "query": "...", "notes": "...", "tags": [...]}]
    """
    raise NotImplementedError("Phase 3: implement load_cases()")


def run_eval(
    cases_path: Path,
    output_dir: Path,
    config,  # Config — imported lazily to avoid circular imports
) -> list[EvalResult]:
    """Run all cases through retrieve+generate; write raw results to output_dir.

    Each run produces:
      output_dir/results_<timestamp>.json  — full raw inputs/outputs for inspection

    Args:
        cases_path:  Path to eval/cases.json.
        output_dir:  Directory where run artifacts are written.
        config:      Runtime configuration.

    Returns:
        List of EvalResult objects (also written to disk).
    """
    raise NotImplementedError("Phase 3: implement run_eval()")


if __name__ == "__main__":
    from pathlib import Path
    from .config import load_config

    run_eval(
        cases_path=Path("eval/cases.json"),
        output_dir=Path("eval/results"),
        config=load_config(),
    )
