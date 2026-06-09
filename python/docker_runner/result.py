"""
result.py — Task1Result dataclass

Holds all structured information extracted from a single deglib_evp_task1
container run: timing phases, recall values, SIMD info, and raw logs.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task1Result:
    """Structured result of one deglib_evp_task1 container run."""

    # Run identity
    mode: str
    dataset_size: str  # "small" | "large"
    exit_code: int

    # SIMD instruction set reported by the binary (e.g. "AVX2, SSE")
    simd_info: str = ""

    # ------------------------------------------------------------------ #
    # Timing — all in seconds; None if the phase was not reported         #
    # ------------------------------------------------------------------ #
    load_time_s: float | None = None
    quant_time_s: float | None = None
    build_time_s: float | None = None
    convert_time_s: float | None = None
    explore_time_s: float | None = None
    rerank_time_s: float | None = None
    overall_time_s: float | None = None

    # ------------------------------------------------------------------ #
    # Recall results                                                       #
    # Each entry is (max_dist, recall_value) as parsed from the log.      #
    # Multiple entries appear when --max-dist is a comma-separated list.  #
    # ------------------------------------------------------------------ #
    recall_results: list[tuple[int, float]] = field(default_factory=list)

    # Raw log lines (stderr + stdout combined in arrival order)
    raw_logs: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Convenience accessors                                                #
    # ------------------------------------------------------------------ #

    @property
    def best_recall(self) -> float | None:
        """Highest recall value across all max_dist sweep points."""
        if not self.recall_results:
            return None
        return max(r for _, r in self.recall_results)

    @property
    def last_recall(self) -> float | None:
        """Recall from the last reported sweep point (highest max_dist)."""
        if not self.recall_results:
            return None
        return self.recall_results[-1][1]

    @property
    def succeeded(self) -> bool:
        """True when the container exited cleanly (exit code 0)."""
        return self.exit_code == 0

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation / logging."""
        return {
            "mode": self.mode,
            "dataset_size": self.dataset_size,
            "exit_code": self.exit_code,
            "simd_info": self.simd_info,
            "load_time_s": self.load_time_s,
            "quant_time_s": self.quant_time_s,
            "build_time_s": self.build_time_s,
            "convert_time_s": self.convert_time_s,
            "explore_time_s": self.explore_time_s,
            "rerank_time_s": self.rerank_time_s,
            "overall_time_s": self.overall_time_s,
            "recall_results": self.recall_results,
            "best_recall": self.best_recall,
        }

    def __repr__(self) -> str:
        recall_str = (
            f"{self.best_recall:.4f}" if self.best_recall is not None else "n/a"
        )
        time_str = (
            f"{self.overall_time_s:.1f}s" if self.overall_time_s is not None else "n/a"
        )
        return (
            f"Task1Result(mode={self.mode!r}, size={self.dataset_size!r}, "
            f"recall={recall_str}, time={time_str}, exit={self.exit_code})"
        )


@dataclass
class Task2Result:
    """Structured result of one deglib_evp_task2 container run."""

    # Run identity
    mode: str
    dataset_size: str  # "default"
    exit_code: int

    # SIMD instruction set reported by the binary (e.g. "AVX2, SSE")
    simd_info: str = ""

    # ------------------------------------------------------------------ #
    # Timing — all in seconds; None if the phase was not reported         #
    # ------------------------------------------------------------------ #
    load_time_s: float | None = None
    quant_time_s: float | None = None
    build_time_s: float | None = None
    convert_time_s: float | None = None
    explore_time_s: float | None = None
    rerank_time_s: float | None = None
    flas_time_s: float | None = None
    overall_time_s: float | None = None

    # ------------------------------------------------------------------ #
    # Recall results                                                       #
    # Each entry is (max_dist, recall_value) as parsed from the log.      #
    # Multiple entries appear when --max-dist is a comma-separated list.  #
    # ------------------------------------------------------------------ #
    recall_results: list[tuple[int, float]] = field(default_factory=list)

    # Detailed sweep points: list of dicts with {eps_search, max_dist, recall, search_time_ms}
    sweep_points: list[dict[str, Any]] = field(default_factory=list)

    # Raw log lines (stderr + stdout combined in arrival order)
    raw_logs: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Convenience accessors                                                #
    # ------------------------------------------------------------------ #

    @property
    def best_recall(self) -> float | None:
        """Highest recall value across all max_dist sweep points."""
        if not self.recall_results:
            return None
        return max(r for _, r in self.recall_results)

    @property
    def last_recall(self) -> float | None:
        """Recall from the last reported sweep point (highest max_dist)."""
        if not self.recall_results:
            return None
        return self.recall_results[-1][1]

    @property
    def succeeded(self) -> bool:
        """True when the container exited cleanly (exit code 0)."""
        return self.exit_code == 0

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation / logging."""
        return {
            "mode": self.mode,
            "dataset_size": self.dataset_size,
            "exit_code": self.exit_code,
            "simd_info": self.simd_info,
            "load_time_s": self.load_time_s,
            "quant_time_s": self.quant_time_s,
            "build_time_s": self.build_time_s,
            "convert_time_s": self.convert_time_s,
            "explore_time_s": self.explore_time_s,
            "rerank_time_s": self.rerank_time_s,
            "flas_time_s": self.flas_time_s,
            "overall_time_s": self.overall_time_s,
            "recall_results": self.recall_results,
            "sweep_points": self.sweep_points,
            "best_recall": self.best_recall,
        }

    def __repr__(self) -> str:
        recall_str = (
            f"{self.best_recall:.4f}" if self.best_recall is not None else "n/a"
        )
        time_str = (
            f"{self.overall_time_s:.1f}s" if self.overall_time_s is not None else "n/a"
        )
        return (
            f"Task2Result(mode={self.mode!r}, size={self.dataset_size!r}, "
            f"recall={recall_str}, time={time_str}, exit={self.exit_code})"
        )
