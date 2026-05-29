"""
log_parser.py — Real-time log parser for deglib_evp_task1 output.

The binary writes all human-readable output to stderr (fprintf(stderr, ...)).
This parser processes lines as they stream from the container and fills a
Task1Result with structured timing and recall data.

Typical log output (stderr) looks like:

    SIMD: AVX2, SSE
    load time        :   0.6 s
    quantization time:   0.8 s
    build time       :   4.8 s
    convert time     :   0.2 s
    explore time     :   3.8 s
    rerank time      :   0.9 s
    Recall@15        : 0.8249  (max_dist=200)
    overall time     :  10.2 s

The exact format depends on the mode and is defined in the mode*.h headers.
All patterns use case-insensitive prefix matching to be robust against
minor formatting changes.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .result import Task1Result

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# SIMD: AVX-512, AVX2, SSE  |  SIMD: AVX2, SSE  |  SIMD: none (scalar)
_RE_SIMD = re.compile(r"SIMD\s*:\s*(.+)", re.IGNORECASE)

# "load time        :   0.6 s"  →  group(1)="load", group(2)="0.6"
_RE_TIME = re.compile(
    r"(load|quantization|quant|quantize|build|convert|conversion|explore|rerank|overall|total\s+elapsed)\s*time\s*:\s*([\d.]+)\s*s",
    re.IGNORECASE,
)

# "Recall@15        : 0.8249  (max_dist=200)"
# Handles optional trailing "(max_dist=NNN)" annotation
_RE_RECALL = re.compile(
    r"Recall@\d+\s*:\s*([\d.]+)(?:.*?max_dist\s*=\s*(\d+))?",
    re.IGNORECASE,
)

# "Error: ..." or "Fatal error: ..."
_RE_ERROR = re.compile(r"(fatal\s+)?error\s*:", re.IGNORECASE)

# "Warning: ..."
_RE_WARNING = re.compile(r"warning\s*:", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Task1LogParser:
    """
    Stateful, line-by-line parser for deglib_evp_task1 log output.

    Usage::

        parser = Task1LogParser()
        for line in container_log_stream:
            parser.feed(line)
        result = parser.build_result(mode="mode4", dataset_size="small", exit_code=0)
    """

    def __init__(self, *, echo: bool = True) -> None:
        """
        Parameters
        ----------
        echo:
            When True, each incoming line is printed to stdout in real time
            so the user can watch progress without waiting for the container
            to finish.
        """
        self._echo = echo
        self._raw_lines: list[str] = []
        self._errors: list[str] = []
        self._warnings: list[str] = []

        # Populated as lines arrive
        self.simd_info: str = ""
        self.load_time_s: float | None = None
        self.quant_time_s: float | None = None
        self.build_time_s: float | None = None
        self.convert_time_s: float | None = None
        self.explore_time_s: float | None = None
        self.rerank_time_s: float | None = None
        self.overall_time_s: float | None = None
        # List of (max_dist, recall) — may have multiple entries from a sweep
        self.recall_results: list[tuple[int, float]] = []
        # Running max_dist counter used when the annotation is absent
        self._last_max_dist: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, line: str) -> None:
        """Process one log line."""
        line = line.rstrip("\r\n")
        self._raw_lines.append(line)

        if self._echo:
            print(line, flush=True)

        self._parse_line(line)

    def build_result(
        self,
        mode: str,
        dataset_size: str,
        exit_code: int,
    ) -> "Task1Result":
        """Build and return a Task1Result from all lines seen so far."""
        from .result import Task1Result  # local import to avoid circular

        return Task1Result(
            mode=mode,
            dataset_size=dataset_size,
            exit_code=exit_code,
            simd_info=self.simd_info,
            load_time_s=self.load_time_s,
            quant_time_s=self.quant_time_s,
            build_time_s=self.build_time_s,
            convert_time_s=self.convert_time_s,
            explore_time_s=self.explore_time_s,
            rerank_time_s=self.rerank_time_s,
            overall_time_s=self.overall_time_s,
            recall_results=list(self.recall_results),
            raw_logs=list(self._raw_lines),
        )

    @property
    def errors(self) -> list[str]:
        return list(self._errors)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_line(self, line: str) -> None:
        # SIMD info
        m = _RE_SIMD.search(line)
        if m:
            self.simd_info = m.group(1).strip()
            return

        # Timing phases
        m = _RE_TIME.search(line)
        if m:
            phase = m.group(1).lower()
            value = float(m.group(2))
            self._set_time(phase, value)
            return

        # Recall@K
        m = _RE_RECALL.search(line)
        if m:
            recall_val = float(m.group(1))
            if m.group(2):
                max_dist = int(m.group(2))
            else:
                # No annotation — use a monotonically increasing counter
                self._last_max_dist += 1
                max_dist = self._last_max_dist
            self.recall_results.append((max_dist, recall_val))
            return

        # Errors
        if _RE_ERROR.search(line):
            self._errors.append(line)
            return

        # Warnings
        if _RE_WARNING.search(line):
            self._warnings.append(line)

    def _set_time(self, phase: str, value: float) -> None:
        if phase in ("load",):
            self.load_time_s = value
        elif phase in ("quantization", "quant", "quantize"):
            self.quant_time_s = value
        elif phase in ("build",):
            self.build_time_s = value
        elif phase in ("convert", "conversion"):
            self.convert_time_s = value
        elif phase in ("explore",):
            self.explore_time_s = value
        elif phase in ("rerank",):
            self.rerank_time_s = value
        elif phase in ("overall", "total elapsed"):
            self.overall_time_s = value
