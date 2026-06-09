"""
docker_runner — Python package for running deglib_evp_task1 in Docker.

Public API
----------
Task1Runner
    High-level interface: download dataset, start container, stream logs.

Task1Result
    Structured result of one run (timing phases, recall values, etc.).

Task1LogParser
    Low-level line-by-line parser (useful if you integrate log streaming
    yourself).

Quick start::

    from docker_runner import Task1Runner

    runner = Task1Runner()

    # First time only — build the image from the project Dockerfile
    # runner.build_image()

    result = runner.run(mode="mode4", size="small", threads=8, max_dist=200, evp_k=50)

    print(f"Recall@15 : {result.best_recall:.4f}")
    print(f"Overall   : {result.overall_time_s:.1f} s")
    print(f"Exit code : {result.exit_code}")
"""

from .log_parser import Task1LogParser, Task2LogParser
from .result import Task1Result, Task2Result
from .runner import Task1Runner, Task2Runner

__all__ = [
    "Task1Runner",
    "Task1Result",
    "Task1LogParser",
    "Task2Runner",
    "Task2Result",
    "Task2LogParser",
]
