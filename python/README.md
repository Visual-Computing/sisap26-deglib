# deglib — SISAP 2026 Indexing Challenge Submission

This repository contains the Python tooling for the [**SISAP 2026 Indexing Challenge**](https://sisap-challenges.github.io/2026/) submission using the [**Dynamic Exploration Graph (DEG)**](https://github.com/Visual-Computing/DynamicExplorationGraph/tree/evp) in combination with [**EVP (Equi-Voronoi Polytope) quantisation**](https://github.com/MetricSearch/metric_space_rust).

The primary execution path is **Docker-based**: the Python runner handles dataset download, container lifecycle, resource limits, and result parsing. The C++ binary `deglib_sisap` runs inside the container.

**Table of Contents**
- [Task 1 — Wikipedia BGE-M3 (k-NN Graph)](#-task-1--wikipedia-bge-m3-k-nn-graph)
- [Task 2 — Llama-Dev (ANN Search)](#-task-2--llama-dev-ann-search)
- [Quick Start](#-quick-start)
- [Docker Details](#-docker-details)
- [Python Runner API](#-python-runner-api)
- [Benchmark & Submission Scripts](#-benchmark--submission-scripts)
- [Benchmark Results](#-benchmark-results)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)


---

## 🏆 Task 1 — Wikipedia BGE-M3 (k-NN Graph)

| Parameter | Value |
|---|---|
| Dataset | [WIKIPEDIA BGE-M3](https://huggingface.co/datasets/SISAP-Challenges/SISAP2026) — FP16, 1024-dim, normalised |
| Task | Approximate k-NN **graph** for k=15 (metric self-join) |
| Distance metric | Dot product (inner product) |
| Target recall | ≥ 0.8 average |
| Evaluation | Wall-clock time (load + build + explore + postprocessing) |
| Hardware limit | **8 vCPUs, 24 GB RAM**, read-only dataset, 8 hours max |

### Task 1 Datasets

| Size | File | Vectors |
|---|---|---|
| **small** (development) | `benchmark-dev-wikipedia-bge-m3-small.h5` | 200,000 |
| **large** (submission) | `benchmark-dev-wikipedia-bge-m3.h5` | 6,350,000 |

### Task 1 Benchmark Modes

| Mode | Name | Description |
|---|---|---|
| `mode1` | `fp16` | FP16 build + FP16 explore |
| `mode2` | `evp-linear` | EVP quantisation + brute-force linear search |
| `mode3` | `evp` | EVP build + EVP explore (no rerank) |
| `mode4` | `evp-rerank` | EVP build + EVP explore + FP16 rerank ⭐ |
| `mode5` | `evp-build-fp16-external-search` | EVP build + FP16 external graph search |
| `mode6` | `evp-asymmetric` | EVP build + asymmetric FP16-vs-EVP search |
| `mode7` | `evp-asymmetric-rerank` | EVP build + asymmetric search + FP16 rerank ⭐ |

⭐ Best recall/speed trade-off for the SISAP constraint (24 GB RAM, 8 CPU).


---

## 🏆 Task 2 — Llama-Dev (ANN Search)

| Parameter | Value |
|---|---|
| Dataset | [Llama-Dev](https://huggingface.co/datasets/SISAP-Challenges/SISAP2026) — FP32, 128-dim |
| Task | Approximate k-NN **search** for k=30 |
| Distance metric | Inner product (converted to L2 space) |
| Target recall | ≥ 0.8 average |
| Evaluation | Wall-clock time (load + FLAS + build + search) |
| Hardware limit | **8 vCPUs, 24 GB RAM**, graph built **single-threaded**, 8 hours max |

### Task 2 Benchmark Modes

| Mode | Name | Description |
|---|---|---|
| `mode1` | `baseline` | FP32 build + FP32 inner-product explore |
| `mode2` | `fp16-build-fp16-explore` | FP16 build + FP16 IP explore |
| `mode3` | `baseline-fp16` | FP32 build + FP16 IP explore |
| `mode4` | `l2-converted` | FP32 L2(d+1) build + FP32 L2 explore |
| `mode5` | `l2-fp16-ip` | FP32 L2(d+1) build + FP16 IP explore ⭐ |
| `mode6` | `l2-fp16-l2` | FP32 L2(d+1) build + FP16 L2 explore |
| `mode7` | `l2-fp16-d2` | FP32 L2(d+2) build + FP16 L2 explore ⭐ |

⭐ Submission candidates — best recall/speed trade-off.

> **Note:** For Task 2, graph construction is always single-threaded (`build_threads=1`),
> while query exploration uses all available CPU threads.


---

## 🚀 Quick Start

### Prerequisites
- [Docker Desktop](https://docs.docker.com/get-docker/) (or Docker Engine on Linux)
- [uv](https://astral.sh/uv) (handles Python installation automatically)

### Install dependencies
```bash
git clone https://github.com/Visual-Computing/sisap26-deglib
cd sisap26-deglib/python
uv sync
```

### Run a benchmark
```bash
# Task 1 — small dataset
uv run python benchmark_task1_small.py

# Task 1 — large dataset
uv run python benchmark_task1_large.py

# Task 2 — Llama-Dev
uv run python benchmark_task2.py
```

### Run submission scripts
```bash
# Task 1 — small dataset submission configurations
uv run python submission_task1_small.py

# Task 1 — large dataset submission configurations
uv run python submission_task1_large.py

# Task 2 — Mode 5 & Mode 7 with FLAS
uv run python submission_task2.py
```

### Run with AVX2-optimised image
If `FORCE_AVX2` is set as an environment variable, the runner automatically switches to the
`sisap26-deglib-cpp:avx2` Docker image (built with `-DFORCE_AVX2=ON`):

```bash
# Inline — single command only
FORCE_AVX2=1 uv run python submission_task2.py

# Persistent — all commands in this shell session
export FORCE_AVX2=1
uv run python benchmark_task1_small.py
uv run python submission_task1_large.py
```

When `FORCE_AVX2` is set the runner prints:
```
[Task1Runner] FORCE_AVX2 detected — switching image to 'sisap26-deglib-cpp:avx2'
```

> **Note:** The AVX2 image must be built at least once. `build_image()` detects the `:avx2`
> tag and passes `FORCE_AVX2=ON` as a Docker build argument automatically.


---

## 🐳 Docker Details

### Images

| Tag | Build flags | Use case |
|---|---|---|
| `sisap26-deglib-cpp` | default (`-march=native`) | Standard image |
| `sisap26-deglib-cpp:avx2` | `FORCE_AVX2=ON` | Explicit AVX2 (e.g. for deployment on different hardware) |

### Multi-stage Dockerfile

| Stage | Base | Purpose |
|---|---|---|
| `builder` | `ubuntu:24.04` | Clones repo, installs `cmake g++`, compiles |
| `runtime` | `ubuntu:24.04` | Copies only the binary — minimal footprint |

### Container Limits (SISAP-compliant)
```bash
--cpus=8                # 8 vCPUs max
--memory=24g            # 24 GB RAM max
--memory-swap=24g       # no swap beyond RAM limit
--memory-swappiness=0   # disable swap usage
```

### Volume Mounts
| Host | Container | Mode |
|---|---|---|
| HuggingFace cache dir | `/data` | **read-only** |
| `./results` | `/results` | **read-write** |


---

## 🐍 Python Runner API

### Task 1 Runner

```python
from pathlib import Path
from docker_runner import Task1Runner

runner = Task1Runner(results_dir=Path("./results"), echo_logs=True)
runner.build_image(force=False)

# Thread count comes from the container CPU limit (runner.cpu_limit)
num_threads = runner.cpu_limit

result = runner.run(
    mode="evp-rerank",
    size="small",           # "small" (200K) or "large" (6.35M)
    threads=num_threads,
    max_dist="100,200,300", # comma-separated → recall sweep
    evp_k=50,
)

if result.succeeded:
    print(f"Recall@15 : {result.best_recall:.4f}")
    print(f"Total     : {result.overall_time_s:.1f} s")
```

### Task 2 Runner

```python
from pathlib import Path
from docker_runner import Task2Runner

runner = Task2Runner(results_dir=Path("./results"), echo_logs=True)
runner.build_image(force=False)

num_threads = runner.cpu_limit

result = runner.run(
    mode="mode5",
    threads=num_threads,
    build_threads=1,        # Task 2: graph always built single-threaded
    max_dist="5000,6000,7000,8000",
    eps_search="0.18",
    num_runs=10,
    use_flas=True,
)

if result.succeeded:
    print(f"Recall@30 : {result.best_recall:.4f}")
    print(f"Total     : {result.overall_time_s:.1f} s")
```

### `runner.cpu_limit`

Returns the number of CPU threads allocated to the container (derived from the SISAP resource limit `--cpus=8`). Always use this instead of `os.cpu_count()` — the container environment may report a different value.

```python
num_threads = runner.cpu_limit  # → 8
```

### `Task1Result` fields

```python
result.best_recall        # float | None — highest recall across sweep
result.last_recall        # float | None — recall at highest max_dist
result.recall_results     # list[tuple[int, float]] — (max_dist, recall)
result.overall_time_s     # float | None
result.load_time_s        # float | None
result.quant_time_s       # float | None
result.build_time_s       # float | None
result.convert_time_s     # float | None
result.explore_time_s     # float | None
result.rerank_time_s      # float | None
result.simd_info          # str — e.g. "AVX2, SSE"
result.exit_code          # int
result.succeeded          # bool
result.to_dict()          # dict for JSON serialisation
```

### `Task2Result` fields

```python
result.best_recall        # float | None — highest recall across sweep
result.sweep_points       # list[dict] — [{eps_search, max_dist, recall, search_time_ms}]
result.overall_time_s     # float | None
result.load_time_s        # float | None
result.build_time_s       # float | None
result.flas_time_s        # float | None
result.simd_info          # str — e.g. "AVX2, SSE"
result.exit_code          # int
result.succeeded          # bool
result.to_dict()          # dict for JSON serialisation
```


---

## 📊 Benchmark & Submission Scripts

| Script | Task | Dataset | Output |
|---|---|---|---|
| `benchmark_task1_small.py` | Task 1 | small (200K) | `results/benchmark/task1_small/` |
| `benchmark_task1_large.py` | Task 1 | large (6.35M) | `results/benchmark/task1_large/` |
| `benchmark_task2.py` | Task 2 | Llama-Dev | `results/benchmark/task2/` |
| `submission_task1_small.py` | Task 1 | small (200K) | `results/submission/task1_small/` |
| `submission_task1_large.py` | Task 1 | large (6.35M) | `results/submission/task1_large/` |
| `submission_task2.py` | Task 2 | Llama-Dev | `results/submission/task2/` |

Each script saves three artefacts:
- `results.json` — full structured results for further processing
- `results.md` — human-readable Markdown summary table
- `recall_vs_time.png` — Recall vs Search Time plot

---

## 🗂️ Project Structure

```
python/
├── benchmark_task1_small.py     # Benchmark all modes — small dataset
├── benchmark_task1_large.py     # Benchmark all modes — large dataset
├── benchmark_task2.py           # Benchmark all Task 2 modes
├── submission_task1_small.py    # Submission configs — Task 1 small
├── submission_task1_large.py    # Submission configs — Task 1 large
├── submission_task2.py          # Submission configs — Task 2 (Mode 5 & 7 + FLAS)
│
├── docker_runner/               # Python package for Docker-based runs
│   ├── __init__.py              # Public API: Task1Runner, Task2Runner, Task1Result, Task2Result
│   ├── runner.py                # HuggingFace download + container management
│   ├── log_parser.py            # Real-time log parsing (timing, recall)
│   └── result.py                # Task1Result / Task2Result dataclasses
│
├── results/                     # All output files (mounted as /results:rw inside container)
│   ├── benchmark/
│   │   ├── task1_small/
│   │   ├── task1_large/
│   │   └── task2/
│   └── submission/
│       ├── task1_small/
│       ├── task1_large/
│       └── task2/
│
├── pyproject.toml               # Python project config (uv)
├── uv.lock
└── README.md
```


---

## 🔧 Troubleshooting

| Problem | Command |
| --- | --- |
| **Force build without cache** | `docker build --no-cache -t sisap26-deglib-cpp .` or `runner.build_image(force=True)` |
| **Build AVX2 image** | `FORCE_AVX2=1 uv run python benchmark_task1_small.py` (automatically builds `:avx2` image) |
| **Delete old image** | `docker image rm sisap26-deglib-cpp` |
| **Clean up all intermediate images** | `docker builder prune -f` |
| **View container logs** | `docker logs <container-id>` |
| **Stop running container** | `docker ps` → `docker stop <container-id>` |
| **Delete all containers & volumes** | `docker system prune -a --volumes` |
