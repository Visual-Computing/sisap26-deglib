# SISAP 2026 — Wikipedia BGE-M3 Similarity Search (Task 1)

This repository implements a solution for [**SISAP 2026 Challenge — Task 1: K-Nearest Neighbor Graph**](https://sisap-challenges.github.io/2026/) using the [**Dynamic Exploration Graph (DEG)**](https://github.com/Visual-Computing/DynamicExplorationGraph/tree/evp) library with [**EVP (Equi-Voronoi Polytope) quantisation**](https://github.com/MetricSearch/metric_space_rust).

The primary execution path is **Docker-based**: a single container clones, builds, and runs the optimised C++ binary `deglib_evp_task1`, while a Python runner handles dataset download, container lifecycle, and result parsing.

**Table of Contents**  
- [Task Overview](#-task-overview)
- [Dataset](#-dataset)
- [Quick Start](#-quick-start)
- [Docker Details](#-docker-details)
- [Python Runner API](#-python-runner-api)
- [Benchmark Modes](#-benchmark-modes)
- [CLI Parameters](#-cli-parameters)
- [Benchmark Results](#-benchmark-results)
- [Project Structure](#-project-structure)
- [Smoke Test](#-smoke-test)
- [Troubleshooting](#-troubleshooting)


## 🏆 Task Overview

| Parameter | Value |
|---|---|
| Dataset | [WIKIPEDIA BGE-M3](https://huggingface.co/datasets/SISAP-Challenges/SISAP2026) (6.35 M vectors, 1024-dim, normalised, FP16) |
| Task | Approximate k-NN graph for k=15 (metric self-join) |
| Distance metric | Dot product (inner product) |
| Target recall | ≥ 0.8 average |
| Evaluation | Wall-clock time (load + build + search + postprocessing) |
| Hardware limit | **8 vCPUs, 24 GB RAM**, read-only dataset mount, 8 hours max |



## 📦 Dataset

The [Wikipedia BGE-M3 dataset](https://huggingface.co/datasets/SISAP-Challenges/SISAP2026) is downloaded automatically by the Python runner on first use (cached locally via HuggingFace Hub).

| Size | File | Vectors |
|---|---|---|
| **small** (development) | `benchmark-dev-wikipedia-bge-m3-small.h5` | 200,000 |
| **large** (evaluation) | `benchmark-dev-wikipedia-bge-m3.h5` | 6,350,000 |

See [main.py](main.py) for a working end-to-end example.



## 🚀 Quick Start

### Prerequisites
- [Docker Desktop](https://docs.docker.com/get-docker/) (or Docker Engine on Linux)
- [uv](https://astral.sh/uv) (handles Python installation automatically)

### 1. Install Python dependencies
```bash
git clone https://github.com/Visual-Computing/sisap26-deglib
cd sisap26-deglib
uv sync
```

### 2. Run — via Python (recommended)
The Python runner handles image build, dataset download, container lifecycle, and logging — all in one step.

```bash
uv run python main.py
```

Or in your own script:
```python
from docker_runner import Task1Runner

runner = Task1Runner()
runner.build_image()        # builds the Docker image only if missing

# Automatically downloads the small dataset from HuggingFace (first run only)
result = runner.run(mode="mode4", size="small", threads=8, max_dist=200, evp_k=50)

print(f"Recall@15 : {result.best_recall:.4f}")
print(f"Overall   : {result.overall_time_s:.1f} s")
```

### 3. Run — manually (docker build + docker run)
Only needed if you want to run the container without Python, e.g. in a restricted environment:

```bash
docker build -t sisap26-deglib .
```

```bash
docker run \
    --cpus=8 \
    --memory=24g \
    --memory-swap=24g \
    --memory-swappiness=0 \
    --volume "$(python -c 'from docker_runner import Task1Runner; print(Task1Runner().get_data_dir())'):/data:ro" \
    --volume "$(pwd)/results:/results:rw" \
    sisap26-deglib \
    /data/benchmark-dev-wikipedia-bge-m3-small.h5 evp-rerank \
    --threads 8 --max-dist 200 --evpK 50
```



## 🐳 Docker Details

### Multi-stage Dockerfile

| Stage | Base | Purpose |
|---|---|---|
| `builder` | `ubuntu:24.04` | Clones repo, installs `cmake g++`, compiles with `-march=native` |
| `runtime` | `ubuntu:24.04` | Copies only the binary; minimal footprint |

**CMake flags used:**
```
-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"
```
- `-march=native` → compiles the binary optimized specifically for the host CPU (enabling AVX2, AVX-512, etc. depending on machine capabilities)

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



## 🐍 Python Runner API

### Full example (see also [main.py](main.py))

```python
from pathlib import Path
from docker_runner import Task1Runner

runner = Task1Runner(
    image_tag="sisap26-deglib",        # Docker image name
    results_dir=Path("./results"),        # local folder → mounted as /results:rw
    echo_logs=True,                       # stream container logs to stdout
)

# Build the image (skipped if already exists; use force=True to rebuild)
runner.build_image(force=False)

# Run mode4 on the small dataset with a max_dist sweep
result = runner.run(
    mode="mode4",
    size="small",                         # downloads dataset automatically if needed
    threads=8,
    max_dist="100,200,300",              # comma-separated → recall sweep
    evp_k=50,
)

# Structured results
if result.succeeded:
    print(f"SIMD          : {result.simd_info}")
    print(f"Load          : {result.load_time_s:.1f} s")
    print(f"Quantisation  : {result.quant_time_s:.1f} s")
    print(f"Build         : {result.build_time_s:.1f} s")
    print(f"Explore       : {result.explore_time_s:.1f} s")
    print(f"Rerank        : {result.rerank_time_s:.1f} s")
    print(f"Total         : {result.overall_time_s:.1f} s")
    print()
    for max_dist, recall in result.recall_results:
        print(f"  max_dist={max_dist:>4}  →  Recall@15 = {recall:.4f}")
else:
    print(f"Container failed with exit code {result.exit_code}")
```

### `runner.build_image()` parameters

| Parameter | Default | Description |
|---|---|---|
| `tag` | `None` | Override the image tag. Defaults to the tag passed to the constructor. |
| `force` | `False` | When `True`, force a fresh build with all caches disabled. |

### `runner.run()` parameters

| Parameter | Default | Description |
|---|---|---|
| `mode` | — | Benchmark mode: `"mode1"`–`"mode7"` or alias (e.g. `"evp-rerank"`) |
| `size` | `"small"` | Dataset: `"small"` (200 K) or `"large"` (6.35 M) |
| `threads` | `8` | `--threads` |
| `non_zeros` | `600` | `--non-zeros` (EVP sparsity) |
| `k_top` | `15` | `--k-top` |
| `k_graph` | `32` | `--k-graph` (graph degree) |
| `max_dist` | `200` | `--max-dist` — int or comma-separated sweep string |
| `evp_k` | `50` | `--evpK` — candidate pool for reranking |
| `prune_worst` | `16` | `--prune-worst` |
| `no_recall` | `False` | `--no-recall` |
| `output` | `None` | `--output <path inside container>` |
| `graph` | `None` | `--graph <path inside container>` |

For detailed descriptions of each flag see [CLI Parameters](#-cli-parameters).

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



## 🔧 Benchmark Modes

| Mode | Name | Description |
|---|---|---|
| `mode1` | `fp16` / `fp16-build-fp16-explore` | FP16 build + FP16 explore |
| `mode2` | `evp-linear` / `evp-linear-search` | EVP quantisation + brute-force linear search |
| `mode3` | `evp` / `evp-build-evp-explore` | EVP build + EVP explore (no rerank) |
| `mode4` | `evp-rerank` / `evp-build-evp-explore-fp16-rerank` | EVP build + EVP explore + FP16 rerank ⭐ |
| `mode5` | `evp-build-fp16-external-search` | EVP build + FP16 external graph search |
| `mode6` | `evp-asymmetric` / `evp-build-fp16-asymmetric-search` | EVP build + asymmetric FP16-vs-EVP search |
| `mode7` | `evp-asymmetric-rerank` / `evp-build-fp16-asymmetric-search-rerank` | EVP build + asymmetric search + FP16 rerank ⭐ |

⭐ Best recall/speed trade-off for the SISAP constraint (24 GB RAM, 8 CPU).

## 🔧 CLI Parameters

```
  --threads <n>      Number of CPU worker threads used for parallel EVP quantization,
                     even-regular graph construction, and query exploration (default: 6).
  --non-zeros <n>    EVP Quantization sparsity parameter. Specifies the exact number of non-zero
                     elements in each quantized sparse vector (default: 600).
  --k-top <n>        The final number of nearest neighbors (top-K) retrieved per query
                     and evaluated for recall or written to the output file (default: 15).
  --k-graph <n>      The degree of the regular graph. Specifies the exact number of edges
                     (neighbors) per vertex. Must be an even integer >= 4 (default: 32).
  --k-ext <n>        The search size (k-top parameter) used during graph construction. Decides
                     how many good neighbors are shown to each newly added node, from which it
                     selects nodes to connect with up to k-graph (default: 32).
  --eps-ext <f>      Search expansion parameter used together with k-ext during graph construction.
                     Decides if nodes whose distance is slightly worse (e.g. 0.01 = 1%) than the
                     current worst in the search list should be explored (default: 0.001).
  --no-recall        Disables loading ground-truth datasets and calculating Recall@K metrics.
                     Required when exporting search results to an output file.
  --output <path>    Path to a binary `.ivecs` file where the retrieved nearest-neighbor indices
                     will be saved (one row per query; uint32_t count followed by indices).
  --evpK <list>      Candidate pool size or comma-separated list of sizes. Graph search retrieves `evpK` candidates
                     which are then reranked using exact FP16 inner product. Used in Mode 4 and Mode 7 (default: 50).
  --max-dist <list>  Exploration search budget or comma-separated list of budgets. Specifies the maximum number of
                     distance computations allowed per query. Main parameter to trade search speed for recall (default: 200).
  --graph <path>     File path to save the pre-built graph to, or load a pre-built graph from
                     to bypass the construction phase.
  --prune-worst <n>  Number of worst (least similar) neighbors per vertex to replace with self-loops.
                     Applies to all modes (default: 16).
```

## 📊 Benchmark Results

To reproduce the small-dataset table on your own hardware:
```bash
uv run python benchmark_task1_small.py
```

### Small dataset (200K vectors, AMD Ryzen 5 5600G, AVX2, 32 GB RAM)

| Mode | Method | Settings | Load | Quant | Build | Convert | Explore | Rerank | Total | Recall |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | FP16 Build+Explore | `M=32, MaxDist=100` | 0.6 s | 0.0 s | 18.9 s | 0.1 s | 1.5 s | 0.0 s | 21.3 s | 0.8539 |
| 2 | EVP linear search | — | 0.6 s | 0.8 s | 0.0 s | 0.0 s | 209.3 s | 0.0 s | 211 s | 0.7124 |
| 3 | EVP Build+Explore | `M=32, MaxDist=200` | 0.6 s | 0.8 s | 4.8 s | 0.0 s | 0.9 s | 0.0 s | 7.1 s | 0.6814 |
| 4 | EVP Build+Explore+Rerank | `M=32, MaxDist=200, evpK=50` | 0.6 s | 0.8 s | 4.8 s | 0.0 s | 1.2 s | 0.9 s | 8.3 s | 0.8418 |
| 5 | EVP build+FP16 Explore | `M=32, MaxDist=200` | 0.6 s | 0.8 s | 4.8 s | 0.2 s | 3.4 s | 0.0 s | 10.0 s | 0.8483 |
| 6 | EVP build+Asym Explore | `M=32, MaxDist=200` | 0.6 s | 0.8 s | 4.8 s | 0.0 s | 1.3 s | 0.0 s | 7.5 s | 0.7376 |
| 7 | EVP build+Asym+Rerank | `M=32, MaxDist=200, evpK=50` | 0.6 s | 0.8 s | 4.8 s | 0.0 s | 1.6 s | 0.9 s | 8.7 s | 0.8483 |

### Large dataset (6.35M vectors, AMD Ryzen 5 5600G, AVX2, 32 GB RAM)

| Mode | Method | Settings | Load | Quant | Build | Convert | Explore | Rerank | Total | Recall |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | FP16 Build+Explore | `M=32, MaxDist=100` | 15 s | 0 s | 816 s | 0 s | 55 s | 0 s | 886 s | 0.7686 |
| 3 | EVP Build+Explore | `M=32, MaxDist=200` | 15 s | 22 s | 265 s | 0 s | 40 s | 0 s | 340 s | 0.6447 |
| 4 | EVP Build+Explore+Rerank | `M=32, MaxDist=200, evpK=50` | 15 s | 22 s | 265 s | 0 s | 55 s | 20 s | 377 s | 0.7632 |
| 5 | EVP build+FP16 Explore | `M=32, MaxDist=200` | 15 s | 22 s | 265 s | 4 s | 125 s | 0 s | 431 s | 0.7687 |
| 6 | EVP build+Asym Explore | `M=32, MaxDist=200` | 15 s | 22 s | 265 s | 2 s | 57 s | 0 s | 361 s | 0.6880 |
| 7 | EVP build+Asym+Rerank | `M=32, MaxDist=200, evpK=50`  | 15 s | 22 s | 265 s | 2 s | 72 s | 20 s | 396 s | 0.7676 |




## 🗂️ Project Structure

```
sisap26-deglib/
├── Dockerfile                   # Multi-stage: clone evp → cmake → binary
├── main.py                      # End-to-end example using Task1Runner
├── benchmark_small.py            # Reproduce the small-dataset benchmark table
├── PLAN.md                      # Detailed implementation plan
├── SISAP_2026_Task1.md          # SISAP 2026 task specification (German)
│
├── docker_runner/               # Python package for Docker-based runs
│   ├── __init__.py              # Public API: Task1Runner, Task1Result
│   ├── runner.py                # HuggingFace download + container management
│   ├── log_parser.py            # Real-time log parsing (timing, recall)
│   └── result.py                # Task1Result dataclass
│
├── results/
│   └── task1/                   # Output files (mounted as /results:rw)
│       └── .gitkeep
│
├── pyproject.toml               # Python project config (uv)
├── uv.lock
└── .gitignore
```



## 🧪 Smoke Test

Verify the binary runs correctly (no dataset needed):
```bash
docker build -t sisap26-deglib .
docker run --rm sisap26-deglib
# Expected: usage help on stderr, exit code 1
```

## 🔧 Troubleshooting

| Problem | Command |
|---|---|
| **Cached layers verwenden alte Version** — Build erzwingen ohne Cache | `docker build --no-cache -t sisap26-deglib .` oder `runner.build_image(force=True)` |
| **Image erneut bauen** (Cache wo möglich nutzen) | `docker build -t sisap26-deglib .` |
| **Altes Image löschen** | `docker image rm sisap26-deglib` |
| **Alle Zwischen-Images und Bauteile aufräumen** | `docker builder prune -f` |
| **Kompaktes Image — History anzeigen** | `docker history sisap26-deglib:latest` |
| **Container-Logs bei Fehlern ansehen** (nach `exit code`) | `docker logs <container-id>` |
| **Laufenden Container sehen & stoppen** | `docker ps` → `docker stop <container-id>` |
| **Alle Container & Volumes restlos löschen** | `docker system prune -a --volumes` |

