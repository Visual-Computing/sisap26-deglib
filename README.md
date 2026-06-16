# SISAP 2026 — deglib

Submission for the [SISAP 2026 Indexing Challenge](https://sisap-challenges.github.io/2026/).
The index is a [**Dynamic Exploration Graph (DEG)**](https://github.com/Visual-Computing/DynamicExplorationGraph/tree/evp)
combined with [**EVP (Equi-Voronoi Polytope) quantization**](https://github.com/MetricSearch/metric_space_rust) and
[**FLAS (Fast Linear Assignment Sorter)**](https://github.com/Visual-Computing/LAS_FLAS) for optimized insertion order,
implemented in C++ (`cpp/`) and driven by the official baseline Python harness
(`submission/`). Everything ships as a **single Docker container** built by TIRA
from this repo.

## Approach

We configure the even-regular Dynamic Exploration Graph (deglib) library to target the specific constraints and scoring metrics of the two tasks:

- **Task 1** (k-NN self-join, scored on **total build + search time**):
  We build the graph using EVP-quantized representations (`EvpBits` metric). Since this is a self-join, every database element has a corresponding vertex in the graph. We optimize the search by starting the traversal directly at the target vertex's position, bypassing the entry-point routing phase. We evaluate two configurations on this graph:
  - **`mode4` (evp-rerank):** The traversal walks the local neighborhood starting from the target vertex using quantized `EvpBits` distances. The retrieved candidates are then reranked using exact FP16 inner products.
  - **`mode7` (evp-asymmetric-rerank):** The traversal walks the local neighborhood starting from the target vertex using an asymmetric distance function (the vertex's original FP16 vector vs. the EVP-quantized vertices in the graph), followed by exact FP16 inner-product reranking of the retrieved candidates.
- **Task 2** (MIPS search, scored on **query time**):
  To perform maximum inner product search (MIPS) on the deglib graph, we transform the inner product into an L2-similarity search by extending the vectors' dimensionality. We build the graph once and sweep both `eps_search` and `max_dist` on the built graph to produce multiple operating points. We evaluate two configurations:
  - **`mode5` (l2-fp16-ip):** Vectors are extended to $d+1$ dimensions to transform inner product to L2 distance during the build (speeded up by pre-sorting vectors using **FLAS**). The query search is performed using fast FP16 inner-product exploration.
  - **`mode7` (l2-fp16-d2):** Vectors are extended to $d+2$ dimensions for the graph build (also utilizing FLAS). Query search is performed using fast FP16 L2 distance exploration.

The C++ binary computes neighbors **and** distances during search; the thin Python
entrypoint adapts the output to the official result format. 

## Challenge tasks & constraints

Both tasks run under the same hard limits: **8 vCPUs, 24 GB RAM, ≤ 8 h, read-only
dataset, no internet** in the container (the eval node is an AMD EPYC 7F72, no
AVX-512). The goal is **≥ 0.8 average recall**; among the operating points reaching
it, the fastest on the scored metric wins.

|                | Task 1                                        | Task 2                                             |
|----------------|-----------------------------------------------|----------------------------------------------------|
| Dataset family | Wikipedia BGE-M3 (FP16, 1024-dim, normalized) | Llama-Dev (FP32, 128-dim)                          |
| Problem        | k-NN **graph** self-join, k = 15              | k-NN **search**, k = 30                            |
| Distance       | inner product                                 | inner product (via L2 lift)                        |
| Scored metric  | build + search wall-clock (`buildtime`)       | query time (`querytime`)                           |
| Build threads  | all 8                                         | 1 (configured build thread)                        |

### Datasets

| Task | Variant      | File                                      | Vectors                                  |
|------|--------------|-------------------------------------------|------------------------------------------|
| 1    | spot-check   | `benchmark-dev-gooaq-small.h5`            | 10,000 (384-dim — off-family smoke test) |
| 1    | small (dev)  | `benchmark-dev-wikipedia-bge-m3-small.h5` | 200,000                                  |
| 1    | large (eval) | `benchmark-dev-wikipedia-bge-m3.h5`       | 6,350,000                                |
| 2    | spot-check   | `benchmark-dev-llama-small.h5`            | 14,000                                   |
| 2    | dev/eval     | `llama-dev.h5`                            | 256,921                                  |

## Repository layout

| Path                                                    | Contents                                                                                                                                                |
|---------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`cpp/`](cpp/)                                          | deglib (DEG) C++ library and the SISAP binary under `cpp/sisap/` (`task1.cpp`, `task2.cpp`, `sisap.cpp`, per-mode headers in `task1/`, `task2/`).       |
| [`submission/`](submission/)                            | TIRA entrypoint `search.py` and evaluation tools (see [submission/README.md](submission/README.md)). |
| [`Dockerfile`](Dockerfile)                              | Two-stage image: build the C++ binary (AVX2), then a thin Python runtime running `search.py`. |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml)  | Builds the image and runs all three spot-checks through the exact TIRA command schema, then evaluates + plots.                                          |
| `python/`                                               | Legacy reference implementation (not used by the submission).                                                                                           |

## Submission via TIRA

Submissions are handled through TIRA ([tira.io/task-overview/sisap-2026](https://www.tira.io/task-overview/sisap-2026)), which provides a reproducible, containerized evaluation framework. Code submissions for SISAP 2026 are handled only through TIRA.

### Step 1 — Register your team

1. Sign up or log in at [tira.io](https://www.tira.io) (GitHub login supported).
2. Navigate to [tira.io/task-overview/sisap-2026](https://www.tira.io/task-overview/sisap-2026) and click **Register**.
3. Optionally add team members via [tira.io/g?type=my](https://www.tira.io/g?type=my).

### Step 2 — Verify locally

To test the containerized submission pipeline locally on your machine, sync the workspace dependencies:

```bash
# Install/update workspace dependencies
uv sync

# Run a dry run against the task-1 spot-check datasets:
uv run tira-cli code-submission \
    --path . \
    --command 'python3 /app/search.py --input $inputDataset/*.h5 --task-description $inputDataset/config.json --output $outputDir' \
    --task sisap-2026 \
    --dataset task-1-spot-check-20260602-training \
    --dry-run

# Run a dry run against the task-2 spot-check datasets:
uv run tira-cli code-submission \
    --path . \
    --command 'python3 /app/search.py --input $inputDataset/*.h5 --task-description $inputDataset/config.json --output $outputDir' \
    --task sisap-2026 \
    --dataset task-2-spot-check-20260602-training \
    --dry-run
```

### Step 3 — Authenticate and submit

Retrieve your authentication token from the TIRA task page (**Submit** → **Code Submissions** → **New Submission** → **I want to submit from my local machine**), then:

```bash
uv run tira-cli login --token AUTH-TOKEN
uv run tira-cli verify-installation --task sisap-2026 --team deglib

uv run tira-cli code-submission \
    --path . \
    --command 'python3 /app/search.py --input $inputDataset/*.h5 --task-description $inputDataset/config.json --output $outputDir' \
    --task sisap-2026 \
    --dataset task-1-spot-check-20260602-training
```

### Step 4 — Trigger evaluation in the TIRA UI

1. Navigate to the task page, click **Submit** → **Code Submissions**.
2. Select your submission, choose a dataset and hardware configuration.
3. The organizers will handle execution on all datasets once your submission looks correct.

---



## Build & run locally

```bash
# Build the submission image
docker build -t sisap-deglib .

# Run one task the way TIRA does (your-dataset-dir holds the .h5 and config-dir the config.json)
mkdir -p results
docker run --rm --cpus=8 --memory=24g \
    -v "$PWD/your-dataset-dir:/app/dataset:ro" \
    -v "$PWD/results:/app/results:rw" \
    sisap-deglib \
    python3 /app/search.py --input '/app/dataset/*.h5' \
        --task-description /app/data/config-dir/config.json --output /app/results

# Score the results against the dataset ground truth
uv --directory submission run eval.py --results ../results ../res.csv
```

### Building just the C++ binary

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DFORCE_AVX2=ON
cmake --build cpp/build --target deglib_sisap -j"$(nproc)"

# Usage: deglib_sisap <task1|task2> <input.h5> <mode> [options]
# Save mode writes one .bin per operating point into the --output directory:
cpp/build/bin/deglib_sisap task2 dataset.h5 mode5 \
    --no-recall --output results_dir \
    --k-top 30 --max-dist 5000,7000 --eps-search 0.18,0.2 --flas
```

`--march`/AVX note: the build is pinned to **AVX2 (no AVX-512)** because the eval
node is an AMD EPYC 7F72 (Zen 2) with 8 vCPU / 24 GB RAM and no AVX-512.

## Continuous integration

On every push the CI builds the image and runs all three spot-checks through the
same command schema TIRA uses, under the eval node's resource limits
(`--cpus=8 --memory=24g`), then runs `eval.py` / `plot.py` /
`show_operating_points.py`. There is no hard recall gate — it builds, runs and
reports, which is what the challenge requires for a valid public submission.
