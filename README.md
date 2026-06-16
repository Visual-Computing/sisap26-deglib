# SISAP 2026 — deglib

Submission for the [SISAP 2026 Indexing Challenge](https://sisap-challenges.github.io/2026/).
The index is a [**Dynamic Exploration Graph (DEG)**](https://github.com/Visual-Computing/DynamicExplorationGraph/tree/evp)
combined with [**EVP (Equi-Voronoi Polytope) quantization**](https://github.com/MetricSearch/metric_space_rust),
implemented in C++ (`cpp/`) and driven by the official baseline Python harness
(`submission/`). Everything ships as a **single Docker container** that TIRA builds
from this repo.

## Approach

- **Index:** deglib's even-regular exploration graph, built once per run, then a
  parameter sweep produces several operating points (build/recall trade-offs) from
  that single build.
- **Task 1** (k-NN self-join, scored on **build + search time**): graph mode `mode4` —
  EVP build + EVP explore + exact FP16 inner-product rerank. Here **search = explore +
  rerank**. Task 1 ranks on the *total* (build + search), which `search.py` packs into
  the `buildtime` attribute with `querytime` = 0 — so `buildtime` is the sum, not a
  claim that search is part of building. Search is a real component, often ≈ half the
  total.
- **Task 2** (MIPS search, scored on **query time**): graph mode `mode5` —
  L2-converted FP32 build (with FLAS pre-sort) + FP16 inner-product search.
- **Task 3** (sparse SPLADE) is out of scope and skipped cleanly (exit 0) so the
  mandatory spot-check CI stays green.

The C++ binary computes neighbors **and** distances during search; the thin Python
entrypoint adapts the output to the official result format. Per-dataset parameters
live in `TASK1_PROFILES` / `TASK2_PROFILES` in [`submission/search.py`](submission/search.py)
— unknown datasets fail fast rather than silently using bad parameters.

## Challenge tasks & constraints

Both tasks run under the same hard limits: **8 vCPUs, 24 GB RAM, ≤ 8 h, read-only
dataset, no internet** in the container (the eval node is an AMD EPYC 7F72, no
AVX-512). The goal is **≥ 0.8 average recall**; among the operating points that reach
it, the fastest on the scored metric wins.

|                | Task 1                                        | Task 2                                             |
|----------------|-----------------------------------------------|----------------------------------------------------|
| Dataset family | Wikipedia BGE-M3 (FP16, 1024-dim, normalized) | Llama-Dev (FP32, 128-dim)                          |
| Problem        | k-NN **graph** self-join, k = 15              | k-NN **search**, k = 30                            |
| Distance       | inner product                                 | inner product (via L2 lift)                        |
| Scored metric  | build + search wall-clock (`buildtime`)       | query time (`querytime`)                           |
| Build threads  | all 8                                         | **1** — graph built single-threaded, per the rules |

### Datasets

| Task | Variant      | File                                      | Vectors                                  |
|------|--------------|-------------------------------------------|------------------------------------------|
| 1    | spot-check   | `benchmark-dev-gooaq-small.h5`            | 10,000 (384-dim — off-family smoke test) |
| 1    | small (dev)  | `benchmark-dev-wikipedia-bge-m3-small.h5` | 200,000                                  |
| 1    | large (eval) | `benchmark-dev-wikipedia-bge-m3.h5`       | 6,350,000                                |
| 2    | spot-check   | `benchmark-dev-llama-small.h5`            | 14,000                                   |
| 2    | dev/eval     | `llama-dev.h5`                            | 256,921                                  |

## Graph modes

The `deglib_sisap` binary implements seven graph modes per task (`mode1`…`mode7`).
The profiles in `search.py` currently use **`mode4` (Task 1)** and **`mode5` (Task 2)**;
⭐ marks the strongest submission candidates (the other ⭐, `mode7`, is a close
alternative that is implemented but not wired into a profile). All modes share the
same save-mode contract (one result file per operating point holding neighbor ids
**and** distances), so they are drop-in benchmark alternatives.

**Task 1** — EVP variants

| Mode        | Name                           | Description                                  |
|-------------|--------------------------------|----------------------------------------------|
| mode1       | fp16                           | FP16 build + FP16 explore                    |
| mode2       | evp-linear                     | EVP quantization + brute-force linear search |
| mode3       | evp                            | EVP build + EVP explore (no rerank)          |
| **mode4** ⭐ | evp-rerank                     | EVP build + EVP explore + FP16 rerank        |
| mode5       | evp-build-fp16-external-search | EVP build + FP16 external graph search       |
| mode6       | evp-asymmetric                 | EVP build + asymmetric FP16-vs-EVP search    |
| mode7 ⭐     | evp-asymmetric-rerank          | EVP build + asymmetric search + FP16 rerank  |

**Task 2** — L2-lift variants

| Mode        | Name                    | Description                             |
|-------------|-------------------------|-----------------------------------------|
| mode1       | baseline                | FP32 build + FP32 inner-product explore |
| mode2       | fp16-build-fp16-explore | FP16 build + FP16 IP explore            |
| mode3       | baseline-fp16           | FP32 build + FP16 IP explore            |
| mode4       | l2-converted            | FP32 L2(d+1) build + FP32 L2 explore    |
| **mode5** ⭐ | l2-fp16-ip              | FP32 L2(d+1) build + FP16 IP explore    |
| mode6       | l2-fp16-l2              | FP32 L2(d+1) build + FP16 L2 explore    |
| mode7 ⭐     | l2-fp16-d2              | FP32 L2(d+2) build + FP16 L2 explore    |

## Repository layout

| Path                                                    | Contents                                                                                                                                                |
|---------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`cpp/`](cpp/)                                          | deglib (DEG) C++ library and the SISAP binary under `cpp/sisap/` (`task1.cpp`, `task2.cpp`, `sisap.cpp`, per-mode headers in `task1/`, `task2/`).       |
| [`submission/`](submission/)                            | TIRA entrypoint `search.py` plus the vendored baseline harness (`eval.py`, `datasets.py`, `plot.py`, `show_operating_points.py`, `data/*/config.json`). |
| [`Dockerfile`](Dockerfile)                              | Two-stage image: build the binary (AVX2), then a thin Python runtime that runs `search.py`.                                                             |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml)  | Builds the image and runs all three spot-checks through the exact TIRA command schema, then evaluates + plots.                                          |
| `python/`                                               | Legacy reference implementation (not used by the submission).                                                                                           |

## How it runs on TIRA

TIRA builds the image from the repo, mounts the dataset (no internet inside the
container), and invokes:

```bash
python3 /app/search.py \
    --input $inputDataset/*.h5 \
    --task-description $inputDataset/config.json \
    --output $outputDir
```

`search.py` reads the task config, decompresses the input on the fly when needed
(the C++ HDF5 reader only handles contiguous datasets, so gzip/chunked inputs are
materialized to an uncompressed temp file via `h5py`), drives the binary once per
profile, and writes one result file per operating point.

## Output format

One HDF5 file per operating point under `$outputDir`, each with:

- datasets `knns` (1-based neighbor ids; if a query returns fewer than k candidates
  the padding slots are the node's own id for Task 1 and `0` for Task 2 — both
  harmless, since the evaluator scores by set membership) and `dists` (float), both
  the same shape — **`n × (k+1)` for Task 1**, **`n × k` for Task 2**;
- root attributes `algo`, `dataset`, `task`, `buildtime`, `querytime`, `params`.

Task 1 prepends the self-reference in column 0 (the extra `+1` column), matching the
ground-truth layout the evaluator uses; Task 2 has no self column. Only `knns` is
scored — `recall = mean_i |knns[i,:k] ∩ gt[i,:k]| / k`.

## Build & run locally

```bash
# Build the submission image
docker build -t sisap-deglib .

# Run one task the way TIRA does (dataset dir holds the .h5 + config.json)
mkdir -p results
docker run --rm --cpus=8 --memory=24g \
    -v "$PWD/your-dataset-dir:/app/data/ds:ro" \
    -v "$PWD/results:/app/results:rw" \
    sisap-deglib \
    python3 /app/search.py --input '/app/data/ds/*.h5' \
        --task-description /app/data/ds/config.json --output /app/results

# Score the results against the dataset ground truth (run from submission/,
# like CI does, so eval.py can import the harness modules)
cd submission && PYTHONPATH=. python3 eval.py --results ../results res.csv
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
