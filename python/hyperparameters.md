# Hyperparameters

- [Prune Worst Sweep](#prune-worst-sweep)
- [Hyperparameter Optimization (Large Dataset)](#hyperparameter-optimization-large-dataset)

---

## `--prune-worst` Sweep

Varies `--prune-worst <n>`: replaces the $N$ worst neighbors per vertex (largest weight / lowest similarity) with a self-loop pointing to the vertex itself (weight `0.0f`). Mode 1 (`fp16`), `MaxDist=100`,  `--non-zeros 512` on the small `wikipedia-bge-m3-small` dataset.

| `prune_worst` | Recall@15 | Explore Time (ms) | Notes |
|:---:|:---:|:---:|:---|
| **0** | **0.8143** | **1321.55** | Baseline (unmodified graph) |
| 1 | 0.8169 | 1339.29 |  |
| 2 | 0.8194 | 1345.70 |  |
| 3 | 0.8220 | 1373.73 |  |
| 4 | 0.8245 | 1388.31 | Good speed/recall balance |
| 5 | 0.8272 | 1423.15 |  |
| 6 | 0.8297 | 1449.03 |  |
| 7 | 0.8323 | 1642.83 |  |
| 8 | 0.8348 | 1432.56 |  |
| 9 | 0.8372 | 1452.57 |  |
| 10 | 0.8397 | 1480.57 |  |
| 11 | 0.8422 | 1503.74 |  |
| 12 | 0.8446 | 1585.43 |  |
| 13 | 0.8469 | 1522.54 |  |
| 14 | 0.8491 | 1541.33 |  |
| 15 | 0.8513 | 1532.97 | >85% recall |
| 16 | 0.8533 | 1650.89 |  |
| 17 | 0.8552 | 1558.66 |  |
| **18** | **0.8561** | **1591.27** | 🏆 **Optimal (+4.18% absolute gain)** |
| 19 | 0.8550 | 1610.55 | Overpruning threshold |
| 20 | 0.8505 | 1842.01 | Connectivity loss starts |
| 21 | 0.8415 | 1657.08 |  |
| 22 | 0.8273 | 1686.73 |  |
| 23 | 0.8079 | 1671.14 | Below baseline |
| 24 | 0.7826 | 1756.54 |  |
| 25 | 0.7505 | 1757.28 |  |
| 26 | 0.7092 | 1728.92 |  |
| 27 | 0.6513 | 1627.95 | Severe graph partitioning |
| 28 | 0.5679 | 1445.85 |  |
| 29 | 0.4446 | 1042.72 | Too few routing edges |
| 30 | 0.2594 | 503.47 | Near disjoint graph |

---

## Hyperparameter Optimization (Large Dataset)

Results for **Wikipedia BGE-M3 Large** with varios hyperparameters settings but `--prune-worst 0` and `--non-zeros 512`, both can increase the recall further.

| Mode | Method | Settings | Load | Quant | Build | Convert | Explore | Rerank | Overall | Recall |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | `evp-asymmetric-rerank` | `M=32`, `MaxDist=200`, `evpK=50` | 15 s | 22 s | 265 s | 2 s | 66 s | 20 s | 390 s | 0.7382 |
| 7 | `evp-asymmetric-rerank` | `M=32`, `MaxDist=300`, `evpK=50` | 15 s | 22 s | 265 s | 2 s | 94 s | 20 s | 418 s | 0.7675 |
| 7 | `evp-asymmetric-rerank` | `M=32`, `MaxDist=400`, `evpK=50` | 15 s | 22 s | 265 s | 2 s | 120 s | 20 s | 444 s | 0.7863 |
| 7 | `evp-asymmetric-rerank` | `M=32`, `MaxDist=400`, `evpK=100` | 15 s | 22 s | 265 s | 2 s | 135 s | 40 s | 479 s | 0.7874 |
| 4 | `evp-rerank` | `M=32`, `MaxDist=400`, `evpK=50` | 15 s | 22 s | 265 s | 0 s | 90 s | 20 s | 412 s | 0.7790 |
| 4 | `evp-rerank` | `M=32`, `MaxDist=500`, `evpK=50` | 15 s | 22 s | 265 s | 0 s | 110 s | 20 s | 432 s | 0.7914 |
| 3 | `evp` | `M=32`, `MaxDist=200`, `non_zeros=600` | 15 s | 22 s | 264 s | 0 s | 41 s | 0 s | 347 s | 0.6327 |

*Note: Build time includes data loading, conversion, and graph construction. Explore time for EVP includes all-pair similarity calculation; for deglib it measures retrieving K=15 neighbors for all elements.*
