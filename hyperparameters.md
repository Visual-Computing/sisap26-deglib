# The hyperparameter, `--prune-worst <n>`

Allowing the benchmark to replace the $N$ worst neighbors per vertex (having the largest weights / lowest similarity) with a self-loop reference pointing to the vertex itself with weight `0.0f`. The following table shows the results for Modi1 `MaxDist=100` on the `wikipedia-bge-m3-small` datase.



| `prune_worst` | Recall@15 | Explore Time (ms) | Impact / Notes |
|:---:|:---:|:---:|:---|
| **0** | **0.8143** | **1321.55** | **Baseline (unmodified graph)** |
| 1 | 0.8169 | 1339.29 | Small improvement |
| 2 | 0.8194 | 1345.70 | Steady recall gain |
| 3 | 0.8220 | 1373.73 | Regularized search graph |
| 4 | 0.8245 | 1388.31 | Good speed/recall balance |
| 5 | 0.8272 | 1423.15 | |
| 6 | 0.8297 | 1449.03 | |
| 7 | 0.8323 | 1642.83 | |
| 8 | 0.8348 | 1432.56 | |
| 9 | 0.8372 | 1452.57 | |
| 10 | 0.8397 | 1480.57 | |
| 11 | 0.8422 | 1503.74 | |
| 12 | 0.8446 | 1585.43 | |
| 13 | 0.8469 | 1522.54 | |
| 14 | 0.8491 | 1541.33 | |
| 15 | 0.8513 | 1532.97 | **>85% Recall** achieved |
| 16 | 0.8533 | 1650.89 | |
| 17 | 0.8552 | 1558.66 | |
| **18** | **0.8561** | **1591.27** | 🏆 **Optimal Recall (+4.18% absolute gain!)** |
| 19 | 0.8550 | 1610.55 | Overpruning threshold boundary |
| 20 | 0.8505 | 1842.01 | Connectivity loss starts |
| 21 | 0.8415 | 1657.08 | |
| 22 | 0.8273 | 1686.73 | |
| 23 | 0.8079 | 1671.14 | Below baseline recall |
| 24 | 0.7826 | 1756.54 | |
| 25 | 0.7505 | 1757.28 | |
| 26 | 0.7092 | 1728.92 | |
| 27 | 0.6513 | 1627.95 | Severe graph partitioning |
| 28 | 0.5679 | 1445.85 | |
| 29 | 0.4446 | 1042.72 | Too few routing edges |
| 30 | 0.2594 | 503.47 | Near disjoint graph |
