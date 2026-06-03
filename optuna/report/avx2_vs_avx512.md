# AVX2 vs AVX-512 — Re-timing the 15 Submission Candidates

## Why
The SISAP evaluation/target server is **AVX2-only** (no AVX-512); our tuning ran on an
AVX-512 VM. Recall is SIMD-independent, so candidate *selection* is unaffected — but the
**timings** that pick the "fastest config with Recall@15 ≥ 0.8" had to be re-measured on
AVX2. We built `sisap26-deglib:avx2` (`docker build --build-arg FORCE_AVX2=ON`; verified
the binary reports `SIMD: AVX2, SSE`) and re-ran all 15 candidates on the 6.35 M dataset.

## Headline: on target-size data, AVX2 ≈ AVX-512
| mode | AVX2 / AVX-512 time ratio |
|---|--:|
| mode4 | **1.05 – 1.07×** (5–7% slower) |
| mode7 | **0.98 – 1.03×** (flat; some *faster* on AVX2) |

Far smaller than feared. For contrast, on the **small** 200 K set AVX2 was **1.46×** slower.
The difference is **memory-boundedness**: at 6.35 M (13+ GB) the work is dominated by RAM
bandwidth and graph-traversal latency, so SIMD width barely matters; at 200 K (largely
cache-resident) it is compute-bound, where AVX-512's wider vectors help.

## What changed, what didn't
- **Winner unchanged:** `mode4, k_graph=26, max_dist=600, evpK=50, nz=608` → recall 0.802 @
  **328 s (AVX2)** (was 313 s on AVX-512). No candidate beats it at ≥ 0.8.
- **Fast mode4 ladder keeps its order** — the eight `k_graph=26` points scale together.
- **Recall identical** within ±0.0006 (floating-point summation-order noise only).
- **Mid-field reordered:** because mode4 slows ~5% while mode7 does not, mode7 climbs.
  e.g. `mode4 kg32/md900` (386 → 411 s) and `mode7 kg28/md400` (401 → 392 s) **swap order**;
  the mode4↔mode7 gap narrows. This does not affect the winner or the fast end.

## Why mode7 doesn't slow down
mode7's explore (FP16-query vs EVP-db) is more FP-heavy and on AVX-512 likely pays the
**AVX-512 license-based down-clocking** penalty without enough SIMD benefit (memory-bound),
so AVX2 — at a higher sustained clock — is equal or faster. mode4's EVP-vs-EVP explore is
popcount-heavy and benefits slightly more from AVX-512.

## All 15 candidates (sorted by AVX2 time)
| mode | kg | max_dist | evpK | nz | recall | AVX-512 (s) | AVX2 (s) | ratio |
|--|--:|--:|--:|--:|--:|--:|--:|--:|
| mode4 | 26 | 500 | 50 | 608 | 0.7918 | 302.6 | 316.7 | 1.047 |
| **mode4** | **26** | **600** | **50** | **608** | **0.8021** | **312.9** | **328.3** | **1.049** |
| mode4 | 26 | 700 | 50 | 608 | 0.8107 | 323.0 | 339.7 | 1.052 |
| mode4 | 26 | 800 | 50 | 608 | 0.8178 | 332.1 | 350.1 | 1.054 |
| mode4 | 26 | 900 | 50 | 608 | 0.8237 | 340.3 | 359.7 | 1.057 |
| mode4 | 26 | 1000 | 50 | 608 | 0.8284 | 348.3 | 368.3 | 1.057 |
| mode4 | 26 | 1200 | 50 | 608 | 0.8351 | 360.9 | 382.6 | 1.060 |
| mode7 | 28 | 400 | 50 | 576 | 0.8067 | 400.8 | 392.3 | 0.979 |
| mode4 | 26 | 1400 | 50 | 608 | 0.8391 | 369.1 | 393.7 | 1.067 |
| mode4 | 32 | 900 | 50 | 512 | 0.8422 | 386.1 | 410.9 | 1.064 |
| mode7 | 32 | 400 | 50 | 512 | 0.8143 | 403.4 | 413.7 | 1.026 |
| mode4 | 32 | 800 | 100 | 512 | 0.8474 | 403.9 | 432.7 | 1.071 |
| mode7 | 32 | 500 | 50 | 512 | 0.8267 | 427.7 | 439.0 | 1.026 |
| mode7 | 32 | 600 | 50 | 512 | 0.8367 | 454.4 | 464.8 | 1.023 |
| mode7 | 28 | 800 | 75 | 576 | 0.8487 | 496.8 | 513.4 | 1.033 |

## Conclusion
- The AVX-512 tuning is a **valid proxy** for the AVX2 target hardware: parameter choices
  and recall transfer 1:1, and the winner is robust.
- Final submission should use the **AVX2 timings**; the fastest config ≥ 0.8 is
  `mode4, k_graph=26, max_dist=600, evpK=50` at ~328 s.
- Data: `optuna/candidates_15_avx2.csv`. Reproduce: `optuna/avx2_retime.py`.
