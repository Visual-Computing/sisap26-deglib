# Task 2 Benchmark Summary — Llama Dev dataset

This table lists the best sweep configuration for each test that minimizes search time while reaching at least 80% recall.

**Host:** INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz (4C/8T) &nbsp;·&nbsp; RAM: 29.4 GiB total
**Container limits:** 8 CPU threads · 24.0 GiB RAM

| Mode | Method | Best Settings | Load Time | Build Time | FLAS Time | Total Time | Search Time | Recall |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mode3 | Mode 3: FP32 Build & FP16 Explore (no FLAS) | eps=0.25, max_dist=25000 | 0.1s | 19.5s | — | 19.7s | 100.00 ms | 80.10% |
| mode3 | Mode 3: FP32 Build & FP16 Explore (+ FLAS) | eps=0.28, max_dist=18000 | 0.1s | 21.4s | 38.6s | 21.7s | 84.70 ms | 81.10% |
| mode5 | Mode 5: L2 Build (d+1) & FP16 IP Explore (no FLAS) | eps=0.18, max_dist=8000 | 0.1s | 22.4s | — | 22.7s | 35.40 ms | 80.12% |
| mode5 | Mode 5: L2 Build (d+1) & FP16 IP Explore (+ FLAS) | eps=0.18, max_dist=7000 | 0.1s | 18.8s | 38.6s | 19.2s | 32.20 ms | 80.33% |
| mode4 | Mode 4: L2 Build (d+1) & FP32 L2 Explore (+ FLAS) | eps=0.008, max_dist=6500 | 0.1s | 18.3s | 38.8s | 18.6s | 57.70 ms | 80.88% |
| mode6 | Mode 6: L2 Build (d+1) & FP16 L2 Explore (+ FLAS) | eps=0.007, max_dist=8000 | 0.1s | 18.8s | 38.3s | 19.1s | 32.10 ms | 80.32% |
| mode7 | Mode 7: L2 Build (d+2) & FP16 L2 Explore (+ FLAS) | eps=0.007, max_dist=6500 | 0.1s | 19.2s | 39.0s | 19.6s | 33.60 ms | 80.15% |
| mode8 | Mode 8: EVP Linear Search | eps=0.0, max_dist=1 | 0.1s | 0.0s | — | 0.6s | 420.20 ms | 0.80% |
| mode9 | Mode 9: EVP Asymmetric Linear Search | eps=0.0, max_dist=1 | 0.1s | 0.0s | — | 1.0s | 856.90 ms | 0.87% |