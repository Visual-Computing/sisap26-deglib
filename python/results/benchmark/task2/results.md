# Task 2 Benchmark Summary — Llama Dev dataset

This table lists the best sweep configuration for each test that minimizes search time while reaching at least 80% recall.

**Host:** INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz (4C/8T) &nbsp;·&nbsp; RAM: 29.4 GiB total
**Container limits:** 8 CPU threads · 24.0 GiB RAM

| Mode | Method | Best Settings | Load Time | Build Time | FLAS Time | Total Time | Search Time | Recall |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mode1 | Mode 1: FP32 Build & FP32 Explore (no FLAS) | eps=0.25, max_dist=25000 | 0.1s | 20.1s | — | 20.4s | 168.20 ms | 80.21% |
| mode3 | Mode 3: FP32 Build & FP16 Explore (no FLAS) | eps=0.25, max_dist=25000 | 0.1s | 19.5s | — | 19.7s | 100.30 ms | 80.10% |
| mode3 | Mode 3: FP32 Build & FP16 Explore (+ FLAS) | eps=0.28, max_dist=18000 | 0.1s | 22.2s | 38.8s | 22.5s | 84.20 ms | 81.10% |
| mode3 | Mode 3: FP32 Build & FP16 Explore (+ IP FLAS) | eps=0.28, max_dist=15000 | 0.1s | 21.7s | 41.3s | 21.9s | 75.40 ms | 80.26% |
| mode5 | Mode 5: L2 Build (d+1) & FP16 IP Explore (no FLAS) | eps=0.18, max_dist=8000 | 0.1s | 22.0s | — | 22.3s | 35.20 ms | 80.12% |
| mode5 | Mode 5: L2 Build (d+1) & FP16 IP Explore (+ FLAS) | eps=0.18, max_dist=7000 | 0.1s | 19.0s | 38.4s | 19.3s | 32.10 ms | 80.33% |
| mode4 | Mode 4: L2 Build (d+1) & FP32 L2 Explore (+ FLAS) | eps=0.008, max_dist=6500 | 0.1s | 18.9s | 38.7s | 19.2s | 57.80 ms | 80.88% |
| mode6 | Mode 6: L2 Build (d+1) & FP16 L2 Explore (+ FLAS) | eps=0.007, max_dist=8000 | 0.1s | 18.5s | 38.5s | 18.8s | 32.20 ms | 80.32% |
| mode7 | Mode 7: L2 Build (d+2) & FP16 L2 Explore (+ FLAS) | eps=0.007, max_dist=6500 | 0.1s | 19.3s | 39.2s | 19.6s | 33.70 ms | 80.15% |
| mode8 | Mode 8: EVP Linear Search | eps=0.0, max_dist=1 | 0.1s | 0.0s | — | 0.6s | 422.00 ms | 0.80% |
| mode9 | Mode 9: EVP Asymmetric Linear Search | eps=0.0, max_dist=1 | 0.1s | 0.0s | — | 1.0s | 856.30 ms | 0.87% |
| mode10 | Mode 10: IP Build (d+1) & FP16 IP Explore (no FLAS) | eps=0.18, max_dist=8000 | 0.1s | 24.8s | — | 25.1s | 35.50 ms | 81.50% |
| mode10 | Mode 10: IP Build (d+1) & FP16 IP Explore (FLAS) | eps=0.18, max_dist=6000 | 0.1s | 21.2s | 38.7s | 21.5s | 28.90 ms | 80.08% |
| mode10 | Mode 10: IP Build (d+1) & FP16 IP Explore (IP FLAS) | eps=0.18, max_dist=7000 | 0.1s | 21.0s | 38.2s | 21.3s | 32.60 ms | 81.13% |