# Task 2 Benchmark Summary — Llama Dev dataset

This table lists the best sweep configuration for each test that minimizes search time while reaching at least 80% recall.

**Host:** AMD Ryzen 5 5600G with Radeon Graphics (6C/12T) &nbsp;·&nbsp; RAM: 31.3 GiB total
**Container limits:** 8 CPU threads · 24.0 GiB RAM

| Mode | Method | Best Settings | Load Time | Build Time | FLAS Time | Total Time | Search Time | Recall |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| mode5 | Mode5 baseline | eps=0.18, max_dist=8000 | 3.2s | 31.5s | — | 35.1s | 70.10 ms | 80.12% |
| mode5 | Mode5 (noflas, k_ext=64, k_graph=32, eps_ext=0.01, max_dist=5000,6000,7000,8000, eps_search=0.18) | eps=0.18, max_dist=8000 | 2.9s | 33.2s | — | 36.6s | 68.80 ms | 81.41% |
| mode5 | Mode5 (noflas, k_ext=80, k_graph=40, eps_ext=0.01, max_dist=5000,6000,7000,8000, eps_search=0.16,0.18,0.20) | eps=0.16, max_dist=7000 | 2.9s | 49.2s | — | 52.5s | 69.70 ms | 80.77% |
| mode5 | Mode5 (noflas, k_ext=100, k_graph=50, eps_ext=0.01, max_dist=5000,6000,7000,8000, eps_search=0.16,0.18,0.20) | eps=0.16, max_dist=7000 | 2.8s | 73.7s | — | 77.0s | 66.40 ms | 80.76% |
| mode5 | Mode5 (noflas, k_ext=48, k_graph=24, eps_ext=0.01, max_dist=5000,6000,7000,8000, eps_search=0.16,0.18,0.20) | eps=0.2, max_dist=8000 | 2.9s | 21.8s | — | 25.1s | 77.00 ms | 78.94% |
| mode5 | Mode5 (noflas, k_ext=96, k_graph=32, eps_ext=0.01, max_dist=5000,6000,7000,8000, eps_search=0.16,0.18,0.20) | eps=0.2, max_dist=7000 | 2.9s | 47.4s | — | 50.8s | 75.50 ms | 81.38% |
| mode7 | Mode7 baseline | eps=0.007, max_dist=7000 | 2.8s | 35.6s | — | 38.9s | 82.90 ms | 79.89% |