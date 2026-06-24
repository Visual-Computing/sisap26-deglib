### Übersicht Zeiten / Recall (Task 2)

| Mode | Method | Best Settings | Load Time | Build Time | FLAS Time | Total Time | Search Time | Recall |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| mode3 | Mode 3: FP32 IP Build & FP16 IP Search (no FLAS) | eps=0.25, max_dist=25000 | 0.06s | 19.50s | — | 19.70s | 100.30 ms | 80.10% |
| mode3 | Mode 3: FP32 IP Build & FP16 IP Search (+ L2 FLAS) | eps=0.28, max_dist=18000 | 0.06s | 22.20s | 38.80s | 22.50s | 84.20 ms | 81.10% |
| mode3 | Mode 3: FP32 IP Build & FP16 IP Search (+ IP FLAS) | eps=0.28, max_dist=15000 | 0.06s | 21.70s | 41.30s | 21.90s | 75.40 ms | 80.26% |
| mode10 | Mode 10: FP32 IP Build (d+1) & FP16 IP Search (no FLAS) | eps=0.18, max_dist=8000 | 0.06s | 24.80s | — | 25.10s | 35.50 ms | 81.50% |
| mode10 | Mode 10: FP32 IP Build (d+1) & FP16 IP Search (+ L2 FLAS) | eps=0.18, max_dist=6000 | 0.06s | 21.20s | 38.70s | 21.50s | 28.90 ms | 80.08% |
| mode10 | Mode 10: FP32 IP Build (d+1) & FP16 IP Search (+ IP FLAS) | eps=0.18, max_dist=7000 | 0.06s | 21.00s | 38.20s | 21.30s | 32.60 ms | 81.13% |
