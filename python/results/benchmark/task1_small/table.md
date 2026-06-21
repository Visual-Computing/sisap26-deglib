### Übersicht Zeiten / Recall

| Ansatz | Quant | Bau+Convert | Expl + Rerank | Overall | Recall |
|:---|:---:|:---:|:---:|:---:|:---:|
| Lin EVP | 0.55 | 0 | 175.40 | 176.20 | 0.7124 |
| Lin EVP Asymm | 0.54 | 0 | 857.40 | 858.10 | 0.7854 |
| DEG FP16 Baseline | 0 | 8.50 | 0.64 | 9.35 | 0.8487 |
| DEG EVP Baseline | 0.54 | 3.72 | 0.38 | 4.86 | 0.6816 |
| DEG EVP-Asym | 0.55 | 3.93 + 0.03 | 1.00 | 5.73 | 0.7378 |
| DEG EVP + Reranking | 0.56 | 3.91 | 0.64 + 0.39 | 5.72 | 0.8484 |
| DEG EVP -> FP16 repl | 0.55 | 3.78 + 0.06 | 1.44 | 6.02 | 0.8487 |
