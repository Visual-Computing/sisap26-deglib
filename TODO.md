Der Text beschreibt den minimalen Arbeitsplan für eine Sisap Submission + Informationen für ein Paper.

# Zielserver (AVX2) — wichtige Messungen erneut laufen lassen
Der Evaluations-/Zielserver beherrscht KEIN AVX512, nur AVX2. Die aktuellen Optuna-/Sweep-Messungen
laufen auf einem AVX512-Server. Recall ist identisch (SIMD-Breite ändert die Ergebnisse nicht), aber
Zeiten und damit das Ranking ("schnellstes Verfahren mit Recall >= 0.8") können sich verschieben.
- Dockerfile unterstützt jetzt `--build-arg FORCE_AVX2=ON` (AVX2-Only-Build, Image-Tag z.B. `sisap26-deglib:avx2`).
- Wichtige Konstellationen (Pareto-Front-Kandidaten nahe der 0.8-Grenze) mit erzwungenem AVX2 nachmessen,
  da die Zeiten über die Submission-Auswahl entscheiden.
- Finale kleine Serie idealerweise auf dem echten Zielserver: mode4 + mode7, je 5–8 Konstellationen
  entlang der Pareto-Front.
- AVX512-Messdaten für das Paper aufheben.

# Was ist der beste Quantisierungsansatz? 
Welcher anderen Quantisierungsansätze gibt es noch? Ist ihre Qualität einstellbar? Muss man sie anlernen? Wie schnell lässt sich ihre Distanzberechen?

Mögliche Verfahren:
 - EVP
 - PCA
 - [ScaNN](https://github.com/google-research/google-research/tree/master/scann)
 - Sisap 2024 Feature Compression
 - [TurboQuant](https://github.com/RyanCodrai/turbovec)

# Baseline
- EVP baseline (lineare Suche wandelt aktuell EVP bits in uint8 um für substraction und fp32 für dot produkt)
- deglib baseline (aktuell nicht möglich weil kein FP16 support)
- HNSW baseline (aus der sisap vorlage)

# deglib Anpassungen
Um EVP in deglib zu testen, brauch deglib:
- ein neues Module was die EVP Features herstellt (parallel)
- eine neue Distanzfunktion
- Python Anbindungen an beides

# SISAP Python submission
- muss die FP16 Features in numpy vorhalten
- die Features in EVP bits umwandeln
- mit EVP den Graphen bauen (parallel)
- den Graphen batchweise mit den EVP features durchsuchen (parallel)
- die Top100 Ergebniss anhand ihrer FP16 Features Reranken (parallel)
