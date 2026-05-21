Der Text beschreibt den minimalen Arbeitsplan für eine Sisap Submission + Informationen für ein Paper.

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
