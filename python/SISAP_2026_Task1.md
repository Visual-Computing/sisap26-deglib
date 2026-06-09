# SISAP 2026 Challenge – Task 1: K-Nearest Neighbor Graph (Metric Self-Join)

---

## Quelle

- **Website:** https://sisap-challenges.github.io/2026/
- **GitHub (Pre-Registration):** https://github.com/sisap-challenges/challenge2026/
- **HuggingFace Datasets:** https://huggingface.co/datasets/sisap-challenges/SISAP2026

---

## Überblick

Die SISAP 2026 Indexing Challenge lädt Forscher und Praktiker ein, an spannenden Aufgaben teilzunehmen, um den Stand der Kunst in Similarity Search und Indexierung voranzubringen. In diesem Jahr werden drei herausfordernde Aufgaben gestellt.

---

## Task 1: K-Nearest Neighbor Graph (a.k.a. Metric Self-Join)

### Aufgabenstellung

In dieser Aufgabe sollen Teilnehmende speichereffiziente Index-Lösungen entwickeln, die zur Berechnung einer Approximation des **k-nächsten-Nachbarn-Graphen** für **k=15** verwendet werden. Jede Lösung wird in einem Linux-Container mit begrenzten Speicher- und Speicherressourcen ausgeführt.

### Rahmenbedingungen

| Parameter | Wert |
|---|---|
| **Container-Spezifikation** | 8 virtuelle CPUs, 24 GB RAM |
| **Datensatz-Mount** | Read-only in den Container gemountet |
| **Wall-Clock Zeit (gesamt)** | 8 Stunden |
| **Datensatz** | WIKIPEDIA (6.4 Millionen Vektoren, 1024 Dimensionen) |
| **Similarity-Metrik** | Dot Product (Vektoren sind normalisiert) |

### Ziel

Berechnung des **k-nächsten-Nachbarn-Graphen** (ohne Selbst-Referenzen), d. h. Finden der k-nächsten Nachbarn für alle Objekte im Datensatz als Queries.

- **Qualitätsmessung:** Recall gegen einen bereitgestellten Gold-Standard und die gesamte Berechnungszeit (inklusive Preprocessing, Indexing, Search und Postprocessing wie Re-Ranking)
- **Operating Point:** Schnellste Graph-Konstruktionszeit, die einen durchschnittlichen Recall von mindestens **0.8** erreicht
- **Entwicklung:** Es wird ein Entwicklungsdatsensatz bereitgestellt; die Evaluationsphase verwendet einen unoffengelegten Datensatz ähnlicher Größe, der mit demselben neuronalen Modell berechnet wurde

### Wichtige Hinweise zu Task 1

- Der Gold-Standard enthält **Selbst-Referenzen** (jeder Punkt ist sein eigener nächster Nachbar). Diese werden vor der Recall-Berechnung entfernt.
- Für Task 1 wird die **gesamte Wall-Clock-Zeit** gemessen (nicht nur die Suchzeit).
- Teilnehmende können einen einzigen Index aufbauen und **15 verschiedene Suchparameter** testen.

---

## Wichtige Daten (alle 2026)

| Datum | Ereignis |
|---|---|
| **23. Februar** | Challenge öffnet |
| **Ende April** | Evaluation Pipeline verfügbar |
| **10. Juni** | Deadline für Einreichung der Lösungsimplementationen |
| **17. Juni** | Deadline für Kurzpaper-Beschreibungen |
| **8. Juli** | Bekanntgabe des endgültigen Rankings |
| **27. Juli** | Paper-Benachrichtigung |
| **13. August** | Camera Ready der Papers |

---

## Organization Committee

- **Eric S. Téllez**, INFOTEC-SECIHTI, Mexiko
- **Martin Aumüller**, ITU Copenhagen, Dänemark
- **Maik Fröbe**, University of Jena, Deutschland
- **Vladimír Míč**, SDU Odense, Dänemark

**Kontakt:** sisap-2026-indexing-challenge@googlegroups.com

---

## Datasets im Detail

### WIKIPEDIA (Hauptdatensatz)

- **Datei:** `benchmark-dev-wikipedia-bge-m3.h5`
- **Quelle:** wikimedia/wikipedia Repo + BGE-M3 Modell
- **Similarity:** Cosine / Dot Product (normalisierte Vektoren)
- **Train:** 6.35M Vektoren, Matrix 1024 × 6.350.000 (f16)
- **In-Distribution (engl. Wikipedia):**
  - `queries`: 10.000 Vektoren, 1024 × 10.000 (f16)
  - `knns`: Gold-Standard für 1.000 NN, 1.000 × 10.000 (i32)
  - `dists`: Gold-Standard Distanzen (1-dot), 1.000 × 10.000 (f32)
- **Out-of-Distribution (span. Wikipedia, cross-lingual):**
  - `queries`: 10.000 Vektoren, 1024 × 10.000 (f16)
  - `knns` / `dists`: gleiche Struktur wie oben
- **allknn:**
  - `knns`: All-KNN Graph, 32 × 6.350.000 (i32)
  - `dists`: 32 × 6.350.000 (f32)

### WIKIPEDIA Small (für Testing/Development)

- **Datei:** `benchmark-dev-wikipedia-bge-m3-small.h5`
- **Train:** 200K Vektoren

### Zusätzliche Datasets für Task 1

Teilnehmende dürfen auch die kleineren Datasets der SISAP 2025 Challenge verwenden:
- **Verfügbar unter:** https://huggingface.co/datasets/SISAP-Challenges/SISAP2025
- **Beispiel:** CCNEWS

---

## Test Data, Queries, Hyperparameter (allgemein)

- Alle Testdaten sind im HDF5-Dateiformat eingebettet (Struktur dokumentiert auf HuggingFace)
- **H5py/HDF5.jl Hinweis:** Matrizen werden in Platform-Order gelesen — Achtung bei Dimension-Permutation
- Teilnehmende können **einen einzigen Index** aufbauen und **15 verschiedene Suchparameter** testen
- Gold-Standards werden als Matrix von Objekt-Identifikatoren bereitgestellt (1-based indexing)
- **Task 2 & 3:** Alle Queries werden gleichzeitig präsentiert, Batch-Processing wird ausdrücklich gefördert
- **Task 1:** Gesamte Wall-Clock-Zeit wird gemessen; Task 2 & 3 nur die Suchphase

### Python: Sparse Matrizen laden (Task 3 Referenz)

```python
import h5py
from scipy.sparse import csr_matrix

def load_sparse_matrix(h5_group):
    indptr = h5_group['indptr'][:]
    indices = h5_group['indices'][:]
    data = h5_group['data'][:]
    shape = tuple(h5_group.attrs['shape'])
    return csr_matrix((data, indices, indptr), shape=shape)
```

---

## Ergebnis-Einreichungsformat

Ergebnisse müssen als HDF5-Dateien mit folgender Struktur bereitgestellt werden:

### File Content
- **`knns`**: Eine n×k-Matrix von Objekt-Identifikatoren (Ganzzahlen), wobei n die Anzahl der Queries und k die Anzahl der Nachbarn ist. Die i-te Zeile enthält die Identifikatoren der k nächsten Nachbarn der i-ten Query. Identifikatoren verwenden **1-based indexing** (der erste Objekt im Datensatz hat ID 1).
- **`dists`**: Eine n×k-Matrix von Distanzen (Floats). Die i-te Zeile enthält die Distanzen der k nächsten Nachbarn der i-ten Query.

**Hinweis:** Matrizen sollten **row-major order** folgen (Standard für C/Python/NumPy).

### Metadata (Attributes am Root-Level)
- **`algo`**: Name des Algorithmus (String)
- **`task`**: Name der Aufgabe (z. B. `task1`, `task2`, `task3`)
- **`buildtime`**: Index-Konstruktionszeit in Sekunden (Float)
- **`querytime`**: Gesamte Suchzeit in Sekunden (Float)
- **`params`**: String zur Beschreibung der Parameter (z. B. `M=16,efConstruction=100`)

### Directory Structure
```
results/<task_name>/<unique_filename>.h5
```
Beispiel: `results/task1/myalgo_M16_ef100.h5`

---

## Docker Container und Evaluation

Die Lösungen werden in einem Docker-Container mit folgenden Limits evaluiert:

```bash
docker run \
    -it \
    --cpus=8 \
    --memory=24g \
    --memory-swap=24g \
    --memory-swappiness 0 \
    --volume $(pwd)/data:/app/data:ro \
    --volume $(pwd)/results:/app/results:rw \
    sisap-baseline --task task3 --dataset fiqa-dev
```

| Flag | Beschreibung |
|---|---|
| `--cpus=8` | Begrenzt den Container auf 8 CPU-Kerne |
| `--memory=24g` | Begrenzt den RAM auf 24 GB |
| `--memory-swap=24g` | Stellt sicher, dass kein Swap über das RAM-Limit hinaus verwendet wird |
| `--memory-swappiness 0` | Deaktiviert Swap-Nutzung |
| `--volume` | Mountet das Data-Verzeichnis als read-only und das Results-Verzeichnis als read-write |
| `sisap-baseline` | Dies sollte durch den eigenen Image-Namen ersetzt werden |
| `--task` / `--dataset` | Beispiel-Argumente für die spezifische Aufgabe und den Datensatz |

---

## Hardware Specifications

Details der Evaluation Machine werden nachgereicht ("will soon be available").

---

## Registration und Teilnahme

1. **Pre-Registration:** Eröffne ein *"Pre-registration request"* Issue im GitHub Repository https://github.com/sisap-challenges/challenge2026/
   - **Erforderliche Angaben:** Team-Name, GitHub-Username des corresponding authors, Liste der interessierten Aufgaben, Team-Mitglieder mit Affiliationen
2. **Entwicklungsphase:** Zugang zu Gold-Standards für alle Aufgaben
3. **GitHub Repositories:** Teams müssen öffentliche GitHub Repositories mit working GitHub Actions und klaren Anleitungen bereitstellen (bis zu 15 Parametersätze). Einreichungen müssen in Docker-Containern laufen. Ergebnisse in Standardformat.
4. **Evaluation:** Repositories werden geklont und getestet. Ergebnisse werden zur Verifizierung mit den Autoren geteilt, bevor das finale Ranking veröffentlicht wird.
5. **Private Evaluation Workloads** werden nach der Evaluation öffentlich geteilt.
6. **Nur eine Person pro Team**

---

## Paper Submissions

- **Ein Kurzpaper pro Team**, das das System detailliert beschreibt
- Bei mehreren Aufgaben: Beschreibung in **einem einzigen Paper** (kann Technical Report referenzieren)
- Akzeptierte Papers werden Teil der Conference Proceedings + Special Session bei SISAP 2026
- Jedes akzeptierte Paper muss als **oral presentation** auf der Session präsentiert werden
- Einreichungen ohne akzeptiertes Kurzpaper werden **disqualifiziert** und aus dem Ranking entfernt

---

## Final Comments

- Jede Transformation des Datasets ist erlaubt: Packing in andere Datentypen, Dimensional Reduction, Locality-Sensitive Hashing, Product Quantization, Transformation in Binary Sketches
- Reproduzierbarkeit und Open Science sind primäre Ziele — nur öffentliche GitHub Repositories mit working GitHub Actions werden akzeptiert
- Indexing Algorithmen können bereits veröffentlicht oder originale Beiträge sein, aber ein dedizierter Aufwand zur Lösung der jeweiligen Aufgabe muss in der Einreichung sichtbar sein

---

## Beispiele

### Julia Example
- **Repo:** https://github.com/sisap-challenges/sisap2026-julia-example
- **Beschreibung:** Working example mit GitHub Actions
- **CI-Workflow:** Julia 1.10.10, Ubuntu latest, x64
- **Dockerfile:** Basis `julia:1.10.10`, läuft mit `Pkg.instantiate()` (8 Threads), dann `sisap2026.jl` mit `-t8 -Cnative -O3`
- **Key Dependencies:** SimilaritySearch 0.14, InvertedFiles, HDF5, JLD2, DataFrames, CSV, MultivariateStats, SearchModels, SparseMatrixCSC
- **Task 1 Ansatz:** `SearchGraph` mit `OptimizeParameters(MinRecall(0.9))`, `RandomHints`, `SatNeighborhood`. Unterstützt PCA + Scalar Quantization Varianten.
- **Evaluation:** Recall-Berechnung via `macrorecall`, Quantile-Statistiken, CSV-Output

### Python Example
- **Repo:** https://github.com/sisap-challenges/sisap26-python-baseline
- **Status:** Work in progress

---

## SISAP 2025 Ergebnisse (Referenz)

### Task 1 (PUBMED23: 23.9M 384-dim Vektoren, 16 GB RAM, 8 vCPUs, 12 Stunden)

**Ranking:** Höchster Durchsatz bei Recall ≥ 0.7 für k=30

| Team | Recall | Throughput (q/s) |
|---|---|---|
| BL-SearchGraph | 0.7322 | 16.769 |
| BrownCICESE | 0.7884 | 6.928 |
| TeamDoubleFiltering | 0.7212 | 1.081 |

**Honorable Mentions:** BrownCICESE (Cole Foster), Crusty Coders (Alan Dearle)

**Full Paper:** "Overview of the SISAP 2025 Indexing Challenge" in SISAP 2025 Proceedings (Springer)

---

**License:** CC BY-SA 4.0 | sisap challenge committee
**Last modified:** March 27, 2026
