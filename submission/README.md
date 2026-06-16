# Submission Harness

This directory contains the Python runner and evaluation tools for the SISAP 2026 Indexing Challenge submission.

## Internal Execution Details

When running under TIRA, `search.py` reads the task config, decompresses the input on the fly when needed (the C++ HDF5 reader only handles contiguous datasets, so gzip/chunked inputs are materialized to an uncompressed temp file via `h5py`), drives the binary once per profile, and writes one result file per operating point.

## Output Format

One HDF5 file is generated per operating point under `$outputDir`. Each file contains:

- **Datasets:**
  - `knns` (1-based neighbor IDs; if a query returns fewer than $k$ candidates, padding slots are the vertex's own ID for Task 1 and `0` for Task 2. This padding is harmless, as the evaluator scores by set membership).
  - `dists` (float).
  - Both datasets have the same shape: **`n × (k+1)` for Task 1**, **`n × k` for Task 2**.
- **Root Attributes:** `algo`, `dataset`, `task`, `buildtime`, `querytime`, `params`.

Task 1 prepends the self-reference in column 0 (the extra `+1` column), matching the ground-truth layout the evaluator uses. Task 2 has no self column. Only `knns` is scored: `recall = mean_i |knns[i,:k] ∩ gt[i,:k]| / k`.
