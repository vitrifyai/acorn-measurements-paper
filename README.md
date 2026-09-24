# ACORN methods paper: reproducibility package

Data, configurations, and scripts supporting *Turning Large Electron Microscopy
Datasets into Quantitative Measurements with Model-Assisted Analysis*.

Start with **`REPRODUCING_THE_PAPER.md`**.

- `paper/data/` — numerical figure and table data, manifests, annotations,
  example images, motion-correction and SEM-intensity measurements
- `paper/configs/` — model configurations, resolved training arguments, checkpoint
  SHA-256 hashes, simulation parameters and seeds
- `paper/scripts/` — figure generation, assembly and analysis
- `DATA_MANIFEST.csv` / `SHA256SUMS` — file inventory and checksums

## Quick validation

From the repository root, run:

```bash
python paper/scripts/smoke_test.py
```

The smoke test uses only the Python standard library. It validates package
checksums, JSON records, portable paths, and required release metadata.

Model weights and native instrument files are stored in the companion DOI
archive rather than Git. Their hashes are recorded in the release manifests and
`paper/configs/model_checkpoints.csv`.

The native data, trained checkpoints, and complete archival package are
available from Constellation at
[doi:10.13139/ORNLNCCS/3920044](https://doi.org/10.13139/ORNLNCCS/3920044).

All scripts use repository-relative defaults. See `REPRODUCING_THE_PAPER.md`
for DOI archive placement, environment overrides, and known limitations.

## Citation

Citation metadata are provided in `CITATION.cff`. The manuscript DOI will be
added when assigned.

## Software license

The software source files are distributed under the MIT License; see `LICENSE`.
