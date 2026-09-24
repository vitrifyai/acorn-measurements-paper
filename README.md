# ACORN methods paper — data, configurations and scripts

Everything needed to regenerate every figure, table and number in the paper.

Start with **`REPRODUCING_THE_PAPER.md`**.

- `paper/data/` — numerical figure and table data, manifests, annotations,
  example images, motion-correction and SEM-intensity measurements
- `paper/configs/` — model configurations, resolved training arguments, checkpoint
  SHA-256 hashes, simulation parameters and seeds
- `paper/scripts/` — figure generation, assembly and analysis
- `DATA_MANIFEST.csv` / `SHA256SUMS` — file inventory and checksums

Model weights and native instrument files are stored in the companion DOI
archive rather than Git. Their hashes are recorded in the release manifests and
`paper/configs/model_checkpoints.csv`.

All scripts use repository-relative defaults. See `REPRODUCING_THE_PAPER.md`
for DOI archive placement, environment overrides, and known limitations.
