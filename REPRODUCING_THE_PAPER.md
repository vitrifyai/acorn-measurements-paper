# Reproducing the paper

Everything needed to regenerate every figure, table and number is in this
repository or the companion DOI archive. Native instrument data and model
weights remain outside Git because of their size.

## 1. Environment

The authoritative lock is `uv.lock` with `pyproject.toml`.

```bash
uv venv --python 3.12.13
uv pip sync uv.lock
```

`requirements-frozen.txt` is the exact resolved set (134 packages) the published
results were produced with, and `environment.yml` is a conda-style equivalent for
readers who prefer it.

Recorded environment: Python 3.12.13, PyTorch 2.6.0+cu126, Ultralytics 8.4.41,
CUDA 12.6, Ubuntu 22.04.5, Tesla V100-SXM3-32GB.

## 2. Layout

```
paper/
  data/
    figure_source_data/   numbers behind every figure and table
    manifests/            image identifiers, train/val/test splits
    annotations/          reference annotations, YOLO polygon format
    example_images/       the 18 held-out acquired SEM micrographs
    motion_correction/    per-movie measurements, all 48
    sem_intensity/        per-field intensity and evaluation data
  configs/
    model_checkpoints.csv training runs with resolved args and weight SHA-256
    training_runs/        one args.yaml per run
    simulation/           simulation parameters and seeds, per image
  scripts/
    figures/              panel generation and 600 dpi assembly
    analysis/             evaluation, calibration, provenance, replication
DATA_MANIFEST.csv         every file with category, size and SHA-256
SHA256SUMS                `sha256sum -c SHA256SUMS` to verify
```

## 3. Configure portable paths

Place the unpacked DOI archive at `external/doi/`, and place an ACORN checkout
at `external/acorn/`. The ACORN checkout must resolve to commit
`a5608b10f11ebb6e58490a8513bcf4c1ec9d5fd3`.

Large assets may live elsewhere. Override any default without editing a script:

```bash
export ACORN_DOI_DIR=/path/to/unpacked-doi-archive
export ACORN_SOURCE_DIR=/path/to/acorn
export ACORN_WORK_DIR=/path/to/writable-work-directory
export ACORN_BUILD_DIR=/path/to/figure-output-directory
```

If unset, writable outputs go to `work/` and `build/` inside the repository.
Archived checkpoints are read from `<DOI>/models/<run>/best.pt`; new training
runs are written under `<WORK>/model_runs/`. Relative paths stored in recorded
configuration and provenance files are interpreted from the repository root.

## 4. Verify the bundle

```bash
python paper/scripts/smoke_test.py
# Equivalent checksum-only validation:
sha256sum -c SHA256SUMS
```

## 5. Regenerate the figures

Panels are generated at final printed size, then assembled; nothing is enlarged.
Run these commands from the repository root:

```bash
python paper/scripts/figures/make_figures.py
python paper/scripts/figures/make_fig5.py
python paper/scripts/figures/make_si.py
python paper/scripts/figures/build_all.py
python paper/scripts/figures/build_all_pdf.py
python paper/scripts/figures/qc_report.py
```

`build_all.py` writes a per-figure inventory including the rescale applied to
each panel and the resulting effective body-text size. Any panel below 6.8 pt is
a defect, not a tolerance.

## 6. Regenerate the analyses

```bash
python paper/scripts/analysis/run_burden_replicates.py
python paper/scripts/analysis/extend_n8.py
python paper/scripts/analysis/sim_full_capture.py
python paper/scripts/analysis/motion_full_capture.py
python paper/scripts/analysis/calibration_validation.py --out build/calibration
python paper/scripts/analysis/provenance_audit.py
```

## 7. What is not in Git

| item | where | why |
|---|---|---|
| Model weights (44 checkpoints) | companion DOI archive | too large for Git; SHA-256 values are in `paper/configs/model_checkpoints.csv` |
| Native cryo-TEM MRC and SEM TIFF files | companion DOI archive | native binary instrument output |
| Exact ACORN source archive | companion DOI archive | preserves the code at the recorded commit independently of GitHub |

## 8. Known limitations

1. **Arial and Helvetica were not installed** on the machine that built the
   figures. Liberation Sans (metric-compatible with Arial) was used. Rebuilding on
   a machine with Arial will produce marginally different glyph shapes at
   identical metrics.
2. **Deterministic runs really are deterministic.** All ten simulation-initialised
   executions at n = 8 return mask mAP@50 = 0.529393 with bit-identical weights.
   That is expected, not a sign the seeds were ignored.
