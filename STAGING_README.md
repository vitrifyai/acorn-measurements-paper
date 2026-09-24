# ACORN methods paper: GitHub upload staging

This is the compact GitHub companion assembled on 2026-09-24.

## Included

- Numerical source data, manifests, annotations, configurations, and scripts.
- Current Figure 2 motion/dose source data and generator.
- Supplementary Table S16 results and validation records.
- Eighteen SEM examples and two compact cryo-TEM example PNGs.
- The ACORN MIT code license.

## Required before public upload

1. Add and test a CPU smoke-test command in a clean environment.
2. Update the reproduction guide for Figure 2, Figure S11, and Table S16.
3. Add an approved data license and `CITATION.cff`.
4. Regenerate the manifest and checksums after final edits.
5. Tag ACORN commit `a5608b10f11ebb6e58490a8513bcf4c1ec9d5fd3`.

Workstation-specific paths were removed on 2026-09-24. Do not upload publicly
until data-release approval is complete. Large native files and all paper
checkpoints are in the DOI folder;
GitHub Release assets should contain only selected native examples and the two
representative checkpoints needed by an optional inference demo.
