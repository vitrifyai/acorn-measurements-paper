# Reviewer validation run log

Generated: 2026-09-21 (UTC)

## Scope and freeze

All evaluation inputs are listed with SHA-256 hashes in `INPUT_MANIFEST.csv`.
No raw image, label, sidecar, checkpoint, existing result, or manuscript file
was changed. New files were written only under `analysis/reviewer_validation/`.

Prespecified settings:

- Synthetic aggregate evaluation: the recorded Ultralytics test path and the
  selected `best.pt` checkpoint from each model-selection run.
- Per-image predictions: confidence 0.5, matching the existing count-analysis
  code in `figures.py`.
- Measurement matching: greedy one-to-one assignment in descending mask IoU,
  class matched, IoU threshold 0.5.
- Confidence intervals: 2,000 bootstrap resamples of source images, seed
  20260921. Objects were never resampled as independent observations.
- Calibration: TEM `px_a / 10`; SEM `pixel_size_nm`, both from the frozen test
  manifest.

## Commands

The first command was attempted in the filesystem sandbox:

```text
external/acorn/.venv-py312/bin/python \
  analysis/reviewer_validation/run_validation.py \
  --out analysis/reviewer_validation --device 0
```

It stopped before inference because CUDA was unavailable to the sandbox. The
input manifest from that attempt is preserved under
`failed_attempt_20260921_cuda_sandbox/`. The identical command was then run
with workstation GPU access and completed successfully.

## Software and hardware

- ACORN 0.3.0
- Python 3.12.13
- Ultralytics 8.4.41
- PyTorch 2.6.0+cu126
- NumPy 2.3.5
- OpenCV 5.0.0.93
- Pillow 12.3.0
- GPU: Tesla V100-SXM3-32GB, driver 580.173.02
- Playground commit: `be8aa3e78275464bbbcfea29b75e97ad81ad28d6`
- Validation script SHA-256:
  `f7ad0c15c569d4cf2ac9f9da4330be4716b8090ecae27c37ca0fac2c451bce08`

The complete machine-readable record is `run_record.json`.

## Checkpoints

- TEM: `0ad19cdfe34cc4515bc864c30481fa4a3277abe9c61a543fe3ea092ffe834597`
- SEM: `eb367758550ad4df1819bb3df26ca105201e1b69616c8b06fb8d5b9c35ab4c3b`

## Notes

The archived table metrics are class-0 metrics, not macro averages over the
target and debris classes. Both scopes are retained in
`synthetic_test_metrics.csv`; the class-0 rows exactly reproduce every
archived synthetic aggregate metric.

The aggregate row records `n_target_objects` from the simulator's attempted
object count. The scored YOLO labels contain 2,250 TEM target instances and
1,083 SEM target instances; these label counts are the denominators used for
matching and are the counts reported in `RESULTS_SUMMARY.md`.

