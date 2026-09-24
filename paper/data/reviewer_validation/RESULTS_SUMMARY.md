# Reviewer validation results

## Completed analyses

### Untouched synthetic partitions

The selected checkpoints were run once on the 60-image TEM and 60-image SEM
test partitions. The class-0 results exactly reproduce the archived values:

| Modality | Labelled targets | Precision | Recall | F1 | Box mAP50 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TEM, nanoparticles | 2,250 | 0.944 | 0.932 | 0.938 | 0.984 | 0.935 | 0.577 |
| SEM, spores | 1,083 | 0.972 | 0.988 | 0.980 | 0.974 | 0.990 | 0.872 |

The manuscript/table values are target-class values. They must not be
described as macro averages over target and debris. Macro-average results are
also retained in `synthetic_test_metrics.csv` for transparency.

At the prespecified confidence of 0.5, count error reproduced the existing
condition table: TEM mean error was -3.9%, -1.2%, and -1.4% at 20, 40, and
60 e-/A2; SEM mean error was +7.5% for coated, +2.5% for uncoated, and +0.4%
for uncoated fields with charging. All six median errors were 0.0%.

### Measurement accuracy

Mask-IoU matching covered 2,097/2,250 TEM targets (93.2%) and 1,069/1,083 SEM
targets (98.7%). Source-image bootstrap results were:

| Modality | Measurement | Bias | MAE | Mean absolute relative error |
|---|---|---:|---:|---:|
| TEM | area (nm2) | +189.8 | 246.3 | 21.5% |
| TEM | equivalent circular diameter (nm) | +2.90 | 3.60 | 10.1% |
| TEM | Feret diameter (nm) | +12.14 | 12.17 | 33.1% |
| SEM | area (nm2) | +36,024 | 58,446 | 7.6% |
| SEM | equivalent circular diameter (nm) | +23.21 | 31.48 | 3.7% |
| SEM | Feret diameter (nm) | +65.84 | 83.95 | 7.4% |

The exact 95% intervals and condition strata are in
`measurement_accuracy.csv`. Calibration-retention error was 0% because the
frozen manifest calibration was carried into every measurement. The remaining
error is segmentation-boundary error.

A Bland-Altman plot was not produced. Differences were strongly non-normal
(Shapiro-Wilk p < 1e-19 for every physical measurement), and area and Feret
errors were heteroscedastic. A conventional limits-of-agreement plot would
therefore imply assumptions these data do not satisfy.

### Manuscript consistency checks

- The 18-field initialization evaluation and the 8-field SEM held-out test are
  different partitions. The legacy `real_ft` experiment trained on 8 fields
  and evaluated on 18. The later experimental SEM benchmark used 26 fields
  split 14 train, 4 validation, and 8 test.
- The 1,375-record denominator is the final SEM annotation store across all 26
  source micrographs: 1,085 imported sidecar records, 51 SAM-origin records,
  and 239 YOLO-origin records. It is not the number of unique biological
  objects in the held-out set. There are also exactly 1,375 measurement rows.
- Motion-correction sharpness was computed as the variance of the vertical
  image gradient: `Var(gradient(image, axis=0))`. The reported percentage is
  `100 * (sharpness_after / sharpness_before - 1)`.
- CryoBlob is no longer appropriately cited as unpublished. A 2025 software
  release exists at DOI 10.5281/zenodo.15548974, and a 2026 SSRN preprint is
  available at DOI 10.2139/ssrn.6905758.
- Correlations based on eight SEM fields are underpowered and should remain
  explicitly descriptive, with no population-level inferential claim.

## Failed or blocked analyses

### Untouched experimental test set

No presently untouched experimental partition was identified. The existing
experimental test fields have already been used for held-out scoring,
representative figure selection, and qualitative review. They must not be
renamed as a new untouched test set. A prospective source-micrograph-level
test dataset is required.

### Independent simulation-initialization replication

The existing replication does not satisfy the requested experiment. Every
simulation-start fine-tuning run reuses the same `spores_sem` pretrained
checkpoint. Repeated fine-tuning seeds therefore do not constitute independent
simulation-pretraining replicates, and ten n=8 simulation runs are known to
have identical tensors and predictions. `initialization_replicates.csv`
preserves these runs but marks every one as non-independent.

Completing this priority requires at least three newly generated, nonoverlapping
simulation seed ranges, three separately trained simulation checkpoints, and
paired natural-image initializations using the same experimental subsets and
fine-tuning seeds. Until then, no general simulation-initialization advantage
is supported.

## Missing inputs

- A prospective experimental dataset untouched by prompt/rule/model/figure
  development.
- Three independent simulation-pretraining datasets and checkpoints for the
  requested initialization replication.

No required input was missing for the synthetic test or measurement-accuracy
analyses.

## Conclusions that changed

- The synthetic metrics are reproducible, but their scope must be stated as
  target class only.
- The full measurement path is more accurate for equivalent circular diameter
  than for Feret diameter, especially in TEM; segmentation boundary shape is
  the source of error, not calibration loss.
- Existing initialization repeats cannot support an independence-based claim.
- The experimental partitions already inspected cannot support a new
  untouched-test claim.

