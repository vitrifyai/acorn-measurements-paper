#!/usr/bin/env python3
"""Build every main and SI figure. Panel counts follow the corrections:
S4 = A-D (key unlettered), S6 = A-C (no D), S8 = A-H (key unlettered)."""
from __future__ import annotations
import sys as _release_sys
from pathlib import Path as _ReleasePath

_RELEASE_SCRIPTS = _ReleasePath(__file__).resolve().parents[1]
if str(_RELEASE_SCRIPTS) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_SCRIPTS))
from release_paths import (
    ACORN_SOURCE_DIR, ADDITIONAL_INFO_DIR, BUILD_DIR, FIGURE_SCRIPTS_DIR,
    MODEL_CHECKPOINTS_DIR, MODEL_RUNS_DIR, RAW_CRYO_DIR, RAW_SEM_DIR, WORK_DIR,
)

import json, sys
from pathlib import Path
B=BUILD_DIR
sys.path.insert(0,str(B))
from assemble import build, FULL_PX, DPI
P=B/"build"; AI=P/"ai/figures"
AI4=P/"ai4/figures"      # same panels drawn at 4-across width
rep=[]

# ---- MAIN FIGURES: all four-panel figures as balanced 2x2 grids -------------
rep.append(build("figure2_motion_correction",
    [[P/"fig5/5A_uncorrected.png", P/"fig5/5B_motion_corrected.png"],
     [P/"fig5/5C_trajectory.png",  P/"fig2_motion/2D_dose_dependence.png"]], B/"main"))
rep.append(build("figure3_cryoTEM_use_case",
    [[P/"fig2/2A.png", P/"fig2/2B.png"],[P/"fig2/2C.png", P/"fig2/2D.png"]], B/"main"))
rep.append(build("figure4_SEM_use_case",
    [[P/"fig3/3A.png", P/"fig3/3B.png"],[P/"fig3/3C.png", P/"fig3/3D.png"]], B/"main"))
# Figure 5: both simulated modalities are shown, so the picked PLGA nanoparticles
# sit beside the picked spores, then the two transport validations, then the two
# summary plots. Three rows of two, paired by what each row is doing.
rep.append(build("figure5_simulation",
    [[P/"fig4/4A_cryoTEM.png", P/"fig4/4A_SEM.png"],
     [P/"fig4/4B_cryoTEM.png", P/"fig4/4B_SEM.png"],
     [P/"fig4/4C.png",         P/"fig4/4D.png"]], B/"main"))

# ---- SUPPORTING FIGURES -----------------------------------------------------
rep.append(build("figureS1_kernel_fidelity",
    [[P/"si/S1/S1A_radial_kernels.png", P/"si/S1/S1B_kernel_vs_trajectory.png"],
     [P/"si/S1/S1C_silicon_SE_kernel.png", P/"si/S1/S1D_weight_discarded.png"]], B/"si"))
rep.append(build("figureS2_counting_error",
    [[P/"si/S2/S2A_scanning_conditions.png", P/"si/S2/S2B_transmission_dose.png"],
     [P/"si/S2/S2C_error_vs_count.png"]], B/"si", row_frac=[1.0, 0.5]))
rep.append(build("figureS3_training_curves",
    [[P/"si/S3/S3A_transmission.png", P/"si/S3/S3B_scanning.png"],
     [P/"si/S3/S3C_loss_components.png"]], B/"si", row_frac=[1.0, 0.5]))
# S4: four image panels A-D; the colour key is NOT panel E
rep.append(build("figureS4_simulated_heldout",
    [[P/"si/S4/S4_transmission_1.png", P/"si/S4/S4_transmission_2.png"],
     [P/"si/S4/S4_scanning_1.png",     P/"si/S4/S4_scanning_2.png"],
     [P/"si/S4/S4_legend.png"]], B/"si", skip_letters={4},
    row_frac=[1.0, 1.0, 0.62]))
# S5: micrographs and histograms need room, so one modality per row
rep.append(build("figureS5_simulated_vs_acquired",
    [[P/"si/S5/S5_transmission_simulated.png", P/"si/S5/S5_transmission_acquired.png"],
     [P/"si/S5/S5_scanning_simulated.png", P/"si/S5/S5_scanning_acquired.png"],
     [P/"si/S5/S5_transmission_histograms.png", P/"si/S5/S5_scanning_histograms.png"]],
    B/"si"))
# S6: A-C only. One large transmission panel, then the two comparable SEM panels.
rep.append(build("figureS6_simulation_only_transfer",
    [[P/"si/S6/S6_transmission_simulation_only.png"],
     [P/"si/S6/S6_scanning_simulation_only.png",
      P/"si/S6/S6_scanning_plus_four_acquired.png"],
     [P/"si/S6/S6_legend.png"]], B/"si", skip_letters={3},
    row_frac=[0.52, 1.0, 0.62]))
rep.append(build("figureS7_reference_composition",
    [[P/"si/S7/S7A_composition.png", P/"si/S7/S7B_recall_precision_by_subset.png"]], B/"si"))
# S8: A-H only; the key is not panel I
rep.append(build("figureS8_heldout_predictions",
    [[AI4/"figS6/S6_transmission_micrograph.png", AI4/"figS6/S6_transmission_reference.png",
      AI4/"figS6/S6_transmission_prediction.png", AI4/"figS6/S6_transmission_errors.png"],
     [AI4/"figS6/S6_scanning_micrograph.png", AI4/"figS6/S6_scanning_reference.png",
      AI4/"figS6/S6_scanning_prediction.png", AI4/"figS6/S6_scanning_errors.png"],
     [AI4/"figS6/S6_legend.png"]], B/"si", skip_letters={8},
    row_frac=[1.0, 1.0, 0.62]))
rep.append(build("figureS9_measurement_trace",
    [[AI/"figS7/S7A_trace_median_object.png"],
     [AI/"figS7/S7B_trace_model_generated.png"],
     [AI/"figS7/S7C_source_micrograph.png",   AI/"figS7/S7D_traced_mask.png"]],
    B/"si", row_frac=[1.0, 1.0, 0.86]))
# S10 is a single composed figure with its own internal panels
rep.append(build("figureS10_sem_intensity",
    [[P/"si/S10/figureS10_sem_intensity.png"]], B/"si", letters=False))

(B/"qc/figure_inventory.json").write_text(json.dumps(rep,indent=1))
for r in rep:
    print(f"  {r['name']:36s} {r['px'][0]}x{r['px'][1]} px  "
          f"{r['inches'][0]}x{r['inches'][1]} in  letters={r['letters'] or '(none)'}")
