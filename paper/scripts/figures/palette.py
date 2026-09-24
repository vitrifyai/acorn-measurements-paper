"""Colour and style for every ACORN figure, main text and supplementary.

AUTHORED HERE. The build brief referred to a supplied `palette.py`, but no file
was attached; this implements the specification given in that brief. If the
original turns up, replace this file rather than merging the two.

The rule that matters
---------------------
**Colour encodes modality and nothing else.** Slate is transmission, ochre is
scanning, in every panel. Where a panel needs a second distinction inside one
modality, use line style or marker shape, never a third colour.

That rule exists because the two codes had previously collided: the burden
curve used one pair for simulation-start against natural-image-start while the
counting panel used the same pair for transmission against scanning. Both
series in the burden curve are scanning data, so both are now ochre and are
separated by solid against dashed with open markers.

Detection overlays -- reference, true positive, false positive, undetected --
are a genuinely categorical set with no modality meaning, so they use their own
family and must not reuse the modality pair.
"""
from __future__ import annotations
import matplotlib as mpl


class P:
    # ── modality: the only thing colour encodes ──────────────────────────────
    TRANSMISSION = "#2F4858"     # slate
    SCANNING     = "#C08A2E"     # ochre

    # ── detection overlays: categorical, no modality meaning ────────────────
    REFERENCE    = "#1B9E77"     # teal   — annotated or verified reference object
    TRUE_POS     = "#7570B3"     # purple — matched detection
    FALSE_POS    = "#E7298A"     # magenta, dashed
    MISSED       = "#525252"     # dark grey, open marker

    # ── neutrals ────────────────────────────────────────────────────────────
    INK   = "#1b1f24"
    GREY  = "#5b6670"
    RULE  = "#c3c9d1"
    BAND  = "#eef0f3"

    @staticmethod
    def modality(tag: str) -> str:
        t = tag.lower()
        if t.startswith(("trans", "cryo", "tem", "nano", "plga")):
            return P.TRANSMISSION
        if t.startswith(("scan", "sem", "spore")):
            return P.SCANNING
        raise KeyError(f"no modality colour for {tag!r}; colour encodes modality only")


def apply_style() -> None:
    """Call once at the top of a figure script."""
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif",
        # Arial/Helvetica as required. Liberation Sans is metric-compatible
        # with Arial and Nimbus Sans is a Helvetica clone; both are installed.
        # Arial and Helvetica are not installed on this host. Liberation Sans is
        # metric-compatible with Arial (identical advance widths) and is the
        # standard substitution; Nimbus Sans is a Helvetica clone. Recorded in
        # the QC report rather than silently substituted.
        "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Arial",
                            "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "axes.edgecolor": P.INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.prop_cycle": mpl.cycler(color=[P.TRANSMISSION, P.SCANNING]),
    })


# ── panel assignments ────────────────────────────────────────────────────────
#
# Main text
#   2A 2B 2C 2D   transmission        -> slate throughout
#   3A 3B 3C 3D   scanning            -> ochre throughout
#   4A            one panel each, coloured by its own modality
#   4B_SEM        scanning            -> ochre
#   4B_cryoTEM    transmission        -> slate
#   4C            bars coloured by the modality each condition belongs to
#   4D            BOTH series ochre: both are scanning. simulation start solid
#                 with filled markers, natural-image start dashed with open
#                 markers.
#   5A-5D         transmission (the movie is cryo-TEM) -> slate
#
# Supplementary
#   S1A-S1D       scanning engine -> ochre; materials separated by marker and
#                 line style, not by colour
#   S2A           scanning -> ochre
#   S2B           transmission -> slate
#   S2C           one series per modality, each in its modality colour
#   S4, S6        detection overlays -> REFERENCE / TRUE_POS / FALSE_POS / MISSED
#   S5            each pair in its modality colour; acquired solid, simulated
#                 dashed, in the histograms
#   S7A           four reference categories: a categorical set, not modality.
#                 Uses the detection family, which carries no modality meaning.
#   S7B           recall against precision is not a modality distinction:
#                 modality colour (scanning, ochre) with solid against hatched.
