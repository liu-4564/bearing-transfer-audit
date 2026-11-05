"""Journal-grade plotting style.

Sources:
- rcParam ranges taken from SciencePlots (github.com/garrettj403/SciencePlots):
  serif text, inward ticks, thin frame, no top or right spine, 8 pt base,
  Type 42 fonts embedded in the PDF.
- Scientific colormaps after Crameri et al. (2020) Nature Communications 11:5444:
  colour-blind safe and perceptually uniform. Heatmaps no longer use RdYlGn, whose
  lightness is not monotonic and which is unreadable with red-green deficiency --
  a point reviewers single out.
- Canvas sizes follow the usual journal requirement: 85 mm single column, 170 mm
  double column, with no type below 6 pt in the final figure.

Usage:
    from .plotstyle import apply, SINGLE, DOUBLE, C, CM_SEQ, text_on
    apply()
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

try:                                    # Crameri colormaps (optional dependency)
    import cmcrameri.cm as _cmc
    CM_SEQ = _cmc.oslo_r                # sequential: white to dark blue, lightness falls monotonically, dark = high
    CM_SEQ2 = _cmc.imola_r              # sequential: pale yellow to dark blue, used for gain-like quantities
    CM_DIV = _cmc.vik                   # diverging: teal to brown, with a neutral midpoint
    CM_ALT = _cmc.hawaii
    HAS_CRAMERI = True
except Exception:                       # fall back to the built-in maps
    CM_SEQ, CM_SEQ2, CM_DIV, CM_ALT = ("viridis", "cividis", "RdBu_r", "magma")
    HAS_CRAMERI = False

# 85 mm single column, 170 mm double column
SINGLE = 3.35
ONE_HALF = 5.0
DOUBLE = 6.9

MIN_FONT = 6.0          # the usual lower bound on final type size

# Okabe-Ito colour-blind-safe palette, for data series
C = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "grey": "#666666",
    "black": "#111111",
}
MODEL_C = {"xgb": C["grey"], "cnn": C["blue"], "resnet": C["vermillion"]}
SPLIT_C = {"Random split": C["vermillion"], "Split by run": C["blue"]}


def apply():
    mpl.rcParams.update({
        # type: journal body text is usually a Times face
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        # sizes: 7-8 pt is the usual double-column range
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7,
        "figure.titlesize": 8.5,
        # lines and ticks
        "axes.linewidth": 0.55,
        "axes.edgecolor": C["black"],
        "axes.labelcolor": C["black"],
        "axes.axisbelow": True,
        "xtick.color": C["black"],
        "ytick.color": C["black"],
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.minor.visible": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.2,
        "patch.linewidth": 0.45,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.45,
        "legend.labelspacing": 0.32,
        "legend.borderaxespad": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed editable TrueType, as submission systems require
        "ps.fonttype": 42,
        "image.composite_image": True,
    })


def text_on(cmap, v, vmin, vmax, dark="#111111", light="white"):
    """Pick white or black cell text from the background lightness,
    so a dark cell never gets dark text."""
    t = (float(v) - vmin) / max(vmax - vmin, 1e-9)
    r, g, b = cmap(min(max(t, 0.0), 1.0))[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return light if lum < 0.52 else dark


def panel(ax, letter, dx=-0.14, dy=1.03):
    """Letter a panel (a), (b), ... in its top-left corner."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", va="bottom", ha="left")


def save(fig, name, both=True):
    """Write PDF for submission and PNG for preview, together."""
    import os
    from . import config
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    # no CreationDate: without this the PDF differs on every run, and two people
    # regenerating the same figure cannot agree on a hash
    fig.savefig(os.path.join(config.FIGURES_DIR, f"{name}.pdf"),
                metadata={"CreationDate": None})
    if both:
        fig.savefig(os.path.join(config.FIGURES_DIR, f"{name}.png"))
    plt.close(fig)
    print(f"{name}.pdf" + (" / .png" if both else ""))
