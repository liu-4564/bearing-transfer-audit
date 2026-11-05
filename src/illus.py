"""Schematic components: bearing cross-section, test rig, rounded cards, arrows, chip labels.

All coordinates are in canvas units (69 x 46 by default, matching 6.9 x 4.6 in at
aspect='equal'), so a circle drawn here stays a circle and is not squashed by the
layout.
"""
import numpy as np
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch, Polygon,
                                Rectangle, Wedge)

# steel parts in a neutral grey-blue; data keep Okabe-Ito, so the two
# vocabularies never collide
STEEL = {"hi": "#E8ECF0", "mid": "#C3CBD4", "lo": "#97A3AF", "dark": "#5C6772"}
ACC = "#D55E00"      # fault / emphasis
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
GREY = "#666666"


def bearing(ax, cx, cy, R, defect=None, lw=0.55, z=3, label=False, fs=5.2,
            defect_angle=-45.0):
    """Bearing cross-section: outer race, 8 rolling elements, inner race, shaft,
    with an optional defect on either race."""
    # outer race
    ax.add_patch(Circle((cx, cy), R, fc=STEEL["hi"], ec=STEEL["dark"], lw=lw, zorder=z))
    ax.add_patch(Circle((cx, cy), R * 0.80, fc="white", ec=STEEL["dark"], lw=lw,
                        zorder=z + 0.1))
    # rolling elements
    for k in range(8):
        a = np.pi / 8 + k * np.pi / 4
        bx, by = cx + R * 0.62 * np.cos(a), cy + R * 0.62 * np.sin(a)
        ax.add_patch(Circle((bx, by), R * 0.135, fc=STEEL["mid"], ec=STEEL["dark"],
                            lw=lw * 0.8, zorder=z + 0.2))
    # inner race
    ax.add_patch(Circle((cx, cy), R * 0.44, fc=STEEL["hi"], ec=STEEL["dark"], lw=lw,
                        zorder=z + 0.3))
    ax.add_patch(Circle((cx, cy), R * 0.30, fc="white", ec=STEEL["dark"], lw=lw,
                        zorder=z + 0.4))
    # shaft
    ax.add_patch(Circle((cx, cy), R * 0.28, fc=STEEL["lo"], ec=STEEL["dark"],
                        lw=lw * 0.8, zorder=z + 0.5))
    ax.add_patch(Circle((cx, cy), R * 0.07, fc="white", ec=STEEL["dark"], lw=lw * 0.6,
                        zorder=z + 0.6, alpha=0.65))

    if defect == "inner":
        a = np.deg2rad(defect_angle)
        ax.add_patch(Wedge((cx, cy), R * 0.44, np.degrees(a) - 11, np.degrees(a) + 11,
                           width=R * 0.14, fc=ACC, ec="none", zorder=z + 0.7))
    elif defect == "outer":
        a = np.deg2rad(105)
        ax.add_patch(Wedge((cx, cy), R, np.degrees(a) - 10, np.degrees(a) + 10,
                           width=R * 0.20, fc=ACC, ec="none", zorder=z + 0.7))

    if label:
        ax.annotate("outer race", xy=(cx + R * 0.90 * np.cos(1.9), cy + R * 0.90 * np.sin(1.9)),
                    xytext=(cx - R * 1.5, cy - R * 1.02), fontsize=fs, color=GREY,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.45, color=GREY,
                                    shrinkA=0, shrinkB=0,
                                    connectionstyle="arc3,rad=0.15"))
        ax.annotate("rolling element", xy=(cx + R * 0.62 * np.cos(np.deg2rad(-70)),
                                           cy + R * 0.62 * np.sin(np.deg2rad(-70))),
                    xytext=(cx + R * 1.05, cy - R * 1.15), fontsize=fs, color=GREY,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.45, color=GREY,
                                    shrinkA=0, shrinkB=0))
        ax.annotate("shaft", xy=(cx + R * 0.12, cy + R * 0.12),
                    xytext=(cx + R * 1.35, cy + R * 0.95), fontsize=fs, color=GREY,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.45, color=GREY,
                                    shrinkA=0, shrinkB=0))


def rig(ax, cx, cy, w, h, lw=0.55, z=3, colors=None):
    """Test rig, side view: motor, coupling, bearing housing, accelerometer."""
    c = colors or {"motor": STEEL["mid"], "house": STEEL["hi"]}
    mw, mh = w * 0.40, h * 0.86
    # base plate
    ax.add_patch(Rectangle((cx - w * 0.5, cy - h * 0.58), w, h * 0.10, fc=STEEL["lo"],
                           ec="none", zorder=z))
    # motor
    ax.add_patch(FancyBboxPatch((cx - w * 0.5, cy - mh / 2), mw, mh,
                                boxstyle="round,pad=0,rounding_size=0.25",
                                fc=c["motor"], ec=STEEL["dark"], lw=lw, zorder=z + 0.2))
    for k in range(1, 4):
        x = cx - w * 0.5 + mw * k / 4.0
        ax.plot([x, x], [cy - mh * 0.36, cy + mh * 0.36], lw=lw * 0.7,
                color=STEEL["dark"], zorder=z + 0.3, alpha=0.75)
    # shaft
    ax.plot([cx - w * 0.5 + mw, cx + w * 0.26], [cy, cy], lw=h * 0.16,
            color=STEEL["lo"], solid_capstyle="butt", zorder=z + 0.1)
    # bearing housing
    ax.add_patch(Rectangle((cx + w * 0.10, cy - h * 0.30), w * 0.34, h * 0.60,
                           fc=c["house"], ec=STEEL["dark"], lw=lw, zorder=z + 0.4))
    ax.add_patch(Circle((cx + w * 0.27, cy), h * 0.17, fc="white", ec=STEEL["dark"],
                        lw=lw * 0.8, zorder=z + 0.5))
    # accelerometer
    ax.add_patch(Polygon([[cx + w * 0.16, cy + h * 0.30], [cx + w * 0.38, cy + h * 0.30],
                          [cx + w * 0.34, cy + h * 0.50], [cx + w * 0.20, cy + h * 0.50]],
                         closed=True, fc=ACC, ec="none", alpha=0.85, zorder=z + 0.6))


def card(ax, x, y, w, h, fc="#FAFBFC", ec="#D6DBE0", lw=0.6, rad=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rad}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, p0, p1, color=GREY, lw=0.7, style="-|>", rad=0.0, ls="-", z=5, ms=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=color, ls=ls, zorder=z,
                                 shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))


def chip(ax, x, y, text, fc=STEEL["hi"], ec="none", tc="#333333", fs=5.4, pad=0.42, z=6):
    ax.text(x, y, text, fontsize=fs, color=tc, ha="center", va="center", zorder=z + 1,
            bbox=dict(boxstyle=f"round,pad={pad},rounding_size=0.9", fc=fc, ec=ec, lw=0))
