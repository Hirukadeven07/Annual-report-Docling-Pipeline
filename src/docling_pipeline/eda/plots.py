"""Shared chart style and small plotting helpers for the EDA notebook."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("eda_blue", SEQUENTIAL_STEPS)
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
TIER_COLORS = {"A": STATUS["good"], "B": STATUS["warning"], "C": STATUS["critical"]}
TEXT_STATUS_COLORS = {"pass": STATUS["good"], "partial": STATUS["warning"], "fail": STATUS["critical"]}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
SURFACE = "#fcfcfb"


def apply_style() -> None:
    """Set matplotlib defaults: light surface, recessive grid and axes, thin marks, fixed series order."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK["secondary"],
        "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["secondary"],
        "ytick.labelcolor": INK["secondary"],
        "text.color": INK["primary"],
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


def stacked_share(table: pd.DataFrame, colors: dict[str, str] | list[str], title: str, ax: plt.Axes | None = None) -> plt.Axes:
    """Draw horizontal 100% stacked bars, one per row of ``table``, with a 2px surface gap between segments."""
    shares = table.div(table.sum(axis=1), axis=0).fillna(0)
    ax = ax or plt.subplots(figsize=(9, 0.32 * len(shares) + 1.2))[1]
    left = pd.Series(0.0, index=shares.index)
    palette = colors if isinstance(colors, dict) else dict(zip(shares.columns, colors))
    for column in shares.columns:
        ax.barh(shares.index.astype(str), shares[column], left=left, color=palette[column], label=str(column), edgecolor=SURFACE, linewidth=1.5, height=0.72)
        left += shares[column]
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.set_title(title, pad=24)
    ax.legend(ncols=min(len(shares.columns), 5), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5)
    return ax


def heatmap(table: pd.DataFrame, title: str, fmt: str = ".0%", ax: plt.Axes | None = None, vmin: float | None = 0, vmax: float | None = None, annotate: bool = True, colorbar: bool = True) -> plt.Axes:
    """Draw a single-hue heatmap with direct labels in the cells."""
    import seaborn as sns

    width = max(6, 0.55 * len(table.columns) + 3)
    height = max(2.5, 0.3 * len(table.index) + 1.2)
    ax = ax or plt.subplots(figsize=(width, height))[1]
    sns.heatmap(table, ax=ax, cmap=SEQUENTIAL, vmin=vmin, vmax=vmax, annot=annotate and table.size <= 400, fmt=fmt,
                annot_kws={"fontsize": 7.5}, linewidths=1, linecolor=SURFACE, cbar=colorbar, cbar_kws={"shrink": 0.6} if colorbar else None)
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)
    return ax
