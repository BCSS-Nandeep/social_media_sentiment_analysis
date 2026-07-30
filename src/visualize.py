"""All benchmark visualizations (static PNG, matplotlib).

Palette: validated 2-series categorical palette — slot 1 blue for IndicBERT,
slot 2 orange for MuRIL (fixed assignment, never cycled); a single-hue blue
sequential ramp for confusion-matrix heatmaps; neutral gray for ground-truth
reference bars. Text always wears ink colors, never series colors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend — must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from config import LABELS

logger = logging.getLogger("benchmark.visualize")

# --------------------------------------------------------------------------- #
# Palette (light mode, validated)
# --------------------------------------------------------------------------- #
SERIES_COLORS: dict[str, str] = {}  # display_name -> hex, filled by register_series
_SLOT_ORDER = ["#2a78d6", "#eb6834"]  # blue, orange — fixed categorical order

GT_COLOR = "#898781"     # muted gray for ground-truth reference bars
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def register_series(display_names: list[str]) -> None:
    """Bind each model to its fixed categorical slot (assignment, not cycling)."""
    for name, color in zip(display_names, _SLOT_ORDER):
        SERIES_COLORS[name] = color


def _color(name: str) -> str:
    return SERIES_COLORS.get(name, _SLOT_ORDER[0])


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def _finish(fig: Figure, path: Path) -> Figure:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    logger.info("Saved plot: %s", path.name)
    return fig


def _legend_above(ax: plt.Axes) -> None:
    """Legend above the axes so it never collides with tall bars' value labels."""
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=4, borderaxespad=0)


def _bar_value_labels(ax: plt.Axes, bars, fmt: str) -> None:
    for bar in bars:
        ax.annotate(
            format(bar.get_height(), fmt),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, color=INK_SECONDARY,
        )


# --------------------------------------------------------------------------- #
# Comparison bars
# --------------------------------------------------------------------------- #
def plot_metric_bar(
    title: str,
    values: dict[str, float],
    path: Path,
    fmt: str = ".3f",
    ylabel: str | None = None,
    ylim_unit: bool = True,
) -> Figure:
    """A single-metric two-bar comparison chart."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    names = list(values)
    heights = [float(values[n]) for n in names]
    bars = ax.bar(
        names, heights,
        color=[_color(n) for n in names],
        width=0.45, zorder=3,
    )
    ax.grid(axis="y", zorder=0)
    ax.set_title(title, fontsize=12, color=INK, pad=12)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim_unit:
        ax.set_ylim(0, 1.08)
    _bar_value_labels(ax, bars, fmt)
    return _finish(fig, path)


def plot_metrics_overview(metrics_list: list[dict[str, Any]], path: Path) -> Figure:
    """Grouped bars: accuracy / precision / recall / F1 for both models."""
    _apply_style()
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "F1\n(weighted)"]
    x = np.arange(len(keys))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for i, m in enumerate(metrics_list):
        offset = (i - (len(metrics_list) - 1) / 2) * (width + 0.02)
        bars = ax.bar(
            x + offset, [float(m[k]) for k in keys],
            width=width, label=m["model"], color=_color(m["model"]), zorder=3,
        )
        _bar_value_labels(ax, bars, ".2f")
    ax.grid(axis="y", zorder=0)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model quality comparison", fontsize=12, color=INK, pad=24)
    _legend_above(ax)
    return _finish(fig, path)


def plot_grouped_bars(
    categories: list[str],
    series: dict[str, list[float]],
    title: str,
    path: Path,
    fmt: str = ".1f",
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Generic grouped-bar comparison: one group per category, one bar per series."""
    _apply_style()
    x = np.arange(len(categories))
    width = 0.8 / max(2, len(series))
    fig, ax = plt.subplots(figsize=(max(5.6, 1.8 * len(categories)), 4.2))
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (len(series) - 1) / 2) * (width + 0.02)
        bars = ax.bar(x + offset, values, width=width, label=name, color=_color(name), zorder=3)
        _bar_value_labels(ax, bars, fmt)
    ax.grid(axis="y", zorder=0)
    ax.set_xticks(x, categories)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=12, color=INK, pad=24)
    _legend_above(ax)
    return _finish(fig, path)


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], path: Path) -> Figure:
    """Micro-averaged one-vs-rest ROC curves, one line per model."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot([0, 1], [0, 1], color=BASELINE, linestyle="--", linewidth=1, zorder=2)
    for name, (fpr, tpr, auc_value) in curves.items():
        ax.plot(
            fpr, tpr, color=_color(name), linewidth=2, zorder=3,
            label=f"{name} (AUC = {auc_value:.3f})",
        )
    ax.grid(zorder=0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (micro-average, one-vs-rest)", fontsize=12, color=INK, pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    return _finish(fig, path)


def plot_inference_time(values: dict[str, float], path: Path) -> Figure:
    return plot_metric_bar(
        "Average inference time", values, path,
        fmt=".1f", ylabel="ms / post", ylim_unit=False,
    )


def plot_memory_usage(
    gpu_mb: dict[str, float], model_size_mb: dict[str, float], path: Path
) -> Figure:
    """Model size and GPU peak memory side by side."""
    _apply_style()
    names = list(model_size_mb)
    x = np.arange(2)
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * (width + 0.02)
        bars = ax.bar(
            x + offset,
            [model_size_mb[name], gpu_mb.get(name, 0.0)],
            width=width, label=name, color=_color(name), zorder=3,
        )
        _bar_value_labels(ax, bars, ".0f")
    ax.grid(axis="y", zorder=0)
    ax.set_xticks(x, ["Model size (MB)", "GPU peak (MB)"])
    ax.set_title("Memory footprint", fontsize=12, color=INK, pad=24)
    ax.set_ylabel("MB")
    _legend_above(ax)
    return _finish(fig, path)


def plot_language_metric(
    lang_df: pd.DataFrame, metric: str, title: str, path: Path
) -> Figure:
    """Grouped bars of a per-language metric, one group per language."""
    _apply_style()
    pivot = lang_df.pivot(index="language", columns="model", values=metric)
    languages = list(pivot.index)
    models = list(pivot.columns)
    x = np.arange(len(languages))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(5.6, 1.6 * len(languages)), 4.2))
    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * (width + 0.02)
        heights = pivot[model].fillna(0.0).to_numpy(dtype=float)
        bars = ax.bar(
            x + offset, heights, width=width,
            label=model, color=_color(model), zorder=3,
        )
        _bar_value_labels(ax, bars, ".2f")
    ax.grid(axis="y", zorder=0)
    ax.set_xticks(x, languages)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Detected language")
    ax.set_title(title, fontsize=12, color=INK, pad=24)
    _legend_above(ax)
    return _finish(fig, path)


def plot_prediction_distribution(
    ground_truth: list[str], predictions: dict[str, list[str]], path: Path
) -> Figure:
    """Class distribution: ground truth (gray reference) vs each model."""
    _apply_style()
    series = {"Ground truth": ground_truth, **predictions}
    x = np.arange(len(LABELS))
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (name, labels) in enumerate(series.items()):
        counts = [labels.count(cls) for cls in LABELS]
        offset = (i - (len(series) - 1) / 2) * (width + 0.015)
        color = GT_COLOR if name == "Ground truth" else _color(name)
        bars = ax.bar(x + offset, counts, width=width, label=name, color=color, zorder=3)
        _bar_value_labels(ax, bars, ".0f")
    ax.grid(axis="y", zorder=0)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Posts")
    ax.set_title("Prediction distribution vs ground truth", fontsize=12, color=INK, pad=24)
    _legend_above(ax)
    return _finish(fig, path)


def plot_confusion_matrix(cm: np.ndarray, model_name: str, path: Path) -> Figure:
    """Single-hue sequential heatmap with count annotations."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(cm, cmap=SEQ_CMAP, vmin=0)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"Confusion matrix — {model_name}", fontsize=12, color=INK, pad=12)
    ax.spines[:].set_visible(False)

    threshold = cm.max() * 0.55 if cm.max() else 0.5
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col, row, str(int(cm[row, col])),
                ha="center", va="center", fontsize=11,
                color="#ffffff" if cm[row, col] > threshold else INK,
            )
    fig.colorbar(image, ax=ax, shrink=0.8, label="Posts")
    return _finish(fig, path)


def text_page(title: str, lines: list[str]) -> Figure:
    """A text-only figure used as the PDF report's summary page."""
    _apply_style()
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(0.08, 0.94, title, fontsize=18, color=INK, weight="bold")
    fig.text(
        0.08, 0.90, "\n".join(lines),
        fontsize=9.5, color=INK_SECONDARY,
        family="monospace", va="top",
    )
    return fig
