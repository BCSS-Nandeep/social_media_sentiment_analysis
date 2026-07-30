"""Evaluation metrics: global, per-language and ROC/AUC helpers."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from config import LABEL2ID, LABELS

logger = logging.getLogger("benchmark.metrics")


def _to_ids(labels: list[str]) -> np.ndarray:
    return np.asarray([LABEL2ID[label] for label in labels])


def compute_roc_auc(y_true: list[str], probs: np.ndarray) -> float:
    """ROC-AUC that degrades gracefully.

    Binary ground truth  -> standard ROC-AUC on the positive class.
    Multiclass           -> macro one-vs-rest; falls back to the micro-average
                            AUC when a class is missing from ``y_true``.
    """
    y_ids = _to_ids(y_true)
    present = np.unique(y_ids)
    if len(present) < 2:
        return float("nan")
    try:
        if len(present) == 2:
            pos = int(present.max())
            return float(roc_auc_score((y_ids == pos).astype(int), probs[:, pos]))
        return float(roc_auc_score(y_ids, probs, multi_class="ovr", average="macro"))
    except ValueError:
        fpr, tpr, _ = micro_roc_curve(y_true, probs)
        return float(auc(fpr, tpr))


def micro_roc_curve(y_true: list[str], probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Micro-averaged one-vs-rest ROC curve (robust to missing classes)."""
    y_ids = _to_ids(y_true)
    y_bin = label_binarize(y_ids, classes=list(range(len(LABELS))))
    if y_bin.shape[1] == 1:  # binary edge case: expand to two columns
        y_bin = np.hstack([1 - y_bin, y_bin])
    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs[:, : y_bin.shape[1]].ravel())
    return fpr, tpr, float(auc(fpr, tpr))


def evaluate_model(
    display_name: str,
    y_true: list[str],
    y_pred: list[str],
    probs: np.ndarray,
    confidences: np.ndarray,
    times_ms: np.ndarray,
) -> dict[str, Any]:
    """All global metrics for one model."""
    metrics: dict[str, Any] = {
        "model": display_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
        ),
        "roc_auc": compute_roc_auc(y_true, probs),
        "avg_confidence": float(np.mean(confidences)),
        "avg_time_ms": float(np.mean(times_ms)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, zero_division=0, digits=4
        ),
    }
    logger.info(
        "%s — accuracy %.4f | macro-F1 %.4f | avg %.1f ms/post",
        display_name, metrics["accuracy"], metrics["f1_macro"], metrics["avg_time_ms"],
    )
    return metrics


def language_breakdown(
    display_name: str,
    languages: list[str],
    y_true: list[str],
    y_pred: list[str],
) -> pd.DataFrame:
    """Per-language accuracy and F1 for one model."""
    frame = pd.DataFrame(
        {"language": languages, "y_true": y_true, "y_pred": y_pred}
    )
    rows = []
    for lang, group in frame.groupby("language"):
        rows.append(
            {
                "model": display_name,
                "language": lang,
                "n_posts": len(group),
                "accuracy": accuracy_score(group["y_true"], group["y_pred"]),
                "f1_macro": f1_score(
                    group["y_true"], group["y_pred"],
                    labels=LABELS, average="macro", zero_division=0,
                ),
                "f1_weighted": f1_score(
                    group["y_true"], group["y_pred"],
                    labels=LABELS, average="weighted", zero_division=0,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("n_posts", ascending=False).reset_index(drop=True)


def language_confusions(
    languages: list[str], y_true: list[str], y_pred: list[str]
) -> dict[str, np.ndarray]:
    """Per-language confusion matrices for one model."""
    frame = pd.DataFrame({"language": languages, "y_true": y_true, "y_pred": y_pred})
    return {
        str(lang): confusion_matrix(group["y_true"], group["y_pred"], labels=LABELS)
        for lang, group in frame.groupby("language")
    }


def pick_winner(metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Select the better model: macro-F1, then accuracy, then inference speed.

    Returns the winning metrics dict and the human-readable decision trail.
    """
    trail: list[str] = []
    for key, label, higher_is_better in (
        ("f1_macro", "macro F1-score", True),
        ("accuracy", "accuracy", True),
        ("avg_time_ms", "average inference time", False),
    ):
        a, b = round(float(metrics_a[key]), 4), round(float(metrics_b[key]), 4)
        if a == b:
            trail.append(f"{label}: tie ({a} vs {b}) — falling through to next criterion")
            continue
        a_wins = a > b if higher_is_better else a < b
        winner = metrics_a if a_wins else metrics_b
        trail.append(
            f"{label}: {metrics_a['model']} {a} vs {metrics_b['model']} {b} "
            f"-> {winner['model']} wins"
        )
        return winner, trail
    trail.append("All criteria tied — defaulting to the first model.")
    return metrics_a, trail
