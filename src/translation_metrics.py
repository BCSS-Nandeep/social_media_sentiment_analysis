"""Translation quality metrics: BLEU, chrF, COMET, latency, memory.

Scoring modes
-------------
* With a ``reference_translation`` column: corpus BLEU / chrF / COMET against
  the human references (only over posts that were actually translated).
* Without references: reference-free COMET-QE when ``unbabel-comet`` is
  installed and the (gated) CometKiwi checkpoint is accessible; otherwise
  cross-pipeline agreement chrF/BLEU is reported for context — agreement is
  symmetric, so in that case the winner falls back to latency.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from config import (
    COMET_QE_MODEL,
    COMET_REF_MODEL,
    TRANSLATION_WINNER_PRIORITY,
)

logger = logging.getLogger("benchmark.translation_metrics")

_METRIC_LABELS = {
    "comet": "COMET",
    "chrf": "chrF",
    "bleu": "BLEU",
    "avg_time_ms": "average translation latency",
}
_HIGHER_IS_BETTER = {"comet": True, "chrf": True, "bleu": True, "avg_time_ms": False}


def corpus_scores(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    """Corpus BLEU and chrF via sacrebleu."""
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {"bleu": float(bleu.score), "chrf": float(chrf.score)}


def comet_score(
    sources: list[str],
    hypotheses: list[str],
    references: Optional[list[str]],
    device: str = "cpu",
    enabled: bool = True,
) -> Optional[float]:
    """System-level COMET (reference-based) or COMET-QE (reference-free).

    Returns ``None`` — never raises — when the optional ``unbabel-comet``
    package or its checkpoints are unavailable.
    """
    if not enabled:
        return None
    try:
        from comet import download_model, load_from_checkpoint

        model_name = COMET_REF_MODEL if references is not None else COMET_QE_MODEL
        checkpoint = download_model(model_name)
        model = load_from_checkpoint(checkpoint)
        data = [
            {"src": src, "mt": hyp, **({"ref": ref} if references is not None else {})}
            for src, hyp, ref in zip(
                sources, hypotheses, references or [""] * len(sources)
            )
        ]
        gpus = 1 if device == "cuda" else 0
        output = model.predict(data, batch_size=8, gpus=gpus, progress_bar=True)
        return float(output.system_score)
    except ImportError:
        logger.warning("unbabel-comet not installed — COMET score skipped.")
    except Exception as exc:
        logger.warning("COMET scoring failed (%s) — skipped.", exc)
    return None


def evaluate_translation(
    display_name: str,
    sources: list[str],
    hypotheses: list[str],
    references: Optional[list[str]],
    times_ms: np.ndarray,
    total_time_s: float,
    translated_mask: np.ndarray,
    model_size_mb: float,
    gpu_peak_mb: float,
    device: str = "cpu",
    use_comet: bool = True,
) -> dict[str, Any]:
    """All translation-stage metrics for one pipeline.

    BLEU/chrF/COMET are computed only over posts that were actually translated
    (pass-through English posts would inflate the scores).
    """
    idx = np.flatnonzero(translated_mask)
    metrics: dict[str, Any] = {
        "model": display_name,
        "n_translated": int(len(idx)),
        "bleu": float("nan"),
        "chrf": float("nan"),
        "comet": float("nan"),
        "avg_time_ms": float(np.mean(times_ms[idx])) if len(idx) else 0.0,
        "total_time_s": float(total_time_s),
        "model_size_mb": float(model_size_mb),
        "gpu_peak_mb": float(gpu_peak_mb),
        "has_references": references is not None,
    }
    if len(idx) == 0:
        logger.warning("%s: no posts were translated — quality metrics skipped.", display_name)
        return metrics

    sub_src = [sources[i] for i in idx]
    sub_hyp = [hypotheses[i] for i in idx]
    sub_ref = [references[i] for i in idx] if references is not None else None

    if sub_ref is not None:
        try:
            metrics.update(corpus_scores(sub_hyp, sub_ref))
        except ImportError:
            logger.warning("sacrebleu not installed — BLEU/chrF skipped.")
    comet = comet_score(sub_src, sub_hyp, sub_ref, device=device, enabled=use_comet)
    if comet is not None:
        metrics["comet"] = comet

    logger.info(
        "%s — BLEU %.2f | chrF %.2f | COMET %s | %.1f ms/post",
        display_name, metrics["bleu"], metrics["chrf"],
        f"{metrics['comet']:.4f}" if not np.isnan(metrics["comet"]) else "n/a",
        metrics["avg_time_ms"],
    )
    return metrics


def agreement_scores(
    hyp_a: list[str], hyp_b: list[str], translated_mask: np.ndarray
) -> dict[str, float]:
    """Symmetric cross-pipeline agreement (context only — cannot rank)."""
    idx = np.flatnonzero(translated_mask)
    if len(idx) == 0:
        return {"agreement_bleu": float("nan"), "agreement_chrf": float("nan")}
    a = [hyp_a[i] for i in idx]
    b = [hyp_b[i] for i in idx]
    try:
        import sacrebleu

        bleu = (
            sacrebleu.corpus_bleu(a, [b]).score + sacrebleu.corpus_bleu(b, [a]).score
        ) / 2
        chrf = (
            sacrebleu.corpus_chrf(a, [b]).score + sacrebleu.corpus_chrf(b, [a]).score
        ) / 2
        return {"agreement_bleu": float(bleu), "agreement_chrf": float(chrf)}
    except ImportError:
        return {"agreement_bleu": float("nan"), "agreement_chrf": float("nan")}


def pick_translation_winner(
    metrics_a: dict[str, Any], metrics_b: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Select the better translation pipeline.

    Walks ``TRANSLATION_WINNER_PRIORITY`` (COMET -> chrF -> BLEU -> latency),
    skipping metrics that are unavailable for this run.
    """
    trail: list[str] = []
    for key in TRANSLATION_WINNER_PRIORITY:
        a, b = float(metrics_a.get(key, float("nan"))), float(metrics_b.get(key, float("nan")))
        label = _METRIC_LABELS[key]
        if np.isnan(a) or np.isnan(b):
            trail.append(f"{label}: unavailable for this run — skipped")
            continue
        a_r, b_r = round(a, 4), round(b, 4)
        if a_r == b_r:
            trail.append(f"{label}: tie ({a_r} vs {b_r}) — falling through")
            continue
        a_wins = a_r > b_r if _HIGHER_IS_BETTER[key] else a_r < b_r
        winner = metrics_a if a_wins else metrics_b
        trail.append(
            f"{label}: {metrics_a['model']} {a_r} vs {metrics_b['model']} {b_r} "
            f"-> {winner['model']} wins"
        )
        return winner, trail
    trail.append("All criteria tied/unavailable — defaulting to the first pipeline.")
    return metrics_a, trail
