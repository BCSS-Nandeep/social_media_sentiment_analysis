"""End-to-end orchestration of the two-stage translate-then-classify benchmark.

Stage 1: preprocess -> detect language -> translate with both pipelines ->
         BLEU/chrF/COMET/latency -> select best translation.
Stage 2: run both English sentiment models on the winning translations ->
         full sentiment metrics -> plots -> reports -> best end-to-end pipeline.
"""
from __future__ import annotations

import gc
import logging
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import config
from src import metrics as metrics_mod
from src import translation_metrics as tm
from src import visualize as viz
from src.inference import InferenceResult, SentimentClassifier
from src.language_detector import LanguageDetector
from src.preprocessing import preprocess_dataframe
from src.translation import TranslationResult, Translator, TranslatorLoadError
from src.utils import (
    format_seconds,
    get_peak_ram_mb,
    resolve_device,
    resolve_path,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("benchmark")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the CSV dataset with a clear error message on failure."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s; retrying with utf-8-sig.", path)
        df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Dataset {path} contains no rows.")
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _cm_text(cm: np.ndarray) -> str:
    """Confusion matrix rendered as a fenced markdown block."""
    width = max(len(label) for label in config.LABELS) + 2
    header = " " * width + "".join(f"{label:>{width}}" for label in config.LABELS)
    body = "\n".join(
        f"{label:>{width}}" + "".join(f"{int(v):>{width}}" for v in row)
        for label, row in zip(config.LABELS, cm)
    )
    return f"```\n(rows = ground truth, columns = predicted)\n{header}\n{body}\n```"


def _fmt(value: float, spec: str = ".4f") -> str:
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else format(value, spec)


def run_benchmark(args: Namespace) -> None:
    overall_start = time.perf_counter()
    logger_ = setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    set_seed(args.seed)
    device = resolve_device(args.device)

    input_path = resolve_path(args.input, config.BASE_DIR)
    out_dir = resolve_path(args.output_dir, config.BASE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load, validate, preprocess, detect language
    # ------------------------------------------------------------------ #
    df = load_dataset(input_path)
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        logger_.info("Sampled %d rows (--sample).", len(df))
    df = preprocess_dataframe(df)

    detector = LanguageDetector(seed=args.seed)
    df["language"] = detector.detect_batch(df["clean_text"].tolist())
    lang_counts = df["language"].value_counts()
    logger_.info("Language distribution: %s", lang_counts.to_dict())

    texts: list[str] = df["clean_text"].tolist()
    y_true: list[str] = df["ground_truth"].tolist()
    languages: list[str] = df["language"].tolist()

    references: list[str] | None = None
    if config.REFERENCE_COLUMN in df.columns:
        references = df[config.REFERENCE_COLUMN].fillna("").astype(str).tolist()
        if not any(ref.strip() for ref in references):
            references = None
    logger_.info(
        "Reference translations %s — BLEU/chrF/COMET will be %s.",
        "found" if references else "not found",
        "reference-based" if references else "reference-free/agreement",
    )

    # ------------------------------------------------------------------ #
    # Stage 1 — translation with both pipelines (serially, freeing memory)
    # ------------------------------------------------------------------ #
    trans_results: dict[str, TranslationResult] = {}
    trans_metrics: dict[str, dict[str, Any]] = {}
    for key, cfg in config.TRANSLATION_MODELS.items():
        try:
            translator = Translator(cfg, device)
        except TranslatorLoadError as exc:
            logger_.error(
                "%s excluded from the benchmark:\n%s", cfg.display_name, exc
            )
            continue
        result = translator.translate(
            texts, languages, batch_size=args.translation_batch_size
        )
        trans_results[key] = result
        trans_metrics[key] = tm.evaluate_translation(
            cfg.display_name,
            sources=texts,
            hypotheses=result.texts,
            references=references,
            times_ms=result.times_ms,
            total_time_s=result.total_time_s,
            translated_mask=result.translated_mask,
            model_size_mb=translator.size_mb,
            gpu_peak_mb=result.gpu_peak_mb,
            device=device.type,
            use_comet=not args.skip_comet,
        )
        translator.free()
        del translator
        gc.collect()

    trans_keys = list(trans_results)
    if not trans_keys:
        raise ValueError(
            "No translation pipeline could be loaded — fix the errors above "
            "(HuggingFace authentication / access / transformers version) and re-run."
        )

    if len(trans_keys) == 2:
        agreement = tm.agreement_scores(
            trans_results[trans_keys[0]].texts,
            trans_results[trans_keys[1]].texts,
            trans_results[trans_keys[0]].translated_mask
            | trans_results[trans_keys[1]].translated_mask,
        )
        logger_.info(
            "Cross-pipeline agreement — BLEU %s, chrF %s",
            _fmt(agreement["agreement_bleu"], ".2f"), _fmt(agreement["agreement_chrf"], ".2f"),
        )
        trans_winner, trans_trail = tm.pick_translation_winner(
            *[trans_metrics[k] for k in trans_keys]
        )
    else:
        agreement = {"agreement_bleu": float("nan"), "agreement_chrf": float("nan")}
        trans_winner = trans_metrics[trans_keys[0]]
        trans_trail = [
            f"{trans_winner['model']} was the only pipeline that loaded — "
            "selected by default (no comparison possible)."
        ]
    trans_winner_key = next(
        k for k in trans_keys
        if config.TRANSLATION_MODELS[k].display_name == trans_winner["model"]
    )
    logger_.info("Best translation pipeline: %s", trans_winner["model"])
    english_texts = trans_results[trans_winner_key].texts

    # translations.csv — both pipelines side by side
    translations_df = pd.DataFrame(
        {
            "id": df["id"] if "id" in df.columns else np.arange(1, len(df) + 1),
            "language": df["language"],
            "post_text": df["post_text"],
        }
    )
    for key in trans_keys:
        translations_df[f"{key}_translation"] = trans_results[key].texts
        translations_df[f"{key}_time_ms"] = np.round(trans_results[key].times_ms, 3)
    if references:
        translations_df["reference_translation"] = references
    translations_df["selected_pipeline"] = trans_winner["model"]
    translations_df.to_csv(out_dir / "translations.csv", index=False, encoding="utf-8-sig")

    trans_table = pd.DataFrame(
        {
            "metric": [
                "BLEU", "chrF", "COMET", "Posts translated",
                "Avg latency (ms/post)", "Total time (s)",
                "Model size (MB)", "GPU peak (MB)",
            ],
            **{
                config.TRANSLATION_MODELS[k].display_name: [
                    _fmt(trans_metrics[k]["bleu"], ".2f"),
                    _fmt(trans_metrics[k]["chrf"], ".2f"),
                    _fmt(trans_metrics[k]["comet"], ".4f"),
                    trans_metrics[k]["n_translated"],
                    f"{trans_metrics[k]['avg_time_ms']:.1f}",
                    f"{trans_metrics[k]['total_time_s']:.1f}",
                    f"{trans_metrics[k]['model_size_mb']:.0f}",
                    f"{trans_metrics[k]['gpu_peak_mb']:.0f}",
                ]
                for k in trans_keys
            },
        }
    )
    trans_table.to_csv(out_dir / "translation_metrics.csv", index=False)

    # ------------------------------------------------------------------ #
    # Stage 2 — sentiment classification on the winning translations
    # ------------------------------------------------------------------ #
    sent_results: dict[str, InferenceResult] = {}
    resources: dict[str, dict[str, float | bool]] = {}
    for key, cfg in config.SENTIMENT_MODELS.items():
        classifier = SentimentClassifier(cfg, device, max_length=args.max_length)
        result = classifier.predict(english_texts, batch_size=args.batch_size)
        sent_results[key] = result
        resources[key] = {
            "model_size_mb": classifier.size_mb,
            "gpu_peak_mb": result.gpu_peak_mb,
            "is_binary": classifier.is_binary,
        }
        classifier.free()
        del classifier
        gc.collect()

    model_metrics: dict[str, dict[str, Any]] = {}
    lang_frames: list[pd.DataFrame] = []
    lang_cms: dict[str, dict[str, np.ndarray]] = {}
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for key, cfg in config.SENTIMENT_MODELS.items():
        res = sent_results[key]
        model_metrics[key] = metrics_mod.evaluate_model(
            cfg.display_name, y_true, res.labels, res.probs, res.confidences, res.times_ms
        )
        model_metrics[key]["total_time_s"] = res.total_time_s
        lang_frames.append(
            metrics_mod.language_breakdown(cfg.display_name, languages, y_true, res.labels)
        )
        lang_cms[key] = metrics_mod.language_confusions(languages, y_true, res.labels)
        roc_curves[cfg.display_name] = metrics_mod.micro_roc_curve(y_true, res.probs)
    lang_df = pd.concat(lang_frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    # Output CSVs (sentiment stage)
    # ------------------------------------------------------------------ #
    base_cols = pd.DataFrame(
        {
            "id": df["id"] if "id" in df.columns else np.arange(1, len(df) + 1),
            "platform": df.get("platform", pd.Series([""] * len(df))),
            "topic": df.get("topic", pd.Series([""] * len(df))),
            "post_text": df["post_text"],
            "english_text": english_texts,
            "ground_truth": df["ground_truth"],
            "language": df["language"],
        }
    )
    combined = base_cols.copy()
    for key, cfg in config.SENTIMENT_MODELS.items():
        res = sent_results[key]
        per_model = base_cols.copy()
        per_model["prediction"] = res.labels
        per_model["confidence"] = np.round(res.confidences, 4)
        per_model["time_ms"] = np.round(res.times_ms, 3)
        per_model.to_csv(out_dir / f"predictions_{key}.csv", index=False, encoding="utf-8-sig")

        prefix = cfg.display_name.split()[0]  # Cardiff / SieBERT
        combined[f"{prefix}_prediction"] = res.labels
        combined[f"{prefix}_confidence"] = np.round(res.confidences, 4)
        combined[f"{prefix}_time_ms"] = np.round(res.times_ms, 3)
    combined.to_csv(out_dir / "predictions_combined.csv", index=False, encoding="utf-8-sig")

    metric_rows = [
        ("Accuracy", "accuracy", ".4f"),
        ("Precision (macro)", "precision_macro", ".4f"),
        ("Recall (macro)", "recall_macro", ".4f"),
        ("F1 (macro)", "f1_macro", ".4f"),
        ("F1 (weighted)", "f1_weighted", ".4f"),
        ("ROC-AUC", "roc_auc", ".4f"),
        ("Avg confidence", "avg_confidence", ".4f"),
        ("Avg inference time (ms/post)", "avg_time_ms", ".2f"),
        ("Total inference time (s)", "total_time_s", ".2f"),
    ]
    comparison = pd.DataFrame(
        {"metric": [row[0] for row in metric_rows] + ["Model size (MB)", "GPU peak memory (MB)", "Binary model (Neutral approximated)"]}
    )
    for key, cfg in config.SENTIMENT_MODELS.items():
        column = [format(float(model_metrics[key][field]), fmt) for _, field, fmt in metric_rows]
        column += [
            f"{resources[key]['model_size_mb']:.1f}",
            f"{resources[key]['gpu_peak_mb']:.1f}",
            str(bool(resources[key]["is_binary"])),
        ]
        comparison[cfg.display_name] = column
    comparison.to_csv(out_dir / "comparison_metrics.csv", index=False)
    lang_df.to_csv(out_dir / "language_metrics.csv", index=False)

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    viz.register_series([cfg.display_name for cfg in config.TRANSLATION_MODELS.values()])
    viz.register_series([cfg.display_name for cfg in config.SENTIMENT_MODELS.values()])
    figures: list[plt.Figure] = []

    quality_categories, quality_series = [], {}
    for metric_key, label in (("bleu", "BLEU"), ("chrf", "chrF")):
        if not any(np.isnan(trans_metrics[k][metric_key]) for k in trans_keys):
            quality_categories.append(label)
    if quality_categories:
        quality_series = {
            config.TRANSLATION_MODELS[k].display_name: [
                trans_metrics[k]["bleu" if cat == "BLEU" else "chrf"]
                for cat in quality_categories
            ]
            for k in trans_keys
        }
        figures.append(
            viz.plot_grouped_bars(
                quality_categories, quality_series,
                "Translation quality (higher is better)",
                out_dir / "translation_quality.png", fmt=".1f", ylabel="score",
            )
        )
    figures.append(
        viz.plot_grouped_bars(
            ["Avg latency (ms/post)"],
            {
                config.TRANSLATION_MODELS[k].display_name: [trans_metrics[k]["avg_time_ms"]]
                for k in trans_keys
            },
            "Translation latency", out_dir / "translation_latency.png",
            fmt=".0f", ylabel="ms / post",
        )
    )

    def by_name(field: str) -> dict[str, float]:
        return {
            cfg.display_name: float(model_metrics[key][field])
            for key, cfg in config.SENTIMENT_MODELS.items()
        }

    figures.append(
        viz.plot_metrics_overview(list(model_metrics.values()), out_dir / "metrics_comparison.png")
    )
    for metric_key, title, filename in (
        ("accuracy", "Accuracy comparison", "accuracy_comparison.png"),
        ("precision_macro", "Precision (macro) comparison", "precision_comparison.png"),
        ("recall_macro", "Recall (macro) comparison", "recall_comparison.png"),
        ("f1_macro", "F1 (macro) comparison", "f1_comparison.png"),
    ):
        figures.append(viz.plot_metric_bar(title, by_name(metric_key), out_dir / filename))
    figures.append(viz.plot_roc(roc_curves, out_dir / "roc_curve.png"))
    figures.append(viz.plot_inference_time(by_name("avg_time_ms"), out_dir / "inference_time.png"))
    figures.append(
        viz.plot_memory_usage(
            {cfg.display_name: float(resources[key]["gpu_peak_mb"]) for key, cfg in config.SENTIMENT_MODELS.items()},
            {cfg.display_name: float(resources[key]["model_size_mb"]) for key, cfg in config.SENTIMENT_MODELS.items()},
            out_dir / "memory_usage.png",
        )
    )
    figures.append(
        viz.plot_language_metric(lang_df, "accuracy", "Language-wise accuracy", out_dir / "language_accuracy.png")
    )
    figures.append(
        viz.plot_language_metric(lang_df, "f1_macro", "Language-wise F1 (macro)", out_dir / "language_f1.png")
    )
    figures.append(
        viz.plot_prediction_distribution(
            y_true,
            {cfg.display_name: sent_results[key].labels for key, cfg in config.SENTIMENT_MODELS.items()},
            out_dir / "prediction_distribution.png",
        )
    )
    for key, cfg in config.SENTIMENT_MODELS.items():
        figures.append(
            viz.plot_confusion_matrix(
                model_metrics[key]["confusion_matrix"], cfg.display_name,
                out_dir / f"confusion_matrix_{key}.png",
            )
        )

    # ------------------------------------------------------------------ #
    # Winner + reports
    # ------------------------------------------------------------------ #
    metric_values = list(model_metrics.values())
    sent_winner, sent_trail = metrics_mod.pick_winner(metric_values[0], metric_values[1])
    total_runtime = time.perf_counter() - overall_start
    peak_ram = get_peak_ram_mb()

    _write_summary_md(
        out_dir, input_path, df, lang_counts, trans_metrics, agreement, trans_winner,
        trans_trail, model_metrics, resources, lang_df, lang_cms, sent_winner,
        sent_trail, total_runtime, peak_ram, device.type, references is not None,
    )
    _write_best_pipeline_md(
        out_dir, trans_winner, trans_trail, model_metrics, lang_df, sent_winner, sent_trail
    )
    _write_pdf(out_dir, figures, trans_winner, model_metrics, sent_winner, total_runtime, peak_ram)

    for fig in figures:
        plt.close(fig)

    logger_.info("=" * 64)
    logger_.info("BEST TRANSLATION PIPELINE : %s", trans_winner["model"])
    logger_.info("BEST SENTIMENT MODEL      : %s", sent_winner["model"])
    logger_.info(
        "BEST END-TO-END PIPELINE  : %s -> %s", trans_winner["model"], sent_winner["model"]
    )
    logger_.info("Total runtime: %s | Peak RAM: %.0f MB", format_seconds(total_runtime), peak_ram)
    logger_.info("All outputs written to %s", out_dir)


# --------------------------------------------------------------------------- #
# Report writers
# --------------------------------------------------------------------------- #
def _write_summary_md(
    out_dir: Path,
    input_path: Path,
    df: pd.DataFrame,
    lang_counts: pd.Series,
    trans_metrics: dict[str, dict[str, Any]],
    agreement: dict[str, float],
    trans_winner: dict[str, Any],
    trans_trail: list[str],
    model_metrics: dict[str, dict[str, Any]],
    resources: dict[str, dict[str, float | bool]],
    lang_df: pd.DataFrame,
    lang_cms: dict[str, dict[str, np.ndarray]],
    sent_winner: dict[str, Any],
    sent_trail: list[str],
    total_runtime: float,
    peak_ram: float,
    device_type: str,
    has_references: bool,
) -> None:
    parts: list[str] = ["# Benchmark Summary\n"]
    parts.append(f"- **Dataset:** `{input_path.name}` ({len(df)} posts)")
    parts.append(f"- **Device:** {device_type}")
    parts.append(f"- **Total runtime:** {format_seconds(total_runtime)}")
    parts.append(f"- **Peak RAM:** {peak_ram:.0f} MB")
    parts.append(f"- **Best translation pipeline:** **{trans_winner['model']}**")
    parts.append(f"- **Best sentiment model:** **{sent_winner['model']}**")
    parts.append(
        f"- **Best end-to-end pipeline:** **{trans_winner['model']} → {sent_winner['model']}**\n"
    )

    parts.append("## Dataset composition\n")
    class_rows = [[label, int((df["ground_truth"] == label).sum())] for label in config.LABELS]
    parts.append(_md_table(["Sentiment", "Posts"], class_rows) + "\n")
    parts.append(
        _md_table(["Language", "Posts"], [[lang, int(n)] for lang, n in lang_counts.items()]) + "\n"
    )

    parts.append("## Stage 1 — Translation quality\n")
    if not has_references:
        parts.append(
            "> No `reference_translation` column found — BLEU/chrF are unavailable; "
            "COMET (if shown) is reference-free quality estimation.\n"
        )
    headers = ["Metric"] + [m["model"] for m in trans_metrics.values()]
    rows = [
        ["BLEU"] + [_fmt(m["bleu"], ".2f") for m in trans_metrics.values()],
        ["chrF"] + [_fmt(m["chrf"], ".2f") for m in trans_metrics.values()],
        ["COMET"] + [_fmt(m["comet"], ".4f") for m in trans_metrics.values()],
        ["Posts translated"] + [m["n_translated"] for m in trans_metrics.values()],
        ["Avg latency (ms/post)"] + [f"{m['avg_time_ms']:.1f}" for m in trans_metrics.values()],
        ["Total time (s)"] + [f"{m['total_time_s']:.1f}" for m in trans_metrics.values()],
        ["Model size (MB)"] + [f"{m['model_size_mb']:.0f}" for m in trans_metrics.values()],
        ["GPU peak (MB)"] + [f"{m['gpu_peak_mb']:.0f}" for m in trans_metrics.values()],
    ]
    parts.append(_md_table(headers, rows) + "\n")
    parts.append(
        f"Cross-pipeline agreement: BLEU {_fmt(agreement['agreement_bleu'], '.2f')}, "
        f"chrF {_fmt(agreement['agreement_chrf'], '.2f')}\n"
    )
    parts.append("### Translation decision\n")
    parts.extend(f"1. {line}" for line in trans_trail)
    parts.append(f"\n**Selected translations: {trans_winner['model']}**\n")

    parts.append("## Stage 2 — Sentiment metrics\n")
    headers = ["Metric"] + [m["model"] for m in model_metrics.values()]
    rows = []
    for label, field, fmt in (
        ("Accuracy", "accuracy", ".4f"),
        ("Precision (macro)", "precision_macro", ".4f"),
        ("Recall (macro)", "recall_macro", ".4f"),
        ("F1 (macro)", "f1_macro", ".4f"),
        ("F1 (weighted)", "f1_weighted", ".4f"),
        ("ROC-AUC", "roc_auc", ".4f"),
        ("Avg confidence", "avg_confidence", ".4f"),
        ("Avg inference (ms/post)", "avg_time_ms", ".2f"),
        ("Total inference (s)", "total_time_s", ".2f"),
    ):
        rows.append([label] + [format(float(m[field]), fmt) for m in model_metrics.values()])
    rows.append(["Model size (MB)"] + [f"{r['model_size_mb']:.1f}" for r in resources.values()])
    rows.append(["GPU peak (MB)"] + [f"{r['gpu_peak_mb']:.1f}" for r in resources.values()])
    rows.append(
        ["Binary (Neutral approximated)"] + [str(bool(r["is_binary"])) for r in resources.values()]
    )
    parts.append(_md_table(headers, rows) + "\n")

    for key, m in model_metrics.items():
        parts.append(f"## {m['model']}\n")
        parts.append("### Classification report\n")
        parts.append(f"```\n{m['classification_report']}\n```\n")
        parts.append("### Confusion matrix\n")
        parts.append(_cm_text(m["confusion_matrix"]) + "\n")
        parts.append("### Language-wise confusion matrices\n")
        for lang, cm in sorted(lang_cms[key].items()):
            parts.append(f"**{lang}**\n\n" + _cm_text(cm) + "\n")

    parts.append("## Language-wise metrics\n")
    lang_rows = [
        [r.model, r.language, r.n_posts, f"{r.accuracy:.4f}", f"{r.f1_macro:.4f}", f"{r.f1_weighted:.4f}"]
        for r in lang_df.itertuples()
    ]
    parts.append(
        _md_table(["Model", "Language", "Posts", "Accuracy", "F1 (macro)", "F1 (weighted)"], lang_rows) + "\n"
    )

    parts.append("## Sentiment decision\n")
    parts.extend(f"1. {line}" for line in sent_trail)
    parts.append(
        f"\n**Best end-to-end pipeline: {trans_winner['model']} → {sent_winner['model']}**\n"
    )

    (out_dir / "benchmark_summary.md").write_text("\n".join(parts), encoding="utf-8")
    logger.info("Wrote benchmark_summary.md")


def _write_best_pipeline_md(
    out_dir: Path,
    trans_winner: dict[str, Any],
    trans_trail: list[str],
    model_metrics: dict[str, dict[str, Any]],
    lang_df: pd.DataFrame,
    sent_winner: dict[str, Any],
    sent_trail: list[str],
) -> None:
    loser = next(m for m in model_metrics.values() if m["model"] != sent_winner["model"])
    pivot = lang_df.pivot(index="language", columns="model", values="f1_macro")
    better_langs = [
        str(lang)
        for lang, row in pivot.iterrows()
        if row.get(sent_winner["model"], 0) > row.get(loser["model"], 0)
    ]

    lines = [
        "# Best End-to-End Pipeline\n",
        f"## Winner: {trans_winner['model']} → {sent_winner['model']}\n",
        "## Stage 1 — Translation\n",
        f"- Selected pipeline : {trans_winner['model']}",
        f"- BLEU : {_fmt(trans_winner['bleu'], '.2f')}",
        f"- chrF : {_fmt(trans_winner['chrf'], '.2f')}",
        f"- COMET : {_fmt(trans_winner['comet'], '.4f')}",
        f"- Avg latency : {trans_winner['avg_time_ms']:.1f} ms/post\n",
        "### Decision trail\n",
    ]
    lines.extend(f"1. {line}" for line in trans_trail)
    lines += [
        "\n## Stage 2 — Sentiment\n",
        f"- Selected model : {sent_winner['model']}",
        f"- Accuracy : {sent_winner['accuracy'] * 100:.1f}%",
        f"- F1 (macro) : {sent_winner['f1_macro'] * 100:.1f}%",
        f"- F1 (weighted) : {sent_winner['f1_weighted'] * 100:.1f}%",
        f"- Avg inference : {sent_winner['avg_time_ms']:.1f} ms/post",
    ]
    if better_langs:
        lines.append(
            f"- Outperforms {loser['model']} on macro-F1 for: {', '.join(sorted(better_langs))}"
        )
    lines.append("\n### Decision trail (highest macro F1 → accuracy → latency)\n")
    lines.extend(f"1. {line}" for line in sent_trail)
    lines += [
        "\n## Runner-up (sentiment)\n",
        f"- {loser['model']}: accuracy {loser['accuracy'] * 100:.1f}%, "
        f"macro-F1 {loser['f1_macro'] * 100:.1f}%, {loser['avg_time_ms']:.1f} ms/post",
    ]

    (out_dir / "BEST_PIPELINE.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote BEST_PIPELINE.md")


def _write_pdf(
    out_dir: Path,
    figures: list[plt.Figure],
    trans_winner: dict[str, Any],
    model_metrics: dict[str, dict[str, Any]],
    sent_winner: dict[str, Any],
    total_runtime: float,
    peak_ram: float,
) -> None:
    lines = [
        f"Best translation pipeline : {trans_winner['model']}",
        f"Best sentiment model      : {sent_winner['model']}",
        f"Best end-to-end pipeline  : {trans_winner['model']} -> {sent_winner['model']}",
        "",
    ]
    for m in model_metrics.values():
        lines += [
            f"{m['model']}",
            f"  accuracy       : {m['accuracy']:.4f}",
            f"  F1 (macro)     : {m['f1_macro']:.4f}",
            f"  F1 (weighted)  : {m['f1_weighted']:.4f}",
            f"  ROC-AUC        : {m['roc_auc']:.4f}",
            f"  avg inference  : {m['avg_time_ms']:.2f} ms/post",
            "",
        ]
    lines += [f"Total runtime : {format_seconds(total_runtime)}", f"Peak RAM      : {peak_ram:.0f} MB"]

    pdf_path = out_dir / "benchmark_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        cover = viz.text_page("Translate-then-Classify Sentiment Benchmark", lines)
        pdf.savefig(cover)
        plt.close(cover)
        for fig in figures:
            pdf.savefig(fig)
    logger.info("Wrote %s", pdf_path.name)
