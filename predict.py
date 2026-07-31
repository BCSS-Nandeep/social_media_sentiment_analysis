"""CLI for the finalized production pipeline: IndicTrans2 -> Cardiff RoBERTa.

Usage:
    # Single post
    python predict.py --text "ప్రభుత్వం ప్రకటించిన కొత్త పథకం చాలా బాగుంది"

    # Several posts at once
    python predict.py --text "First post" --text "Second post"

    # Batch over a CSV (needs a post_text column; extra columns are preserved)
    python predict.py --input data/dataset.csv --output outputs/pipeline_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is importable regardless of the invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final end-to-end multilingual sentiment pipeline: "
            "IndicTrans2 (AI4Bharat) -> Cardiff Twitter RoBERTa."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--text", action="append",
        help="A post to classify (repeat the flag for several posts).",
    )
    source.add_argument("--input", help="CSV file with a post_text column.")
    parser.add_argument(
        "--output", default=str(config.OUTPUTS_DIR / "pipeline_predictions.csv"),
        help="Output CSV for --input mode (default: outputs/pipeline_predictions.csv)",
    )
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--translation-batch-size", type=int, default=config.TRANSLATION_BATCH_SIZE
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default=config.DEVICE
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    from src.utils import setup_logging  # deferred: heavy imports (torch)

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    from src.pipeline import SentimentPipeline
    from src.translation import TranslatorLoadError

    try:
        pipeline = SentimentPipeline(device=args.device, seed=args.seed)
    except TranslatorLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        if args.text:
            results = pipeline.predict_batch(
                args.text,
                batch_size=args.batch_size,
                translation_batch_size=args.translation_batch_size,
            )
            print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
            return 0

        return _run_csv(pipeline, args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    finally:
        pipeline.free()


def _run_csv(pipeline, args: argparse.Namespace) -> int:
    import pandas as pd

    from src.preprocessing import normalize_label
    from src.utils import resolve_path

    input_path = resolve_path(args.input, config.BASE_DIR)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")
    if "post_text" not in df.columns:
        raise ValueError(
            f"Input CSV must contain a 'post_text' column; found {list(df.columns)}"
        )

    results = pipeline.predict_batch(
        df["post_text"].astype(str).tolist(),
        batch_size=args.batch_size,
        translation_batch_size=args.translation_batch_size,
    )

    out = df.copy()
    out["language"] = [r.language for r in results]
    out["english_text"] = [r.english_text for r in results]
    out["was_translated"] = [r.was_translated for r in results]
    out["sentiment"] = [r.sentiment for r in results]
    out["confidence"] = [r.confidence for r in results]
    out["translation_time_ms"] = [r.translation_time_ms for r in results]
    out["sentiment_time_ms"] = [r.sentiment_time_ms for r in results]
    out["total_time_ms"] = [r.total_time_ms for r in results]

    output_path = resolve_path(args.output, config.BASE_DIR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} predictions to {output_path}")

    # Optional evaluation when ground truth is available.
    if "ground_truth_sentiment" in df.columns:
        try:
            y_true = [normalize_label(v) for v in df["ground_truth_sentiment"]]
        except ValueError:
            return 0
        from sklearn.metrics import accuracy_score, f1_score

        y_pred = [r.sentiment for r in results]
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, labels=config.LABELS, average="macro", zero_division=0)
        avg_ms = sum(r.total_time_ms for r in results) / len(results)
        print(
            f"Evaluation vs ground truth — accuracy {accuracy:.4f}, "
            f"macro-F1 {f1:.4f}, avg {avg_ms:.1f} ms/post"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
