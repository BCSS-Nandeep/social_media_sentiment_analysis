"""CLI entry point for the translate-then-classify sentiment benchmark.

Usage:
    python run.py --input data/dataset.csv
    python run.py --input data/dataset.csv --device cpu --batch-size 8
    python run.py --input data/dataset.csv --skip-comet     # skip COMET scoring
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is importable regardless of the invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage benchmark: IndicTrans2 vs NLLB-200 translation, then "
            "Cardiff Twitter RoBERTa vs SieBERT sentiment on the best translation."
        )
    )
    parser.add_argument(
        "--input", default=str(config.DEFAULT_DATASET),
        help="Path to the input CSV (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--output-dir", default=str(config.OUTPUTS_DIR),
        help="Directory for all generated reports and plots (default: outputs/)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.BATCH_SIZE,
        help="Sentiment inference batch size",
    )
    parser.add_argument(
        "--translation-batch-size", type=int, default=config.TRANSLATION_BATCH_SIZE,
        help="Translation batch size",
    )
    parser.add_argument("--max-length", type=int, default=config.MAX_LENGTH)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default=config.DEVICE,
        help="Compute device (default: auto — GPU when available).",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Randomly sample N rows for a quick smoke run.",
    )
    parser.add_argument(
        "--skip-comet", action="store_true",
        help="Skip COMET scoring even when unbabel-comet is installed.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    from src.benchmark import run_benchmark  # deferred: heavy imports (torch)

    try:
        run_benchmark(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
