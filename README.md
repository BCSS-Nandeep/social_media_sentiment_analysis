# Multilingual Social Sentiment — Production Pipeline

**Finalized architecture (selected by benchmark):**

```
   Social Media Post
          │
          ▼
  Text Preprocessing + Language Detection
          │
   ┌──────┴───────────────┐
   │ English?             │ Indian language?
   ▼                      ▼
 bypass          IndicTrans2 (AI4Bharat)
   │              Indic → English
   └──────┬───────────────┘
          ▼
  Cardiff Twitter RoBERTa
 (cardiffnlp/twitter-roberta-base-sentiment-latest)
          │
          ▼
 Positive / Neutral / Negative
 + language, English text, confidence, inference time
```

## How to run — step by step

### Step 0. Prerequisites

- **Python 3.11 or 3.12** (`python --version` to check)
- ~5 GB free disk for model downloads; internet access on first run
- A free [HuggingFace account](https://huggingface.co/join)
- Optional: NVIDIA GPU with CUDA (used automatically; CPU works too, just slower)

### Step 1. Get the code

```bash
git clone https://github.com/BCSS-Nandeep/social_media_sentiment_analysis.git
cd social_media_sentiment_analysis
```

### Step 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows (PowerShell/cmd):
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate
```

### Step 3. Install dependencies

```bash
pip install -r requirements.txt
```

Do **not** upgrade `transformers` afterwards — it is pinned to `4.40.2` on
purpose (newer versions break IndicTrans2's custom code; see the note further
down).

### Step 4. Get access to the gated IndicTrans2 model (one-time)

1. Open <https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M>
   while logged in to HuggingFace and click **"Agree and access repository"**.
2. Create a **read** token at <https://huggingface.co/settings/tokens>, then
   authenticate this machine:

   ```bash
   huggingface-cli login
   ```

   (paste the token when prompted)

### Step 5. Quick test — classify one post

```bash
python predict.py --text "ప్రభుత్వం ప్రకటించిన కొత్త పథకం చాలా బాగుంది"
```

The first run downloads the two models (~1.5 GB total). You should get JSON
like:

```json
[
  {
    "post_text": "ప్రభుత్వం ప్రకటించిన కొత్త పథకం చాలా బాగుంది",
    "language": "te",
    "english_text": "The new scheme announced by the government is very good",
    "was_translated": true,
    "sentiment": "Positive",
    "confidence": 0.94,
    "translation_time_ms": 210.5,
    "sentiment_time_ms": 18.2,
    "total_time_ms": 228.7
  }
]
```

### Step 6. Batch run over a CSV

```bash
python predict.py --input data/dataset.csv --output outputs/pipeline_predictions.csv
```

- The CSV must have a `post_text` column; every other column is preserved.
- Results land in `outputs/pipeline_predictions.csv` with the detected
  language, English text, sentiment, confidence and per-stage timings added.
- If the CSV has a `ground_truth_sentiment` column, the last line printed is
  the evaluation, e.g. `Evaluation vs ground truth — accuracy 0.8421,
  macro-F1 0.8156, avg 240.3 ms/post`.

### Step 7 (optional). Re-run the full model-selection benchmark

```bash
python run.py --input data/dataset.csv
```

This compares IndicTrans2 vs NLLB-200 and Cardiff vs SieBERT again, writing
all metric tables, charts and `BEST_PIPELINE.md` to `outputs/` (details in the
benchmark section below).

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | machine not logged in to HF | Step 4.2 (`huggingface-cli login`) |
| `403 Forbidden` | model access not granted yet | Step 4.1 (click "Agree and access repository", wait a moment) |
| `No module named 'transformers.onnx'` or `past_key_values` shape error | transformers was upgraded | `pip install transformers==4.40.2` |
| Very slow on CPU | large batches / beams | add `--translation-batch-size 4`; keep `TRANSLATION_NUM_BEAMS = 1` in config.py |

## Production usage (reference)

```bash
# Single post — prints JSON with language, translation, sentiment, confidence, timings
python predict.py --text "ప్రభుత్వం ప్రకటించిన కొత్త పథకం చాలా బాగుంది"

# Several posts
python predict.py --text "First post" --text "Second post"

# Batch over a CSV (needs a post_text column; extra columns preserved).
# If a ground_truth_sentiment column exists, accuracy and macro-F1 are reported.
python predict.py --input data/dataset.csv --output outputs/pipeline_predictions.csv

# All options
python predict.py --help
```

Or from Python:

```python
from src.pipeline import SentimentPipeline

pipeline = SentimentPipeline()          # loads IndicTrans2 + Cardiff once
result = pipeline.predict_one("रैली में भीड़ थी लेकिन भाषण में कुछ नया नहीं था।")
print(result.language, result.sentiment, result.confidence, result.english_text)
```

Only these two models are loaded in production. The benchmark harness below is
retained for reference — it is how this architecture was selected — and can be
re-run at any time to re-validate the choice.

---

# Benchmark harness (how the winner was selected)

Two-stage comparison for Indian multilingual political social-media sentiment:

1. **Translation stage** — two competing Indic→English translation pipelines
   are benchmarked (BLEU / chrF / COMET / latency / memory) and the better one
   is selected.
2. **Sentiment stage** — two open-source English sentiment models classify the
   winning translations and are compared on the full metric suite; the best
   **end-to-end pipeline** is reported.

```
                     Social Media Post
                            │
                            ▼
                   Text Preprocessing
       (cleaning, Unicode, URL removal, language detection)
                            │
           ┌────────────────┴────────────────┐
           ▼                                 ▼
   IndicTrans2 (AI4Bharat)            NLLB-200 (600M)
   Translation Pipeline               Translation Pipeline
           │                                 │
           ▼                                 ▼
   English Translation A             English Translation B
           └────────────────┬────────────────┘
                            ▼
              Translation Quality Metrics
         (BLEU · chrF · COMET · latency · memory)
                            │
             Best English Translation Selected
                            │
           ┌────────────────┴────────────────┐
           ▼                                 ▼
   Cardiff Twitter RoBERTa           SieBERT RoBERTa Large
           │                                 │
           ▼                                 ▼
   Positive/Neutral/Negative        Positive/Neutral/Negative
           └────────────────┬────────────────┘
                            ▼
              Sentiment Metrics Comparison
   (accuracy · P/R · F1 · ROC-AUC · latency · memory · per-language)
                            │
                            ▼
                Best End-to-End Pipeline
```

> **Why IndicTrans2 / NLLB instead of IndicBERT v2 / MuRIL?** IndicBERT v2 and
> MuRIL are *encoder-only* masked-LM models — they have no decoder and cannot
> translate. IndicTrans2 is AI4Bharat's actual Indic→English translation
> model; NLLB-200 is the second pipeline (swap it for Google's
> `google/madlad400-3b-mt` in [config.py](config.py) if you specifically want
> Google's open translator — it is ~12 GB and much slower).

## Models

| Stage | Model | HF id |
|---|---|---|
| Translation A | IndicTrans2 distilled 200M | `ai4bharat/indictrans2-indic-en-dist-200M` |
| Translation B | NLLB-200 distilled 600M | `facebook/nllb-200-distilled-600M` |
| Sentiment A | Cardiff Twitter RoBERTa (3-class) | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Sentiment B | SieBERT RoBERTa Large (binary) | `siebert/sentiment-roberta-large-english` |

All four ship with real trained heads — **no fine-tuning is needed**.

## Setup

Python **3.11 – 3.12** recommended, ~5 GB free disk for model downloads. GPU
optional (CUDA used automatically, CPU fallback).

```bash
cd social_sentiment_benchmark
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

> **Pinned transformers version.** `requirements.txt` pins
> `transformers==4.40.2` deliberately — IndicTrans2's custom remote code
> imports `transformers.onnx` (removed in v5) and expects the legacy
> tuple-based `past_key_values` cache (replaced by the new Cache API in later
> 4.5x releases, causing a `past_key_values` shape error). Do not upgrade
> transformers unless AI4Bharat updates the model's remote code.

### HuggingFace authentication (required)

**IndicTrans2 is a gated model** — downloads fail without both of these steps:

1. **Request access** (once): open
   <https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M> while
   logged in and click **"Agree and access repository"**. A **403 Forbidden**
   means your token is valid but this approval hasn't been granted yet.
2. **Authenticate the machine**: create a read token at
   <https://huggingface.co/settings/tokens> and run

   ```bash
   huggingface-cli login
   ```

   A **401 Unauthorized** means this step is missing.

The other three models (NLLB, Cardiff, SieBERT) are public. If a translation
pipeline still can't load, the run logs an actionable error, excludes that
pipeline, and continues with the remaining one (marked in the decision trail).

Optional extras:

- `pip install unbabel-comet` — enables COMET (reference-based) and CometKiwi
  (reference-free; gated model, needs `huggingface-cli login`). Without it,
  COMET is skipped and the translation winner falls back to chrF → BLEU → latency.
- `IndicTransToolkit` — recommended pre/post-processing for IndicTrans2. When
  unavailable the pipeline falls back to plain tag prefixing (works, slightly
  lower quality; a warning is logged).

## Usage

```bash
python run.py --input data/dataset.csv

# Other options
python run.py --input data/dataset.csv --device cpu --batch-size 8
python run.py --input data/dataset.csv --translation-batch-size 4
python run.py --input data/dataset.csv --sample 100      # quick smoke run
python run.py --input data/dataset.csv --skip-comet
python run.py --help
```

## Input format

CSV with at least (extra columns are preserved):

| column | required | description |
|---|---|---|
| `post_text` | yes | the social-media post |
| `ground_truth_sentiment` | yes | `Positive` / `Negative` / `Neutral` |
| `reference_translation` | no | human English reference — enables reference-based BLEU/chrF/COMET |
| `id`, `platform`, `topic`, `sentiment_score` | no | carried through to outputs |

Without `reference_translation`, translation quality uses reference-free
COMET-QE (when installed) plus cross-pipeline agreement; if neither is
available the translation winner is decided by latency (documented in the
decision trail).

## Outputs (written to `outputs/`)

| file | contents |
|---|---|
| `translations.csv` | both pipelines' translations side by side, per-post latency, selected pipeline |
| `translation_metrics.csv` | BLEU / chrF / COMET / latency / memory per pipeline |
| `translation_quality.png`, `translation_latency.png` | translation-stage charts |
| `predictions_cardiff.csv`, `predictions_siebert.csv`, `predictions_combined.csv` | sentiment predictions, confidence, per-post latency |
| `comparison_metrics.csv`, `language_metrics.csv` | sentiment metric tables |
| `confusion_matrix_cardiff.png`, `confusion_matrix_siebert.png`, `roc_curve.png`, `accuracy_comparison.png`, `precision_comparison.png`, `recall_comparison.png`, `f1_comparison.png`, `metrics_comparison.png`, `inference_time.png`, `memory_usage.png`, `language_accuracy.png`, `language_f1.png`, `prediction_distribution.png` | sentiment-stage charts |
| `benchmark_summary.md`, `benchmark_summary.pdf` | full two-stage report incl. classification reports and language-wise confusion matrices |
| `BEST_PIPELINE.md` | the selected end-to-end pipeline with both decision trails |

**Winner rules.** Translation: COMET → chrF → BLEU → latency (unavailable
metrics are skipped). Sentiment: highest macro F1 → highest accuracy → lowest
average inference time.

## Notes & caveats

- **SieBERT is binary** (Positive/Negative). Neutral is approximated: a
  synthetic Neutral probability `2·min(p_neg, p_pos)` is inserted and rows are
  renormalized, so near-ties between the poles read as Neutral (tunable via
  `NEUTRAL_UNCERTAINTY_SCALE` in config). This is flagged in every report.
- English posts skip translation (zero translation latency); BLEU/chrF/COMET
  are computed only over posts that were actually translated.
- Posts whose detected language has no FLORES mapping (e.g. romanized
  code-mixed text misdetected as a European language) are passed through
  untranslated with a warning.
- Local checkpoints dropped into `models/cardiff/` or `models/siebert/`
  (any `save_pretrained` output) override the hub models automatically.
- Reproducibility: all RNGs seeded (`--seed`); CUDA-synchronized timing;
  models loaded/freed serially so memory figures reflect one model at a time.

## Project structure

```
social_sentiment_benchmark/
├── data/dataset.csv              # sample dataset (en / te / hi / code-mixed) with references
├── models/                       # optional local sentiment checkpoints
├── outputs/                      # all generated reports and plots
├── src/
│   ├── preprocessing.py          # cleaning + validation (emojis preserved)
│   ├── language_detector.py      # lingua → langdetect → 'unknown' fallback chain
│   ├── translation.py            # IndicTrans2 / NLLB / MADLAD translation pipelines
│   ├── translation_metrics.py    # BLEU, chrF, COMET, agreement, winner rule
│   ├── inference.py              # sentiment models, label mapping, binary→3-class
│   ├── metrics.py                # sentiment metrics, ROC, per-language, winner rule
│   ├── visualize.py              # all PNG charts (validated palette)
│   ├── benchmark.py              # two-stage benchmark orchestration + reports
│   ├── pipeline.py               # ★ finalized production pipeline (IndicTrans2 → Cardiff)
│   └── utils.py                  # logging, seeding, device & memory helpers
├── config.py                     # every path, model id and hyper-parameter
├── requirements.txt
├── predict.py                    # ★ production CLI (single post or CSV batch)
└── run.py                        # benchmark CLI
```
