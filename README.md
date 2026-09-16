# Multilingual Social Sentiment — Standalone Service

An **independent microservice**. Not part of any other application's
codebase or deployment — it has its own repository, its own dependencies, its
own git history, and is consumed purely over HTTP. Any caller (SOCKEYE or
otherwise) talks to it through `POST /analyze` / `GET /health`; nothing about
this service assumes a specific consumer.

**Finalized architecture (selected by benchmark, extended with romanized-Indic support):**

```
   Social Media Post
          │
          ▼
  Text Preprocessing + Language Detection (lingua/langdetect)
          │
          ▼
  Latin-script text? ── yes ──▶ IndicLID (AI4Bharat fastText + IndicBERT rerank)
          │                     refines "en" -> actual Indic language for
          │                     romanized text (Hinglish, Roman-Telugu, ...)
          ▼ no
          │◀────────────────────────────┘
          ▼
  Romanized Indic? ── yes ──▶ IndicXlit (AI4Bharat) — Roman -> native script
          │
          ▼ no / already native
          │◀────────────────────────────┘
          ▼
  Genuine English? ── yes ──▶ bypass translation
          │
          ▼ no
  IndicTrans2 (AI4Bharat)
  src=detected/refined language, tgt=English
          │
          ▼
  Cardiff Twitter RoBERTa
 (cardiffnlp/twitter-roberta-base-sentiment-latest)
          │
          ▼
 Positive / Neutral / Negative
 + language, English text, was_translated, was_transliterated, confidence, timings
```

Only genuine English is bypassed from translation. Romanized Indic text
(Latin script, Indic language — e.g. "nenu bాgunna" in Roman-Telugu) is never
bypassed: it is first transliterated to native script by IndicXlit, then
still routes through IndicTrans2 like any other Indic-language post. The
general language detector (lingua/langdetect) has no romanized-language
profiles and misreads this text as English — IndicLID exists specifically to
catch and correct that before the bypass decision is made.

**Observed production split (latest analyzed window, percentages only):**

```
                         Posts
                           │
                           ▼
                 Main Deterministic Pipeline
                           │
             ┌─────────────┴─────────────┐
             │                           │
          99.892%                      0.107%
             │                           │
             ▼                           ▼
       Sentiment ready             vLLM fallback
             │                           │
             │                           ▼
             │                 Translation / Sentiment
             │                           │
             └──────────────┬────────────┘
                            │
                            ▼
                    vLLM Intelligence
                            │
                            ▼
                      Extra Metrics
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
| `No module named 'urduhack'` | Both overlapping Indic NLP distributions are installed, or the original one replaced the TensorFlow-free fork | Recreate the virtual environment from `requirements.txt`; only `indic-nlp-library-itt==0.1.1` may provide `indicnlp` |
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

## Running as a service

This is the primary way the pipeline is meant to be consumed — as a standalone
HTTP service, independent of and decoupled from any caller's own codebase or
deploy process:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8003
```

```bash
curl -X POST http://localhost:8003/analyze \
  -H "Content-Type: application/json" \
  -d '{"texts": ["ప్రభుత్వం ప్రకటించిన కొత్త పథకం చాలా బాగుంది"]}'
```

Urdu deployment smoke tests must cover both scripts:

```bash
curl -X POST http://localhost:8003/analyze \
  -H "Content-Type: application/json" \
  -d '{"texts":["جس کسی کی بھی تراویح کی نماز نہیں ہوئی ہے"]}'

curl -X POST http://localhost:8003/analyze \
  -H "Content-Type: application/json" \
  -d '{"texts":["Jis kissi ki bhi Taraweeh ki Namaz Nahi hui hai"]}'
```

`GET /health` reports `{"healthy": true, "device": "cuda"|"cpu"}` once the models finish loading, plus two additive keys: `stages` (which optional stages actually came up) and `limits` (the request ceilings below). `HF_TOKEN` (for the gated IndicTrans2 model) is read from `.env`; `SENTIMENT_DEVICE` / `SENTIMENT_BATCH_SIZE` / `SENTIMENT_TRANSLATION_BATCH_SIZE` / `SENTIMENT_MAX_LENGTH` optionally override `config.py`'s defaults the same way.

### Send posts in batches

`/analyze` takes a **list**, and the whole pipeline is batched end to end —
translation groups the batch by source language and decodes it in one
`generate()` call, then sentiment classifies in one forward pass. One request
carrying 25 posts therefore costs far less than 25 requests carrying one post
each, which pay the per-call setup cost 25 times over.

A caller sending one post per request is the single largest throughput loss in
the current deployment, and it is fixed on the caller's side — the service has
always accepted batches.

### Request limits

Oversized requests are rejected with `413` and a message naming the limit.
Defaults are far above ordinary traffic and are env-overridable:

| Variable | Default | Limit |
| --- | --- | --- |
| `SENTIMENT_API_MAX_TEXTS` | 256 | texts per request |
| `SENTIMENT_API_MAX_TEXT_CHARS` | 5000 | characters per text |
| `SENTIMENT_API_MAX_TOTAL_CHARS` | 200000 | characters per request |

### Latency and throughput tuning

| Variable | Default | Effect |
| --- | --- | --- |
| `SENTIMENT_TRANSLATION_CACHE_SIZE` | 2048 | Memoizes translations on `(language, exact source text)`. A post retried after a client-side timeout is served from cache instead of being translated again. 0 disables. |
| `SENTIMENT_TRANSLATION_LENGTH_RATIO` | 2.5 | Caps generation at `ratio x longest_source_tokens + margin` new tokens instead of a flat 256, so a degenerate decode can't burn the full budget on a short post. Every clamp is logged with the source length. 0 restores the flat budget. |
| `SENTIMENT_TRANSLATION_LENGTH_MARGIN` | 32 | Constant added to the above. |
| `SENTIMENT_GENERATION_MAX_TIME_S` | 30 | Cooperative Hugging Face generation deadline in seconds. IndicTrans2 failures or unusable output are retried through NLLB. |
| `SENTIMENT_INFERENCE_HARD_TIMEOUT_S` | 120 | Hard deadline for each model-generation call. The process exits with code 70 so PM2 can restart it if a model or CUDA call cannot return to Python. Set 0 only for non-production debugging. |
| `SENTIMENT_TRANSLATION_OUTPUT_MAX_RATIO` | 4.0 | Rejects implausibly expanded primary translations and sends them to NLLB. |
| `SENTIMENT_TRANSLATION_REPEAT_TOKEN_RATIO` | 0.5 | Rejects repetitive decode loops when one token dominates an output of at least eight tokens. |
| `SENTIMENT_PIPELINE_FALLBACK_ENABLED` | true | Sends only recoverable deterministic translation/sentiment failures to vLLM after releasing the GPU inference lock. |
| `SENTIMENT_PIPELINE_FALLBACK_TIMEOUT_S` | 60 | Timeout for the schema-constrained vLLM fallback call. Provider failure returns HTTP 503 for retry. |
| `SENTIMENT_PIPELINE_FALLBACK_MAX_WORKERS` | 2 | Service-wide vLLM fallback workers consuming the FIFO queue. |
| `SENTIMENT_PIPELINE_FALLBACK_QUEUE_CAPACITY` | 64 | Maximum pending fallback jobs across all requests. |
| `SENTIMENT_PIPELINE_FALLBACK_QUEUE_TIMEOUT_S` | 2 | Maximum wait to enqueue fallback work before returning HTTP 503. |
| `SENTIMENT_TORCH_NUM_THREADS` | 0 (torch default) | CPU intra-op threads. Worth setting explicitly in a CPU deployment: torch sizes its default from the *visible* core count, so inside a cgroup-limited container it oversubscribes and contends. The effective value is logged at startup either way. |

Inference is serialized on a lock — the pipeline is one set of torch modules
and cannot be driven from two threads at once. Requests queue, and any wait
over a second is logged, so a slow response can be attributed to saturation
rather than to the model. To serve more concurrency, run more replicas.
The production process manager must have automatic restart enabled: the hard
watchdog deliberately terminates a process that cannot release the inference
lock.

### vLLM pipeline fallback

The deterministic stack remains authoritative. If IndicTrans2 and NLLB cannot
produce usable English, or the sentiment classifier fails, the service isolates
the failed post and calls vLLM only after releasing the GPU inference lock.
Successful neighbors in the same batch are not sent to vLLM.

Fallback work enters one bounded, process-wide FIFO queue. A fixed worker pool
limits vLLM concurrency across all requests; the queue never grows without
bound. Saturated submissions wait for the configured enqueue timeout and then
return HTTP 503. `/health.pipeline_fallback.queue` reports active, queued,
accepted, rejected, cancelled, and completed counts.

vLLM is constrained to return exactly `english_text`, `sentiment`, and
`confidence`. The service rejects additional/missing fields, sentiment outside
`Positive | Neutral | Negative`, invalid confidence, and unusable translations.
It constructs the normal result metadata and timing fields itself, so
`/analyze` and `/analyze/intelligence` retain their existing response schemas.
The fallback does not perform policy mapping, risk scoring, alerting, entity
extraction, or recommendations. If it cannot return a valid result, the request
returns HTTP 503 instead of inventing defaults.

A caller integrates purely through this HTTP contract — one example is
SOCKEYE's backend, which can route sentiment requests here via
`SENTIMENT_ANALYSIS=CUSTOM` + `CUSTOM_SENTIMENT_URL` pointed at this service's
`/analyze` endpoint — but this repo has no dependency on SOCKEYE or any other
consumer; it can be deployed and versioned on its own.

Only these two models are loaded in production. The benchmark harness below is
retained for reference — it is how this architecture was selected — and can be
re-run at any time to re-validate the choice.

---

# Intelligence layer (stage 3)

An **additive** second stage that reasons over the pipeline's structured output.
It does not replace, bypass or duplicate any pipeline stage.

```
 [ language detection -> romanized-Indic LID -> IndicXlit -> IndicTrans2
   -> Cardiff RoBERTa ]                                    <- unchanged
                    │
                    ▼   language, english_text, sentiment, confidence,
                        was_translated, was_transliterated  (trusted facts)
                    │
                    ▼
              vLLM (guard-railed)
                    │
                    ▼
   intent | category | risk_score | reasoning | summary | recommended_action
```

**The deterministic pipeline owns** language detection, transliteration,
translation, sentiment and confidence. **vLLM never recomputes any of them** —
the system prompt in [src/intelligence.py](src/intelligence.py) forbids it
explicitly, and those values are passed in as established facts.

## Endpoint

`/analyze` is untouched — same request schema, same response keys, same latency.
Intelligence is a separate endpoint:

```bash
curl -X POST http://localhost:8003/analyze/intelligence \
  -H "Content-Type: application/json" \
  -d '{"texts": ["ప్రభుత్వం ప్రకటించిన కొత్త పథకం చాలా బాగుంది"]}'
```

Optional request fields (all ignored by `/analyze`):

| Field | Default | Purpose |
| --- | --- | --- |
| `policy_pack` | built-in taxonomy | Caller category allowlist (`id` + optional `definition`/`severity`/`keywords`), plus `unknown_label` |
| `intent_mode` | `enum` | `enum` = fixed intent labels; `free` = 2–8 word phrase + `intent_label` enum |
| `timeout_s` | `VLLM_TIMEOUT_S` | Per-request vLLM timeout override (clamped to 5–600) |
| `tenant_name` | _(none)_ | Caller/tenant label. See [Tenant context](#tenant-context) below. |

Example with a SOCKEYE-style pack:

```bash
curl -X POST http://localhost:8003/analyze/intelligence \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["..."],
    "intent_mode": "free",
    "policy_pack": {
      "name": "sockeye-policy-mapping",
      "unknown_label": "Normal",
      "categories": [
        {"id": "Hate_Speech", "definition": "Content attacking a protected group."},
        {"id": "Normal", "definition": "Harmless or neutral content."}
      ]
    }
  }'
```

Each result carries every `/analyze` key plus an `intelligence` object:

```json
{
  "post_text": "...", "language": "te", "english_text": "...",
  "sentiment": "Negative", "confidence": 0.97,
  "was_translated": true, "was_transliterated": false,
  "intelligence": {
    "category": "Protest", "intent": "Protest Mobilization",
    "intent_label": "Protest Mobilization",
    "risk_score": 72,
    "reasoning": "...", "summary": "...", "recommended_action": "Monitor",
    "evidence_confidence": "high",
    "stance": "Support", "stance_confidence": 0.83,
    "signals": ["code_mixed"], "source": "provider",
    "model": "Qwen3-14B-AWQ", "latency_ms": 1840.2,
    "policy_pack_fingerprint": "sha256:…",
    "schema_enforced": true
  }
}
```

`source` is one of `provider` | `triage` | `error`. SOCKEYE clients must treat
`error` as failure (equivalent to a null LLM result), not as a valid verdict.

### Stance (event/issue position)

`stance` and `stance_confidence` are additive `intelligence` fields, a
**separate axis from `sentiment`**:

- `sentiment` — the emotional/polarity tone of the text (unchanged, still
  produced by the deterministic pipeline stage — never by the LLM).
- `stance` — the position the author takes toward the specific subject, event
  or issue the text itself names or clearly implies. One of `Support` |
  `Oppose` | `Neutral` | `Unclear`.
- `stance_confidence` — `0.0`–`1.0`, the model's confidence in `stance`.
  Always a finite number in range; malformed or missing model output is
  clamped to `0.0` rather than propagated, and NaN/Infinity are rejected the
  same way `risk_score` is.

Stance is **not** derived from sentiment — praising how well a protest was
organized (positive sentiment) is not support for the protest's cause, and a
grim factual report on a policy (negative sentiment) is not opposition to it.
When the text names no clear subject, or the post is too short/ambiguous,
`stance` is `Unclear` with a low `stance_confidence` rather than a guess.
Posts resolved by triage or a failed provider call (see Guarantees below)
carry `stance: "Unclear"`, `stance_confidence: 0.0`, consistent with their
other placeholder fields.

### Tenant context

`tenant_name` (request field, optional) labels which caller/tenant an
`/analyze/intelligence` request is for. It is passed to the model as
background context only:

- **Not authorization.** It grants no access and is never checked against
  any allowlist.
- **Not a routing or database key.** The service loads one shared model set
  and stays fully stateless; `tenant_name` never selects a database,
  connection, cache namespace, or code path.
- **One value per request.** The current batch contract (`texts: [...]`) has
  no per-item tenant identity, so `tenant_name` applies to every text in that
  request. Callers needing per-post tenants must split them into separate
  requests.
- **Validated, not trusted as instructions.** Bounded to 200 characters
  (`SENTIMENT_API_MAX_TENANT_NAME_CHARS`), rejected with `422` if longer or if
  it contains control characters. The system prompt explicitly tells the
  model to treat it as inert metadata, never as a directive or as the subject
  of the stance.
- **Backward compatible.** Omitting it (or calling `/analyze`, which ignores
  it) reproduces prior behaviour exactly.

```bash
curl -X POST http://localhost:8003/analyze/intelligence \
  -H "Content-Type: application/json" \
  -d '{"texts": ["..."], "tenant_name": "Tenant 1"}'
```

Both output sets belong in the stored analysis record. `predict.py --intelligence`
writes them as columns alongside the sentiment columns.

Contract checks (no live vLLM required):

```bash
python verify_intelligence_contract.py
python -m unittest tests.test_api_tenant_stance
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `VLLM_BASE_URL` | `http://100.49.109.96/v1` | OpenAI-compatible vLLM base URL (include `/v1`) |
| `VLLM_MODEL` | `Qwen3-14B-AWQ` | Served model id |
| `VLLM_API_KEY` | _(empty)_ | Bearer token when the gateway requires auth |
| `VLLM_TIMEOUT_S` | 120 | Per-request timeout |
| `VLLM_RETRIES` | 0 | In-process vLLM retries (0 = caller owns retry) |
| `VLLM_CONCURRENCY` | 2 | Parallel calls per batch (capped by gate size) |
| `VLLM_GATE_SIZE` | 2 | Process-wide in-flight cap |
| `VLLM_CIRCUIT_FAILURES` | 8 | Consecutive failures before the circuit opens |
| `VLLM_CIRCUIT_COOLDOWN_S` | 30 | How long the circuit stays open |
| `VLLM_TEMPERATURE` | 0 | 0 for reproducible assessments |
| `VLLM_MAX_TOKENS` | 256 | Max generation tokens |
| `VLLM_JSON_SCHEMA` | 1 | Prefer `response_format=json_schema`; auto-downgrades to `json_object` if rejected |
| `SENTIMENT_INTELLIGENCE_PROVIDER` | `vllm` | Selects the provider from the registry |
| `SENTIMENT_API_MAX_TENANT_NAME_CHARS` | 200 | Max length of the optional `tenant_name` request field (422 above this) |

## Guarantees

- **Never fatal.** Provider unreachable, timeout, malformed JSON, or a label
  outside the taxonomy all degrade to a well-formed record with `risk_score` 0,
  `recommended_action` "Human Review" and `source: "error"`. The sentiment
  result is always returned intact. `/analyze` cannot be affected at all — the
  layer is imported defensively at startup and its absence only makes
  `/analyze/intelligence` return 503.
- **Taxonomy is enforced in code, not just in the prompt.** Out-of-vocabulary
  labels are clamped to `Unknown` / `Human Review`, and `risk_score` is clamped
  to 0–100, so consumers can rely on the enums.
- **Runs outside the inference lock.** Intelligence is network-bound work
  against another host; it never blocks sentiment inference.

## Edge cases

Posts with no analyzable content — empty, emoji-only, URL-only, media
placeholders — are resolved deterministically without a model call. Everything
else goes to the model with precomputed **signals** attached, so handling is
consistent and auditable rather than left for the model to rediscover:

`very_short` · `truncated` · `low_sentiment_confidence` · `code_mixed` ·
`romanized_indic_transliterated` · `translation_unavailable` ·
`language_undetermined` · `quoted_or_forwarded` · `possible_spam` ·
`link_heavy` · `possible_ocr_noise` · `duplicate_post`

Duplicate posts within a batch are analyzed once and the record reused. Very
long posts are truncated head-and-tail so both the opening framing and any
closing call to action survive.

## Adding another intelligence backend

Subclass `IntelligenceProvider` (implement `generate` and `describe`) and
register it in `PROVIDERS`. Validation, taxonomy clamping, triage, batching and
fallbacks are handled centrally, and nothing in the sentiment pipeline changes.

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
- English posts are translated too (eng_Latn -> eng_Latn, no bypass); BLEU/chrF/COMET
  are computed only over posts whose detected language had a FLORES mapping.
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
│   ├── script_detection.py       # Unicode-range script detection (no ML)
│   ├── lid_roman.py              # IndicLID (fastText + IndicBERT rerank) — romanized-Indic vs genuine English
│   ├── transliteration.py        # IndicXlit (AI4Bharat, via fairseq) — Roman -> native script
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
├── api_server.py                 # ★ production HTTP service (FastAPI) — see above
└── run.py                        # benchmark CLI
```
