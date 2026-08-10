# Urdu Translation Resilience Design

## Goal

Translate native and romanized Urdu through the existing deterministic pipeline without silently bypassing preprocessing, while preventing one pathological translation from blocking every sentiment request.

## Current failure

The live virtual environment contains both `indic-nlp-library` and `indic-nlp-library-itt`. Both distributions install the same `indicnlp` modules, and the active module is the original package. Its Urdu normalizer imports the absent TensorFlow-based `urduhack` package. `Translator._prepare_batch()` catches that error and sends improperly prepared `urd_Arab` input to IndicTrans2, where generation can stall while the global inference lock remains held.

## Architecture

The normal path remains:

`Roman Urdu -> IndicLID -> IndicXlit -> Urdu script -> IndicTransToolkit -> IndicTrans2 -> English -> Cardiff sentiment -> intelligence`

The dependency set will contain only `indic-nlp-library-itt==0.1.1`, the IndicTrans2-specific fork maintained by the IndicTransToolkit maintainer that embeds Urdu normalization without TensorFlow. Translator startup will probe Urdu preprocessing and fail readiness if it is unavailable.

IndicTrans2 remains authoritative. An NLLB translator, using the existing `nllb` model configuration, provides an in-process fallback only when IndicTrans2 preprocessing or generation raises an ordinary exception. Urdu preprocessing errors must never enter the plain-tag fallback.

Generation receives a configurable soft time budget. Each model-generation call gets a separate hard process deadline because a Python exception or thread timeout cannot interrupt a stuck CUDA kernel. Per-call deadlines prevent a valid large request with several healthy batches from being terminated for exceeding one shared request timer. If a model call exceeds the deadline, the watchdog terminates the sentiment process and PM2 restarts it, releasing the global lock.

## Error handling and observability

- Native and transliterated Urdu use `urd_Arab -> eng_Latn`.
- Urdu preprocessing failure raises a typed error and invokes NLLB.
- Empty, unchanged, excessively repetitive, or implausibly long primary translations invoke NLLB.
- If both translators fail, the request returns `503`; it does not report untranslated text as successfully translated.
- `/health` reports Urdu preprocessing readiness, primary translator, fallback translator, and watchdog deadline.
- Logs identify primary failures, fallback use, and watchdog-triggered restarts without logging complete post bodies.

## Validation

Automated tests cover dependency exclusivity, Urdu preprocessing failure, NLLB fallback selection, output-quality rejection, and watchdog arm/cancel behavior. Deployment verification includes native Urdu, the exact Taraweeh Roman-Urdu post, English bypass, health output, process stability, and backend-to-sentiment traffic.

## Constraints

- Preserve existing HTTP request and response shapes.
- Preserve English bypass and all non-Urdu language behavior.
- Do not install TensorFlow or standalone `urduhack`.
- Pin the TensorFlow-free Urdu preprocessing distribution.
- Keep the current single inference lock; the watchdog bounds its failure impact.
