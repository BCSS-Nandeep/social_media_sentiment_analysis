# Ollama Pipeline Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve recoverable per-post translation or sentiment failures through Ollama while returning the exact existing `PipelineResult` schema and category set.

**Architecture:** The deterministic GPU pipeline returns ordered `PipelineResult | PipelineFailure` outcomes and isolates batch errors per post. The API releases the inference lock before a dedicated Ollama fallback resolves only failures. The resolver validates a three-field provider response and constructs `PipelineResult` itself; `/analyze/intelligence` then continues through its unchanged intelligence stage.

**Tech Stack:** Python 3.12, FastAPI, PyTorch, requests, Ollama `/api/chat`, unittest, PM2.

## Global Constraints

- Existing deterministic pipeline always runs first.
- Successful deterministic records never call Ollama.
- Ollama may return only `english_text`, `sentiment`, and `confidence`.
- Sentiment must be exactly `Positive`, `Neutral`, or `Negative`.
- Public response schemas and metadata fields remain unchanged.
- Ollama network calls must run outside the GPU inference lock.
- Provider failure or malformed output returns `503`; no fabricated defaults.

---

### Task 1: Ordered deterministic outcomes

**Files:**
- Modify: `src/pipeline.py`
- Modify: `tests/test_urdu_translation_resilience.py`

**Interfaces:**
- Produces: `PipelineFailure(post_text, language, was_transliterated, deterministic_time_ms, reason)`
- Produces: `SentimentPipeline.predict_batch_outcomes(...) -> list[PipelineResult | PipelineFailure]`
- Preserves: `SentimentPipeline.predict_batch(...) -> list[PipelineResult]`

- [ ] **Step 1: Write failing outcome-isolation tests**

Inject fake detector, transliterator, translators, and classifier behavior into a model-free pipeline. Make one item fail deterministic translation and assert `predict_batch_outcomes` returns a failure only at that index while neighboring items remain `PipelineResult` objects in original order.

- [ ] **Step 2: Run the focused test**

Run: `python3 -m unittest tests.test_urdu_translation_resilience.PipelineOutcomeTests -v`

Expected: fail because `PipelineFailure` and `predict_batch_outcomes` do not exist.

- [ ] **Step 3: Implement failure records and per-item isolation**

Refactor the locked path into detection/transliteration, translation, sentiment, and assembly helpers. Attempt each model stage in batch first; after a recoverable batch exception, retry that stage one post at a time and create failure records for only the failed indices. Keep `predict_batch` compatible by raising `TranslationOutputError` when any failure remains.

- [ ] **Step 4: Verify focused and existing tests**

Run: `python3 -m unittest tests.test_urdu_translation_resilience -v`

Expected: all tests pass.

### Task 2: Strict Ollama fallback client

**Files:**
- Create: `src/pipeline_fallback.py`
- Modify: `config.py`
- Modify: `tests/test_urdu_translation_resilience.py`

**Interfaces:**
- Produces: `PipelineFallbackError`
- Produces: `OllamaPipelineFallback.resolve(failure: PipelineFailure) -> PipelineResult`
- Consumes: existing `OllamaProvider.generate(...)`

- [ ] **Step 1: Write failing schema-normalization tests**

Inject a fake provider. Assert a valid three-field response becomes a `PipelineResult` whose `to_dict()` keys exactly match a deterministic result. Add separate tests rejecting missing fields, additional fields, invalid sentiment, booleans/non-finite/out-of-range confidence, empty or unchanged non-English translation, and provider exceptions.

- [ ] **Step 2: Run focused tests**

Run: `python3 -m unittest tests.test_urdu_translation_resilience.OllamaFallbackTests -v`

Expected: fail because the fallback module does not exist.

- [ ] **Step 3: Implement the strict client**

Define a JSON schema with `additionalProperties: false`, exact sentiment enum, and bounded confidence. Use a narrow system prompt and the existing `OllamaProvider`; perform independent strict Python validation after parsing. Construct every metadata and timing field in service code.

Add:

```python
PIPELINE_FALLBACK_ENABLED = env bool, default true
PIPELINE_FALLBACK_TIMEOUT_S = env float, default 60
PIPELINE_FALLBACK_MAX_WORKERS = env int, default 2
```

- [ ] **Step 4: Run focused tests**

Run: `python3 -m unittest tests.test_urdu_translation_resilience.OllamaFallbackTests -v`

Expected: all strict normalization tests pass.

### Task 3: Resolve failures outside the inference lock

**Files:**
- Modify: `api_server.py`
- Modify: `tests/test_urdu_translation_resilience.py`

**Interfaces:**
- Produces: `_resolve_pipeline_outcomes(outcomes, request_id) -> list[PipelineResult]`
- Extends: lifespan globals with `_pipeline_fallback`

- [ ] **Step 1: Write failing API orchestration tests**

Patch the deterministic pipeline to return mixed outcomes and inject a fake fallback. Assert `_inference_lock.locked()` is false inside every fallback call, only failure records invoke it, results preserve order, and both endpoints return the existing keys.

- [ ] **Step 2: Run focused tests**

Run: `python3 -m unittest tests.test_urdu_translation_resilience.ApiFallbackOrchestrationTests -v`

Expected: fail because mixed outcome resolution is absent.

- [ ] **Step 3: Implement lifecycle and resolution**

Construct `OllamaPipelineFallback` during startup when enabled, close it during shutdown, call `predict_batch_outcomes` under `_inference_lock`, then resolve failures through a bounded `ThreadPoolExecutor` after lock release. Deduplicate exact `(post_text, language)` failures within a request and restore original ordering.

Map `PipelineFallbackError` to HTTP `503` in both endpoints. Do not catch or alter the later intelligence analyzer output.

- [ ] **Step 4: Verify API tests**

Run: `python3 -m unittest tests.test_urdu_translation_resilience -v`

Expected: all existing and fallback tests pass.

### Task 4: Health, documentation, and full verification

**Files:**
- Modify: `api_server.py`
- Modify: `README.md`
- Modify: `tests/test_urdu_translation_resilience.py`

**Interfaces:**
- Extends: `/health.pipeline_fallback` with additive configuration/readiness data

- [ ] **Step 1: Write failing health test**

Assert health reports fallback enablement, provider, model, timeout, worker limit, and reachability without changing existing health fields.

- [ ] **Step 2: Implement health and documentation**

Document triggering rules, exact provider schema, environment variables, `503` behavior, lock separation, and operational log messages.

- [ ] **Step 3: Run full verification**

Run:

```bash
python3 -m unittest discover -s tests -v
python3 verify_intelligence_contract.py
git diff --check
```

Expected: all tests and 8 contract checks pass; diff check exits zero.

### Task 5: Review, push, deploy, and verify

**Files:**
- Deployment target: `/home/cat-hyd-work-station/BCSS/social_media_sentiment_analysis`

- [ ] **Step 1: Review complete diff**

Run `git status --short`, `git diff --check`, and `git diff`; confirm no secrets or unrelated changes.

- [ ] **Step 2: Commit and push `main`**

Commit source, tests, documentation, specification, and plan; push `origin main`.

- [ ] **Step 3: Deploy and test live**

Fast-forward the clean live checkout, run the test suite, restart only `sentiment-api`, and wait for `/health`.

- [ ] **Step 4: Verify live contracts**

Confirm a normal deterministic post does not use fallback. Exercise a controlled recoverable failure using an injected automated test rather than a production-only API flag. Verify the configured Ollama provider with a representative fallback prompt, then verify `/analyze` and `/analyze/intelligence` retain their exact schemas.

- [ ] **Step 5: Audit runtime logs**

Confirm no fallback occurs for successful traffic, fallback calls happen outside the inference lock, malformed responses produce `503`, and ordinary backend calls continue returning `200`.
