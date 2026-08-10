# Urdu Translation Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep Roman and native Urdu on the existing Indic translation path, add a verified NLLB fallback, and prevent stalled model inference from holding the service lock indefinitely.

**Architecture:** Remove the overlapping Indic NLP distributions and require the TensorFlow-free IndicTrans2 fork. Make Urdu preprocessing failure explicit, validate primary output, and delegate ordinary primary failures to a preloaded NLLB translator. Wrap each model-generation call in a process-level watchdog so PM2 can recover from an uninterruptible model or CUDA stall without terminating healthy multi-batch requests.

**Tech Stack:** Python 3.12, FastAPI, PyTorch, Hugging Face Transformers 4.40.2, IndicTransToolkit/IndicTrans2, NLLB-200, unittest, PM2.

## Global Constraints

- Preserve the existing HTTP wire contract.
- Do not bypass Roman-Urdu transliteration or Urdu-to-English translation.
- Do not install TensorFlow or standalone `urduhack`.
- Only `indic-nlp-library-itt==0.1.1` may provide the `indicnlp` import.
- IndicTrans2 remains primary; NLLB is fallback.
- A hard timeout must release a permanently stuck process through PM2 restart.

---

### Task 1: Dependency and Urdu-preprocessor readiness

**Files:**
- Modify: `requirements.txt`
- Modify: `src/translation.py`
- Create: `tests/test_urdu_translation_resilience.py`

**Interfaces:**
- Produces: `TranslationPreprocessError`
- Produces: `Translator.urdu_preprocessing_ready: bool`

- [ ] **Step 1: Write failing dependency and preprocessing tests**

Assert that requirements contain only `indic-nlp-library-itt==0.1.1`; create a `Translator` without loading models and inject a processor that raises `ModuleNotFoundError("urduhack")`. Assert `_prepare_batch(..., "urd_Arab")` raises `TranslationPreprocessError`, while another language retains the existing plain-tag behavior.

- [ ] **Step 2: Run tests and verify expected failures**

Run: `python3 -m unittest tests.test_urdu_translation_resilience -v`

Expected: dependency exclusivity and typed Urdu failure tests fail.

- [ ] **Step 3: Implement the dependency and typed failure**

Replace `indic-nlp-library>=0.92` with `indic-nlp-library-itt==0.1.1`. Add `TranslationPreprocessError`, prohibit Urdu plain-tag fallback, and probe a short Urdu sentence through `IndicProcessor.preprocess_batch()` during translator construction.

- [ ] **Step 4: Re-run the focused tests**

Run: `python3 -m unittest tests.test_urdu_translation_resilience -v`

Expected: Task 1 tests pass.

### Task 2: Validated NLLB fallback

**Files:**
- Modify: `config.py`
- Modify: `src/translation.py`
- Modify: `src/pipeline.py`
- Modify: `tests/test_urdu_translation_resilience.py`

**Interfaces:**
- Produces: `translation_is_usable(source: str, translated: str) -> bool`
- Produces: `SentimentPipeline._translate_with_fallback(texts, languages, batch_size) -> TranslationResult`

- [ ] **Step 1: Write failing quality and fallback tests**

Test rejection of empty output, unchanged Arabic-script output, repeated n-gram output, and extreme output expansion. Inject fake primary and fallback translators into a model-free pipeline instance; assert a primary exception or unusable output invokes fallback and a valid primary output does not.

- [ ] **Step 2: Run tests and verify expected failures**

Run: `python3 -m unittest tests.test_urdu_translation_resilience -v`

Expected: quality helper and fallback method are absent.

- [ ] **Step 3: Implement minimal fallback behavior**

Add configurable output expansion/repetition limits. Preload `TRANSLATION_MODELS["nllb"]` as `fallback_translator`, route primary exceptions or unusable translated records through it, and preserve result ordering and timing fields.

- [ ] **Step 4: Re-run focused tests**

Run: `python3 -m unittest tests.test_urdu_translation_resilience -v`

Expected: Task 1 and Task 2 tests pass.

### Task 3: Soft generation bound and hard watchdog

**Files:**
- Modify: `config.py`
- Modify: `src/translation.py`
- Create: `src/inference_watchdog.py`
- Modify: `api_server.py`
- Modify: `tests/test_urdu_translation_resilience.py`

**Interfaces:**
- Produces: `InferenceWatchdog(timeout_s: float, abort: Callable[[], None])`
- Consumes: `SENTIMENT_GENERATION_MAX_TIME_S`
- Consumes: `SENTIMENT_INFERENCE_HARD_TIMEOUT_S`

- [ ] **Step 1: Write failing watchdog tests**

Use an injected event-setting abort callback. Assert an armed watchdog fires after its deadline and a cancelled watchdog never fires. Assert generated kwargs include `max_time`.

- [ ] **Step 2: Run tests and verify expected failures**

Run: `python3 -m unittest tests.test_urdu_translation_resilience -v`

Expected: watchdog module and generation time option are absent.

- [ ] **Step 3: Implement soft and hard bounds**

Pass `max_time` to `model.generate()`. Arm and cancel a synchronized watchdog around each generation call. The production abort callback logs, flushes handlers, and calls `os._exit(70)` so PM2 restarts the process.

- [ ] **Step 4: Re-run focused tests**

Run: `python3 -m unittest tests.test_urdu_translation_resilience -v`

Expected: all focused tests pass without waiting for model downloads.

### Task 4: Health reporting and deployment documentation

**Files:**
- Modify: `src/pipeline.py`
- Modify: `api_server.py`
- Modify: `README.md`
- Modify: `tests/test_urdu_translation_resilience.py`

**Interfaces:**
- Extends: `GET /health` additive stage/limit fields only

- [ ] **Step 1: Write failing health-status test**

Construct model-free fake translator objects and assert stage status includes primary translator, fallback translator, Urdu readiness, generation soft limit, and inference hard limit.

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m unittest tests.test_urdu_translation_resilience -v`

Expected: new status fields are missing.

- [ ] **Step 3: Add status and operational documentation**

Expose additive health keys and document dependency exclusivity, new environment variables, PM2 restart requirement, and Urdu smoke-test commands.

- [ ] **Step 4: Run all lightweight verification**

Run:
- `python3 -m unittest discover -s tests -v`
- `python3 -m py_compile api_server.py config.py src/*.py`
- `python3 verify_intelligence_contract.py`

Expected: all commands exit zero.

### Task 5: Commit, push, deploy, and live verification

**Files:**
- Deployment target: `/home/cat-hyd-work-station/BCSS/social_media_sentiment_analysis`

- [ ] **Step 1: Review complete diff and repository status**

Run `git status --short`, `git diff --check`, and `git diff`.

- [ ] **Step 2: Commit and push `main`**

Commit the tested source, tests, specification, and plan with a message describing Urdu translation resilience, then push `origin main`.

- [ ] **Step 3: Deploy from Git**

On `iccc-ws`, verify the checkout is clean, pull `main`, rebuild `.venv` from `requirements.txt` so the overlapping original package is absent, and restart only `sentiment-api` through PM2.

- [ ] **Step 4: Verify the live service**

Confirm `/health`, PM2 stability, native Urdu translation, the Taraweeh Roman-Urdu translation, English bypass, no `urduhack` warning, no decode-budget warning, and successful backend calls into `/analyze/intelligence`.

- [ ] **Step 5: Audit post-restart logs**

Inspect service and backend logs for exceptions, fallback use, watchdog restarts, queue growth, and completed analysis results before reporting completion.
