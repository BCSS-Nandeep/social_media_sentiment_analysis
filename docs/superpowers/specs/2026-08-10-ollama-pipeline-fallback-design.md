# Ollama Pipeline Fallback Design

## Goal

Use the existing deterministic translation and sentiment pipeline first. If an individual post cannot produce a valid translated-and-classified result, send only that post to Ollama and normalize the response into the existing `PipelineResult` contract so no caller or downstream component changes.

## Scope

The fallback handles recoverable per-post translation or sentiment failures. It does not replace successful deterministic results, add response fields, change category names, or perform the later intelligence/policy-mapping task. Native Urdu, Roman Urdu/Hindustani, Hinglish, and every other currently supported language retain their existing primary path.

An uninterruptible CUDA/kernel stall remains process-fatal by design: the watchdog exits and PM2 restarts the service. The backend retry then starts a clean attempt. Python cannot safely recover the active process and call Ollama after such a stall.

## Architecture

The deterministic pipeline will produce one outcome per input:

- `PipelineResult` for a valid translation and Cardiff sentiment result.
- `PipelineFailure` for a recoverable post-specific error, carrying the original text, detected language, transliteration state, elapsed deterministic time, and a sanitized failure reason.

Batch-level deterministic errors will be isolated by retrying the affected batch one post at a time. This prevents one malformed or unsupported post from sending successful neighbors to Ollama.

`api_server._run_pipeline()` continues to own the GPU inference lock. It returns deterministic results and failure records, then releases the lock. A separate `OllamaPipelineFallback` resolves failure records outside the lock, preserving input order. Network waits therefore cannot block healthy GPU inference.

## Ollama contract

Ollama receives the original post, detected language, and a narrowly scoped instruction:

1. Translate the post faithfully into English.
2. Classify sentiment using exactly `Positive`, `Neutral`, or `Negative`.
3. Return confidence from `0.0` through `1.0`.

The provider must return strict JSON with exactly:

```json
{
  "english_text": "string",
  "sentiment": "Positive | Neutral | Negative",
  "confidence": 0.0
}
```

Unknown and additional fields are rejected. Ollama does not choose policy categories, risk levels, entities, summaries, alerts, or intelligence fields.

The service, not Ollama, constructs the existing `PipelineResult`:

- `post_text`: unchanged original input.
- `language`: deterministic detector output.
- `english_text`: validated Ollama translation.
- `was_translated`: true for non-English input when output differs from input.
- `was_transliterated`: preserved from the deterministic attempt.
- `sentiment`: validated existing enum.
- `confidence`: validated finite number in `[0, 1]`.
- `translation_time_ms`: deterministic translation time plus Ollama fallback time.
- `sentiment_time_ms`: `0.0` for the combined Ollama operation.
- `total_time_ms`: sum of the two existing timing fields.

This keeps `/analyze` byte-for-byte schema compatible. `/analyze/intelligence` receives the normalized records and runs the existing intelligence stage exactly as it does for deterministic records.

## Failure detection

Ollama is invoked only when the deterministic path has a recoverable failure:

- IndicTrans2 and NLLB both raise.
- Both translators return output rejected by the existing quality checks.
- The sentiment classifier raises for an otherwise valid translated post.
- A result contains an invalid sentiment category, confidence, or non-finite timing.

English bypass and valid deterministic results never invoke Ollama. A low-confidence but structurally valid deterministic result is not considered failure; this avoids silently replacing the primary model based on an arbitrary confidence threshold.

## Provider behavior

The fallback uses the existing Ollama base URL and model settings, with separate environment variables for enablement, timeout, and bounded concurrency. Calls use temperature zero, JSON-schema output where supported, no streaming, and a token budget sufficient only for the three required fields.

Fallback calls are bounded and deduplicated by exact text within a request. Logs record request IDs, item counts, latency, and sanitized error types without post bodies. `/health` reports whether fallback is enabled, configured, and reachable.

If Ollama times out, is unavailable, emits malformed JSON, adds fields, or returns invalid values, the endpoint returns `503`. The service never invents a translation, category, confidence, or partial success.

## Testing

Automated tests cover:

- Successful deterministic posts never call Ollama.
- A mixed batch sends only failed indices to Ollama and preserves order.
- Ollama output becomes exactly the existing `PipelineResult` fields.
- All existing sentiment categories are accepted and no others are.
- Additional/missing fields, unchanged/non-English output, invalid confidence, timeout, and provider errors return `503`.
- Fallback runs after the inference lock is released.
- `/analyze` and `/analyze/intelligence` retain their current response contracts.
- Existing native Urdu, Roman Urdu, Hinglish, English bypass, and NLLB tests remain green.

Live verification will force a deterministic failure through a test-only injected provider in automated tests, then use a representative post against the configured Ollama service without changing production response schemas.
