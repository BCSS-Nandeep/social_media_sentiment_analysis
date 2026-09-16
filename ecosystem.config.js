// PM2 process definition for sentiment-api.
//
// This did not exist anywhere in version control before now — the running
// process was started by hand at some point and only PM2's live runtime
// state remembered how (confirmed via `pm2 jlist`/`pm2 describe`, not
// guessed). Captured here so the startup command is no longer tribal
// knowledge, and so the --workers count (see below) is an explicit,
// reviewable decision instead of an invisible server-side setting.
//
// --workers: deliberately 1, not 2. This process previously ran with
// `--workers 2`, which uvicorn implements as two fully independent OS
// processes with no shared memory — including two independent LlmGate
// instances (src/llm_gate.py). The tenant-fair-queue feature needs one
// process-wide gate to actually provide the fairness it promises (see
// docs/TENANT_AWARE_SENTIMENT_QUEUE_RESEARCH.md §13 in the Saga repo for
// the full tradeoff analysis). Cost: roughly halves deterministic-stage
// (translation/sentiment) raw throughput versus 2 workers. Revisit only
// with real load data and a real shared-state plan (e.g. Redis) if that
// cost turns out to matter more than fairness in practice.
module.exports = {
  apps: [
    {
      name: 'sentiment-api',
      cwd: __dirname,
      script: '.venv/bin/uvicorn',
      interpreter: 'none',
      args: 'api_server:app --host 0.0.0.0 --port 8003 --workers 1 --timeout-worker-healthcheck 60',
      autorestart: true,
      max_restarts: 10,
      out_file: './.logs/sentiment-api-out.log',
      error_file: './.logs/sentiment-api-error.log',
      env: {
        PYTHONUNBUFFERED: '1',
        HF_HOME: '/data/hf-cache',
        TRANSFORMERS_CACHE: '/data/hf-cache',
        TMPDIR: '/data/tmp',
        SENTIMENT_DEVICE: 'cpu',
      },
    },
  ],
};
