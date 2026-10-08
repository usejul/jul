# Configuration

JuL has no config file. Settings come, from the most specific to the most general, from an argument of
the call, a flag of the command, an environment variable, or the preset. This page lists every
environment variable the code reads and every file it writes.

## Environment variables

### Where and how the model runs

| Variable | Default | What it does |
| --- | --- | --- |
| `JUL_BACKEND` | `mlx` on Apple Silicon if installed, else `torch` | the backend: `mlx`, `torch`, `onnx`, `api` |
| `JUL_DEVICE` | `cuda` > `mps` > `cpu` | the torch device |
| `JUL_DTYPE` | `bfloat16`; `float32` on CPU; `float16` on GPUs without bf16 (T4) | the torch dtype: `bfloat16`, `float16`, `float32` |
| `JUL_BATCH_TOKENS` | `16384` | most tokens in one batch (rows × longest prompt), mlx and torch |
| `JUL_BATCH_SIZE` | `64` | most rows in one batch, mlx and torch |
| `JUL_HOME` | `~/.jul` | where presets, contexts, heads and calibration data live (point it at a read-only package in a Lambda) |
| `JUL_ON_LONG` | `cut` | a state over a reading's limit: `cut` answers on its beginning, `error` refuses the call ([input limits](models.md#input-limits)) |

### ONNX

| Variable | Default | What it does |
| --- | --- | --- |
| `JUL_ONNX_MODEL` | the export's own graph | another graph for the vector model: a path or `s3://bucket/key`, read into memory |
| `JUL_ONNX_CROSS_MODEL` | the cross model's own graph | the same for the cross model |
| `JUL_ONNX_THREADS` | ONNX Runtime's | intra-op threads (in a Lambda: one per whole vCPU) |
| `JUL_ONNX_MAX_TOKENS` | `2048` | longest row; the end of the input is cut, the prompt kept |
| `JUL_ONNX_BATCH_TOKENS` | `2048` | rows × longest per run |

### Embeddings APIs (`--backend api`)

| Variable | Default | What it does |
| --- | --- | --- |
| `JUL_API_BATCH` | `16` | texts per request |
| `JUL_API_MAX_CHARS` | `8000` | each text is cut to this many characters, with a warning |
| `JUL_API_TIMEOUT` | `300` | seconds per request |
| `OLLAMA_HOST` | `http://localhost:11434` | the Ollama server for `ollama:` repos |
| `OPENAI_API_KEY`, `MISTRAL_API_KEY`, `VOYAGE_API_KEY` | — | keys for `openai:`, `mistral:`, `voyage:` repos |

### Serving and escalation

| Variable | Default | What it does |
| --- | --- | --- |
| `JUL_API_KEY` | none | the key `jul serve` requires (`--api-key` overrides it); without one, any caller is accepted |
| `TYPESAFE_API_KEY` | — | Jev's key, for `--escalate-to typesafe` and `jul bench --models typesafe` |
| `CLOUDFLARE_ACCOUNT_ID` | — | 32 hex characters, for `--escalate-to cloudflare` and `jul bench --models cloudflare:clef` |
| `CLOUDFLARE_API_TOKEN` (or `CLOUDFLARE_AUTH_TOKEN`) | — | Workers AI token, for the same |

Escalation and bench keys are read from the environment only (`--escalate-key-env`, or `--remote-key-env`
for `jul bench`, names the variable), and no key is ever logged. The server key can also be given with `--api-key`, but the environment keeps it out of `ps`
and shell history.

### Telemetry ([OpenTelemetry](telemetry.md), off by default)

| Variable | Default | What it does |
| --- | --- | --- |
| `JUL_ENABLE_TELEMETRY` | off | `1` sends metrics and events to the collector named by the standard `OTEL_*` variables (`pip install "jul[otel]"`) |
| `JUL_OTEL_LOG_STATE` | off | sends the state itself instead of `<REDACTED>` |
| `JUL_OTEL_LOG_QUESTION_DETAILS` | off | sends question names, instructions, option keys, context name, error messages |
| `JUL_OTEL_LOG_PROBABILITIES` | off | sends the full distribution over options |
| `JUL_OTEL_LOG_ANSWERS` | on | `0` sends `<REDACTED>` instead of the chosen answer |
| `JUL_OTEL_CONTENT_MAX_LENGTH` | `61440` | content attributes are cut to this many characters |

### Tests only

| Variable | What it does |
| --- | --- |
| `JUL_SLOW` | `1` runs the tests that load a real model (read by `tests/conftest.py`, never by the library) |
| `JUL_TEST_TOKENIZER`, `JUL_TEST_ENCODER_TOKENIZER` | the tokenizers the tiny test models use |
| `JUL_TEST_MODEL` | the model the backend tests load |

### From the Hugging Face libraries

JuL downloads weights with `huggingface_hub`, so its variables apply: `HF_HOME` (cache location, default
`~/.cache/huggingface`), `HF_TOKEN` (gated or private repos), `HF_HUB_OFFLINE=1` (never touch the network
once the weights are cached).

## Files on disk

```text
~/.jul/                              ($JUL_HOME)
├─ presets/<name>@<backend>.json     written by `jul models add`: layers, tau, center, what was measured
├─ contexts/<name>/                  a saved Context
│  ├─ meta.json                      description, calibrations
│  ├─ examples.txt                   its unlabeled texts
│  ├─ centers/*.npy                  one center per model, prompt and layer
│  └─ heads/<digest>.npz + .json     heads trained by `autotune`, one per question
├─ heads/<name>/                     contrastive heads (`jul models add --train-heads`, CLM)
├─ calibration-data/                 the development sets `jul models add` downloads once
└─ text_prefixes.json                optional: input prefixes for encoders, extends the built-in list
~/.cache/huggingface/                the model weights (HF_HOME)
```

A head is keyed by the model, the backend, the question's instructions and its options: change any of them
and the question is answered zero-shot until you run `autotune` again. Deleting `~/.jul` loses fitted
presets, contexts and heads, never the weights.

## Precedence

| Setting | Call argument | CLI flag | Environment | Otherwise |
| --- | --- | --- | --- | --- |
| model | `model=` | `--model` | — | `jul-decision-wemm-4b` |
| backend | `backend=` | `--backend` | `JUL_BACKEND` | detected |
| reading | `method=` | `--method` | — | the preset's, per question type |
| context | `context=` | `--context` | — | none |
| server key | `serve(api_key=)` | `--api-key` | `JUL_API_KEY` | none |
