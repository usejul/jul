# Changelog

Versions come from git tags (see [publishing](docs/publishing.md)): a `v*` tag releases to PyPI, every push to
`main` publishes a dev build to TestPyPI.

## Unreleased

### Added

- **A cut input is no longer silent.** Each time a reading drops the end of a state to fit its limit (a cross model at the
  `max_length` / `max_state` of its `cross.json`, a pointer model at the `max_state_tokens` of its `decision.json`,
  an encoder at its positions, the onnx backend at `JUL_ONNX_MAX_TOKENS`), a warning is logged on the `jul.truncation`
  logger, printed by `jul serve`: the reading, its limit, the tokens dropped. The response carries
  `usage.truncated_tokens`, the largest cut of the call (0 when nothing was cut), so an HTTP client sees it
  too; `jul serve --escalate-to` and remote targets pass it on. The limit of each reading is in
  [docs/models.md](docs/models.md#input-limits). The cut itself is unchanged (#35).

- **`on_long="cut|error"`** (`system_one`, `TypeSafeClient`, a `jul serve` request, `JUL_ON_LONG`,
  `jul bench --on-long`): past a reading's limit, answer on what was read (default, unchanged) or refuse the
  call (`jul.truncation.InputTooLong`, a `ValueError`; HTTP 400) naming the reading and its limit. Only a cut of
  the state is refused; an option description cut to its own limit is logged and counted. `jul bench --on-long
  error` counts the refused rows in `refused_rows`, the accuracy is over the others (#36).

- **Letter-readout decision models** (#41): JevK5, plumb-4b, Quyet-1.0-Medium and spark-s1-4b-v6, which answer
  with the logits of their option letters after their own prompt, divided by a calibration temperature, are read
  as their own runtime reads them. `jul models add NAME --repo REPO` recognises them by their runtime's config
  (`jevk5_config.json`, `quyet_config.json`, `calibration.json`) or a `decision.json` with
  `"method": "letters"`; nothing is fitted. JevK5's knockout reads more than 16 options.
  `scripts/letters_parity.py` compares the probabilities with each runtime on its README examples.

- **Opt-in OpenTelemetry export** (`JUL_ENABLE_TELEMETRY=1`, `pip install "jul[otel]"`): metrics (`jul.session.count`,
  `jul.decision.count`, `jul.token.usage`, `jul.request.duration`, `jul.decision.confidence`, `jul.request.error.count`)
  and events (`jul.request`, `jul.decision`, `jul.request_error`) to the collector set by the standard `OTEL_*`
  variables, named after Claude Code's. State, question details and probabilities stay out unless switched on
  (`JUL_OTEL_LOG_*`); off, OpenTelemetry is never imported. Remote deciders (`SystemOneHTTP`: Jev, Ollama,
  Clef, a remote `jul serve`) report too, under the model the server answered with and `backend=remote`, so
  each escalation tier shows up under its own model. `jul.request.duration` buckets go up to 5 min,
  `jul.decision.confidence` has buckets between 0 and 1. See
  [docs/telemetry.md](docs/telemetry.md) (#6).

### Changed

- The encoder and onnx cuts were a `warnings.warn` shown once per call site; they are now logged on every
  call, like the others.

## 0.5.1 — 2026-10-07

### Fixed

- **MLX with mlx-lm 0.32**: a batch read over a cached prefix (a routed Score on `fast`, any batched vector
  pass) failed with `too many values to unpack`, mlx-lm 0.32 having made `KVCache.state`
  `(keys, values, offset)`. The prefix is now copied from the cache's keys and values up to its offset.
- **MLX with mlx-lm 0.32, hybrid models** (WeMM / Qwen3.5): the state snapshot kept after a cached prefix was
  shared with the queries restored from it, so a query could start from the previous query's recurrent state.
  Each query now restores a copy.

### Changed

- The homepage and the docs take a Macintosh System 1 look, with one amber accent.

## 0.5.0 — 2026-10-06

### Added

- **`jul bench` compares JuL with remote servers**: `--models` also takes `typesafe` (Jev), `ollama[:model]`,
  `cloudflare:clef|clef-flash` and any `/v1/systemone` URL as `URL#model` (Kev, a hosted Laya…), each with the
  user's own key (`--remote-key-env` for a URL). Zero-shot only, network latency included, the report says the
  rows were sent there; a missing key fails that target alone, before anything is sent.
- **`jul bench --escalate-to TARGET`** measures each local model as a cascade (itself, then the remote server for
  the answers below `--min-confidence`, as `jul serve --escalate-to`): the accuracy of the pair and the share of
  rows sent to the server, also on the verdict line when the cascade is picked. The server's failures are
  counted on the row, and a server that failed on every escalated row (a rejected key) makes the row `n/a`.
- **`autotune` works on decision models** (`fast`, `jul-decision-minicpm5-2b`). The head is trained on the
  vectors of the model's vector fallback (the preset's `routing` block, fitted on the same weights). A
  question with an active head is read through that fallback plus the head, and the others keep the pointer
  head. The safety net compares the head with the pointer head for Choice and Noul, and with the vector
  reading for a routed type such as Score. `jul bench` now autotunes decision models too.
- A decision model can **route question types to its vector reading**: `"routing": {"types": ["score"]}` in its
  `decision.json` sends those questions to the vectors fitted by `jul models add` on the same weights, at any
  option count. For pointer heads that read Score worse than the vectors do.

### Changed

- **The alias `fast` is now `jul-decision-minicpm5-2b`**, built in (it was `minicpm5-2b`, read with vectors):
  0.680 against 0.630 on the bench below, at the same speed. Its Score reading is shipped fitted per backend
  (`assets/presets/jul-decision-minicpm5-2b@{mlx,torch}.json`). `one_word_only=True` leaves it as it is (it
  already reads in one pass). On MLX, Choice questions above 20 options go to the vectors (4 to 8x faster, a few
  points less accurate, see the preset's notes); `route_above=0` keeps the pointer head.
- **`jul-decision-wemm-4b` reads Score with the vectors.** Its adapters ([`usejul/jul-decision-wemm-4b`](https://huggingface.co/usejul/jul-decision-wemm-4b)
  `1525e34`) keep Noul and Choice; their Score reading was worse than the vector reading of the same weights.
  On a bench of 300 typed questions written from scratch for it (sentiment, finance, support, agent routing,
  moderation): 0.627 → 0.677 (Score 0.27 → 0.41). It is a change of the adapters' `cross.json`, so jul 0.4.0
  gets it too.
- **`minicpm5-2b-decision` is now [`usejul/jul-decision-minicpm5-2b`](https://huggingface.co/usejul/jul-decision-minicpm5-2b)**
  (and `-mlx-4bit`), the old names redirect. **v2**: trained on human-written typed decisions, Score routed to its
  vector reading; 0.680 on that bench at 85 ms, against 0.637 for v1.1 (`revision="v1.1"`).

### Fixed

- A decision model's questions routed to its vector fallback were read at the pointer preset's temperature
  (1.0) instead of the fallback's own tau, which flattened their probabilities to near uniform (same answer,
  wrong confidence; a Score's expected level collapsed to the middle).
- **EmbeddingGemma read as an encoder.** A model whose config sets `use_bidirectional_attention`
  (`google/embeddinggemma-300m`, a bidirectional `gemma3_text`) is now read like the encoders: mean over the
  text, its declared `query` prompt, no prefix cache. It was read as a decoder (last token, cached prefix):
  0.160 dev accuracy after `jul models add`, 0.655 now (multilingual-e5-small: 0.585, same data). Torch backend.

## 0.4.0 — 2026-10-05

### Added

- **Laya** as a model: `TypeSafeClient(model="laya")`, `--model laya` (also `laya:multilingual`,
  `laya:typed-decisions` or `laya:<hub repo or directory>`), with `pip install "jul[laya]"` or
  `jul setup --model laya`. Laya runs on its own package; JuL hands it the questions and returns its answers
  as JuL's, so `ask`, `run`, `serve`, `bench` and `Escalation` take it. JuL's readings, `autotune` and
  `pack` do not apply to it. See [Models > Laya](docs/models.md#laya).
- **`jul bench`: pick a model on your own data.** `jul bench test.jsonl --train train.jsonl --models
  jul-decision-e5-small@onnx,minicpm5-2b@torch -O results.json` answers every test row with each model and
  reports, per question, accuracy with its 95% interval, latency (p50/p95) and the model to pick: the fastest
  whose interval reaches the best. Each row carries its question (`question, options, state, answer`), so one
  file can mix questions and types. With `--train` (a separate file) each model is also autotuned and measured
  again on the same test rows; train and test are checked for overlap first (exact duplicates after folding
  case, punctuation and digits stop the bench, near duplicates are reported, `--drop-overlap` removes them).
  `--method` / `--features` (or `auto`) compare the zero-shot readings and autotune heads of each model, one
  row each. Tables in the site's amber on a terminal, `--json` or `-O file` for the full results.
- **Embeddings APIs as a backend** (`--backend api`): the vector reading over any embeddings endpoint, local or
  hosted, without the model's weights. `jul models add NAME --repo ollama:qwen3-embedding:0.6b --backend api`
  (also `openai:`, `mistral:`, `voyage:`, or `http(s)://host/v1#model` for any OpenAI-compatible server) fits
  the center and tau as for a local model. One layer, no logits (Noul and Score are read as vectors).
- **Escalation on confidence**: `jul.Escalation` chains deciders (a local client, then `SystemOneHTTP` for
  any `/v1/systemone` server: Jev, Ollama's Nimble, Kev, another `jul serve`); only the questions answered
  below the bar go on to the next one, and the response says per question which tier answered.
  `jul serve --escalate-to typesafe|ollama[:model]|URL --min-confidence 0.8` does the same over HTTP, with each
  provider's usual key variable (`TYPESAFE_API_KEY`).
- **Clef on Cloudflare Workers AI** as an escalation tier: `--escalate-to cloudflare:clef-flash` (or `:clef`),
  `jul.escalate.cloudflare_tier()` in Python, with `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`.
- **`jul-decision-wemm-4b` is the default model** (alias `accurate`): `wemm-4b-4bit` plus LoRA adapters
  ([`usejul/jul-decision-wemm-4b`](https://huggingface.co/usejul/jul-decision-wemm-4b)) that read Noul,
  Score and Choice with the question and the text together. Decision bench (2,108 questions, PyTorch): 0.849,
  Jev 0.873. Attached on PyTorch; on MLX it reads as `wemm-4b-4bit` until the adapters are measured in
  4-bit. `wemm-4b-4bit` stays, vectors only.
- LoRA cross models read a Choice listwise when their `cross.json` has a `choice` entry (text, question and
  every option in one pass), mixed with the vector reading (`mix`).
- A preset's `cross` entry may name a repo for some backends only (`{"torch": ...}`); the others read with
  the vectors.
- **Contrastive heads on any backbone**: a reading mode where two projection heads (InfoNCE) read a frozen
  backbone, set by `contrastive.json` (layer, pooling, rendering, Noul texts, prefix), on MLX, torch or onnx,
  decoders and encoders. `jul models add <name> --repo <backbone> --train-heads rows.jsonl` trains them;
  `autotune` trains its usual head on the same embedding.
- CLM-8B is one profile of it: `jul models add clm-8b --repo Contrastive-LM/CLM-v0.1-8B` converts CLM's heads
  and runs them on their frozen Qwen3-8B (MLX 8-bit or torch bf16), without vLLM; CLM's reference answers
  reproduced within 0.025. docs/models.md#contrastive-heads-any-backbone-clm-8b.

## 0.3.0 — 2026-09-28

### Added

- **Cross models**: a second reading that takes the question and the text together, for the questions an
  embedding model answers near chance (paraphrase, inference, compositions, dates, "is it late?"). A preset
  gets one with `jul models add --cross`, or by itself from a repo that carries a `cross/` folder; its
  `cross.json` declares the types it reads (Noul and Score), the vectors read everything else.
  `method="vector"` or `method="cross"` forces one reading for a call.
- **LoRA cross models**: the cross reading as adapters on the preset's own model instead of a second
  encoder (`"method": "lora"` in `cross.json`), switched on only while a Noul or a Score is read, so one model
  sits in memory and the vector reading is unchanged. MLX and PyTorch.
  [`usejul/jul-decision-wemm-4b-4bit-mlx`](https://huggingface.co/usejul/jul-decision-wemm-4b-4bit-mlx):
  yes/no 0.841 on Kev's typed decisions, against 0.762 for the vectors of the default model.
- `jul pack` ships only the models a bundle's questions read with (vectors, cross, or both);
  `JUL_ONNX_CROSS_MODEL` points the cross graph at S3 as `JUL_ONNX_MODEL` does the vector one.
- Encoders read their input prefixes from the model's `config_sentence_transformers.json`, else from
  `assets/text_prefixes.json` (`$JUL_HOME/text_prefixes.json` extends it).
- Models: [`usejul/jul-decision-e5-small`](https://huggingface.co/usejul/jul-decision-e5-small) and its ONNX
  8-bit build, e5-small trained on jul decisions, with a cross model; a step-by-step AWS Lambda guide.

### Changed

- `autotune`: a head for a Noul or a Score must beat the cross model on the labeled examples to take the
  question; otherwise the question stays with the cross model.
- PyTorch: texts behind a cached prefix that cannot be repeated over a batch (the recurrent state of
  Qwen3.5 and WeMM-Embedding) are read in one batch, the prefix run again with each: about 150 texts/s
  instead of 12 for WeMM-Embedding-4B on an A10G. MLX keeps one at a time.

### Fixed

- `jul models add <built-in preset> --cross ...` saved the preset as `<name>@None.json`.

## 0.2.0 — 2026-09-26

### Added

- **onnx backend** (ONNX Runtime on CPU, no torch) and **encoders as backbones** (BERT, XLM-R,
  multilingual-e5): read as they were trained, mean-pooled, with their own input prefix. `JUL_HOME` moves
  everything jul writes.
- `jul pack` (first named `jul compile`): a fixed set of questions and its models as a bundle, for a
  deployment such as AWS Lambda; per-question formulations.
- `autotune` with lexical and hybrid heads (`features=`): e5-small reaches 0.790 on the Jev benchmark with a
  hybrid head, at 6 ms per text.
- `jul serve`: a local HTTP server that speaks the Jev protocol (`/v1/models` included), so the official SDK
  talks to it; single-option Choice.

### Changed

- The version comes from git (setuptools-scm): `main` publishes dev builds to TestPyPI, tags release.

## 0.1.1 — 2026-09-24

### Changed

- `wemm-4b-4bit` (WeMM-Embedding-4B, MLX 4-bit) is the new default model, in place of `minicpm5-2b`;
  `qwen3.5-9b` is dropped.
- Project links move to the `usejul` organization.

## 0.1.0 — 2026-09-24

First release: typed decisions (`Choice`, `Noul`, `Score`) with calibrated probabilities from one forward
pass of a local model, behind the interface of TypeSafe's (Jev) Python SDK.

- MLX and PyTorch backends; presets for `minicpm5-2b` (the default) and `qwen3.5-9b`; `jul models add` fits
  a preset for any model on the dev sets.
- Decision models read with their own pointer format, with a vector fallback and a measured routing
  threshold.
- `Context`, `autotune` (per-task heads), `jul synth` (synthetic labeled data), `jul setup`.
- Apache-2.0.
