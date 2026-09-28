# Changelog

Versions come from git tags (see [publishing](docs/publishing.md)): a `v*` tag releases to PyPI, every push to
`main` publishes a dev build to TestPyPI.

## Unreleased

### Added

- **Contrastive models (CLM-8B)**: `jul models add clm-8b --repo Contrastive-LM/CLM-v0.1-8B` converts CLM's
  projection heads once and runs them on their frozen Qwen3-8B, on MLX (8-bit) or torch (bf16), without vLLM.
  Same texts as CLM's `build_pairs`; CLM's reference answers reproduced within 0.025. `autotune` trains its heads
  on the encoder embedding. `jul/contrastive.py`, docs/models.md#contrastive-models-clm-8b.

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
