# Models

## Every model measured

Jev scores 0.753 on the same benchmark. `wemm-4b-4bit` and `minicpm5-2b` are built in; any other is
one command away,
`jul models add <name> --repo <repository>`, which fits it on the dev sets.

| Name | Repository | Memory | Jev bench, zero-shot | + autotune |
| --- | --- | ---: | ---: | ---: |
| `wemm-4b-4bit` | [`usejul/WeMM-Embedding-4B-mlx-4bit`](https://huggingface.co/usejul/WeMM-Embedding-4B-mlx-4bit) | 2.6 GB | 0.857 | **0.897** |
| `wemm-4b` | [`tencent/WeMM-Embedding-4B`](https://huggingface.co/tencent/WeMM-Embedding-4B) | 4.5 GB | **0.877** | 0.862 |
| `wemm-9b` | [`tencent/WeMM-Embedding-9B`](https://huggingface.co/tencent/WeMM-Embedding-9B) | 9.0 GB | 0.863 | 0.857 |
| `wemm-2b` | [`hfadam/WeMM-Embedding-2B-MLX-4bit`](https://huggingface.co/hfadam/WeMM-Embedding-2B-MLX-4bit) | 1.5 GB | 0.777 | 0.805 |
| `f2llm-8b` | converted locally, not published | 4.5 GB | 0.820 | 0.850 |
| `f2llm-4b` | [`fcmeyer/F2LLM-v2-4B-mlx-6bit`](https://huggingface.co/fcmeyer/F2LLM-v2-4B-mlx-6bit) | 3.0 GB | 0.840 | 0.853 |
| `f2llm-1.7b` | converted locally, not published | 1.0 GB | 0.817 | 0.833 |
| `f2llm-0.6b` | [`fcmeyer/F2LLM-v2-0.6B-bf16-mlx`](https://huggingface.co/fcmeyer/F2LLM-v2-0.6B-bf16-mlx) | 1.1 GB | 0.623 | 0.817 |
| `qwen3-embedding-8b` | [`mlx-community/Qwen3-Embedding-8B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-8B-4bit-DWQ) | 4.0 GB | 0.773 | 0.806 |
| `qwen3-embedding-4b` | [`mlx-community/Qwen3-Embedding-4B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-4B-4bit-DWQ) | 2.1 GB | 0.733 | 0.775 |
| `qwen3-embedding-0.6b` | [`mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ) | 0.32 GB | 0.637 | 0.760 |
| `harrier-0.6b` | [`majentik/harrier-oss-v1-0.6b-MLX-4bit`](https://huggingface.co/majentik/harrier-oss-v1-0.6b-MLX-4bit) | **0.31 GB** | 0.667 | 0.814 |
| `minicpm5-2b` | [`openbmb/MiniCPM5-2B-MLX`](https://huggingface.co/openbmb/MiniCPM5-2B-MLX) | 2.7 GB | 0.617 | 0.757 |
| `minicpm5-2b-decision` | [`usejul/minicpm5-2b-decision-mlx-4bit`](https://huggingface.co/usejul/minicpm5-2b-decision-mlx-4bit) | 1.3 GB | 0.796 |  |
| `ternary-bonsai-1.7b` | [`prism-ml/Ternary-Bonsai-1.7B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit) | 0.46 GB | 0.640 | 0.760 |
| `ternary-bonsai-8b` | [`prism-ml/Ternary-Bonsai-8B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-mlx-2bit) | 1.75 GB | 0.563 | 0.753 |
| `bitnet-2b` | [`mlx-community/bitnet-b1.58-2B-4T`](https://huggingface.co/mlx-community/bitnet-b1.58-2B-4T) | 1.1 GB | 0.617 | 0.723 |
| `e5-small` | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small), onnx 8-bit | **0.09 GB** | 0.543 | 0.713 (**0.790** hybrid) |

`+ autotune` is a head on the vectors (`features="vector"`). With a hybrid head (vectors + TF-IDF, see
[`features=`](tuning.md#what-the-head-reads-features)), `e5-small` — an encoder, see [Micro models](#micro-models-encoders) —
reaches **0.790**, above Jev's 0.753 (AG News 0.95, Banking77 0.88, Emotion 0.54), at 6 ms per text on
an M4 Pro. The other models were not measured with a hybrid head.

## Adding a model

```bash
jul models add my-model --repo org/Some-Instruct-3B            # on the default backend
jul models add minicpm5-2b --backend torch                     # a known preset, on another backend
```

One command runs the protocol that produced the built-in presets, on the dev datasets only (never on
the Jev benchmark):

1. **checks** — the model loads, the prefix cache leaves the vectors unchanged, a call does not
   disturb the next one, the letters reading (Noul, Score) has single-token markers;
2. **extraction** — 4 dev sets x 50 examples and 200 generic texts, every layer of the upper half
   read in the same pass;
3. **choice** — layers and center by dev accuracy averaged over neighbouring layers (a plateau, not a
   lucky peak), then tau by pooled NLL;
4. **output** — `~/.jul/presets/<name>@<backend>.json` and its generic center, used from then on by
   `--model <name>` on that backend. `jul models` lists it with what was measured.

The calibration data is downloaded once from BTZSC into `~/.jul/calibration-data` (needs
`pip install "jul[calibrate]"`), or taken from `--data <dir>`. The dev accuracy it reports comes
with its standard error (±3.5 points at n=200): it orients, it does not rank close models. Measure
on the Jev bench separately, once.

## Micro models: encoders

An encoder (BERT, XLM-R, multilingual-e5…) is a backbone like any other: `jul models add`,
`autotune` and `jul pack` run on it unchanged, on the torch and onnx backends. JuL recognizes one by
its `model_type` and reads it as it was trained, not as a decoder (`lib/jul/encoder.py`):

- the vector is the mean of the layer over the whole sequence (the sentence embedding e5 was trained to
  produce), not a last token; zero-shot is plain embedding similarity between state and options;
- the prompts are the model's own input convention (`query: {state}` for e5), no chat template and no
  "in one word" cue (`Backbone.templates`). A model declares it in its `config_sentence_transformers.json`
  (`"prompts": {"query": "query: "}`); for repos that do not, the official e5 ones included, JuL
  reads it from a list by repo name, `lib/jul/assets/text_prefixes.json`, which
  `$JUL_HOME/text_prefixes.json` extends or overrides;
- attention is bidirectional, so no prefix can be cached: the prefix runs again with each query, and a
  sequence is cut to the model's positions (512), the input first, then the prefix;
- no logits: the letters reading and decision models need a decoder.

```bash
python -m jul.backends.onnx_export intfloat/multilingual-e5-small models/e5-small-onnx
python -m jul.backends.onnx_export models/e5-small-onnx models/e5-small-onnx-w8 --int8   # 86 MB
jul models add e5-small --repo models/e5-small-onnx-w8 --backend onnx
```

Why bother: on a support-triage task (jul-lambda, 2026-09-25), multilingual-e5-small (21 M
parameters outside its embedding) with a hybrid head matched Harrier 0.6B (440 M) — emotion 0.787
against 0.791, Banking77 0.890 against 0.890 — in **4 ms per message against 37 ms** on an M4 Pro,
and 17 ms on a 1,769 MB AWS Lambda ($0.59 per million calls, cold start 2.4 s). Zero-shot, on the
calibration dev sets, it scored 0.590 against 0.475 for Harrier 0.6B. The heads carry it: alone,
zero-shot, a small encoder is no match for a 4B embedding model.

## Cross models: reading the question and the text together

The vector reading encodes the text and each option apart: the model never sees both at once. That is
what makes it fast and cacheable, and also why a question about how two things relate — is this a
paraphrase of that sentence, does it follow that…, does the request mention a place and not a date — is
answered near chance by an embedding model of any size. A *cross model* is a second small encoder,
fine-tuned on pairs: `<s> query: question </s></s> text </s>` goes through it once, and the mean of its
last layer feeds a head chosen by the question type (`lib/jul/cross.py`):

- `Noul`: one pass, three logits (yes, no, unknown); `noul = p(yes) + p(unknown) / 2`. Its own descriptions
  of true and false, when given, are appended to the question;
- `Score`: one pass per level, one logit each;
- `Choice`: supported, not routed by default (the vector reading classifies as well and reads each option
  once for every call).

It sits next to a vector preset, not in place of it. `jul models add` attaches one; the types its
`cross.json` declares (Noul and Score) are then read by it, and everything else by the vectors. A
question with a tuned head or a calibration from `autotune` keeps the vector reading those were fitted on,
and `method="vector"` or `method="cross"` forces one reading for a call.

```bash
jul models add jul-decision-e5-small --repo usejul/jul-decision-e5-small-onnx --backend onnx   # cross/ attached
# a cross model of your own: export it like any encoder, then attach it
python -m jul.backends.onnx_export <cross model> models/cross-onnx --layers 11
python -m jul.backends.onnx_export models/cross-onnx models/cross-onnx-w8 --int8 --embedding-bits 4   # 98 MB
cp <cross model>/cross.json <cross model>/cross_heads.npz models/cross-onnx-w8/
jul models add jul-decision-e5-small --backend onnx --cross models/cross-onnx-w8
```

`cross.json` holds the input prefix, the length the pairs are cut to (as in training), the layer, the
separator between the two segments and the types; `jul.cross.write_spec` writes it from a trained model.

Which reading answers what:

| | no labeled examples | labeled examples (`autotune`) |
| --- | --- | --- |
| `Choice` | vectors | a head on the vectors |
| `Noul`, `Score` | the cross model | the cross model, unless a head on the vectors beats it on those examples |

`autotune` judges a head for a Noul or Score against the cross model's zero-shot answers on the same
examples, not against the vectors' (which are weaker there): if the head wins, the question is answered
by it; otherwise it stays with the cross model. Choice stays on vectors because they carry to labels the
models never saw — read through a cross model trained on the same data, tool routing went from 0.54 to
0.76 and classification from 0.82 to 0.86, but the zero-shot Jev bench fell from 0.557 to 0.460
(Banking77, 77 intents, 0.59 to 0.35) — and because a vector reading is what heads are trained on. Its
own vectors do not survive the cross training (0.61 to 0.39 on the dev sets), which is why the cross model
is a second set of weights.

`jul pack` ships only what its questions need: a bundle of Noul and Score questions (with no tuned head)
needs the cross model alone, a bundle of Choice questions the vector model alone (`bundle.json`, `models`).
On onnx, `JUL_ONNX_CROSS_MODEL` points the cross graph at S3 as `JUL_ONNX_MODEL` does the vector one.

Measured on `jul-decision-e5-small` (its vectors and its cross model, both ONNX 8-bit, M4 Pro), on Kev's
typed-decision questions it never trained on (`transfer-v9` development split, clean questions):

| | Noul | Choice | Score | all | Noul p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| vectors only | 0.579 | 0.401 | 0.275 | 0.460 | 19 ms |
| with the cross model | **0.726** | 0.401 | **0.500** | **0.524** | **7 ms** |
| Jev (published) | 0.847 | 0.833 | 0.950 | 0.854 | — |

Paraphrase goes from 0.50 to 0.79, QNLI from 0.55 to 0.70, offensive posts from 0.725 to 0.80. The cross
model was trained on relational yes/no (paraphrase, inference, compositions, dates), single-text
decisions and questions about the writer's tone: "is the customer angry?" gets 0.84–0.99 on angry
support messages and 0.00–0.02 on calm ones. What it does not do well yet: urgency (no training data
under a commercial license), hard but polite complaints read as offensive, sentences with the same words
in another order, and knowledge questions (MMLU does not move — that needs a larger model, not another
reading).

### A cross model on the preset's own weights (LoRA)

A cross model can also be a set of adapters on the vector model itself instead of a second encoder: one
model in memory for both readings. Its `cross.json` says `"method": "lora"` and names LoRA adapters (A and
B for each Linear layer, by module path, as transformers and mlx-lm name them) and the heads. The engine
attaches them to the backbone it already holds and switches them on only while it reads a Noul or a
Score, so the vector reading is unchanged: the features are the same with the adapters attached
(`tests/test_cross.py`). The prompt is `Text: "<text>"`, then `Question: <question>` (and, for a Score,
`Candidate answer: <level>`), then `Verdict:`, read on the last token of the last layer after the final
norm: one pass for a Noul, one per level for a Score. MLX and PyTorch.

```bash
# a directory with cross.json, adapter.npz and cross_heads.npz, on top of the default model
jul models add wemm-4b-4bit --backend mlx --cross models/jul-decision-wemm-4b-4bit
jul ask noul "Was it paid on time?" --state "Invoice due May 9; paid May 3."
```

`jul-decision-wemm-4b-4bit` is WeMM-Embedding-4B with such a cross model (rank-16 adapters on its 248 Linear layers: 32.5M parameters, 65 MB in
float16; trained on relational yes/no, single-text decisions and scores). Measured on the same Kev
questions as above, never trained on:

| | Noul | Score |
| --- | ---: | ---: |
| `wemm-4b-4bit`, vectors only (PyTorch bf16) | 0.762 | 0.325 |
| `jul-decision-wemm-4b-4bit`, PyTorch bf16 | **0.859** | **0.550** |
| `jul-decision-wemm-4b-4bit`, MLX 4-bit (M4 Pro, ~115 ms per Noul) | 0.841 | 0.300 |

Paraphrase goes from 0.69 to 0.93 and QNLI from 0.79 to 0.87 (MLX), and Noul's calibration error from 0.126
to 0.043. On date questions worded freely (warranties, trials, bookings, ages, deadlines: 390 cases in
English and French), comparing two dates is solved — which comes first, same month, on time: 1.00, against
0.62–0.90 for the vectors — but a gap to compute (an age, a warranty in months, a trial in days) stays near
chance, with or without the adapters. That is also why Score barely moves in 4-bit: its questions here are
deadlines, close calls that 4-bit weights flip. Choice questions keep the vectors, so the Jev benchmark
(0.857 zero-shot) is untouched by construction.

## Decision models

A *decision model* is a model trained to answer questions about a state, rather than to write text. It
reads the state and the options through delimiter tokens it learned, and a small head scores each
option against the question. `jul` runs one with no code of its own: everything that model needs sits
next to its weights, in a `decision.json` (delimiters, layout, readout, head file, temperature, and the
longest state and question it was trained on).

```bash
jul models add minicpm5-2b-decision --repo usejul/minicpm5-2b-decision-mlx-4bit   # MLX, 1.3 GB
jul models add minicpm5-2b-decision --repo usejul/minicpm5-2b-decision --backend torch
jul ask choice "Which team should handle this ticket?" -o billing -o shipping -o access \
    --state "I was charged twice for order 4411" --model minicpm5-2b-decision
```

A repo (or a local directory) holding a `decision.json` is registered as it is: there is nothing to
fit, no layer to choose and no tau, so the command returns at once. The API is the same as for any other model, and all three
question types go through the same format. The state is encoded once per call and every question
continues from it, so questions never see each other.

The state is paid once per call: a ticket with four questions (two `Choice`, a `Noul` and a `Score`)
answers in **180 ms** on an M4 Pro, against 65 ms for the first question alone. What costs is the
options — they are re-read on every request — so a three-option question runs in 64 ms where a
fifty-nine-option one takes 596 ms. Weights: [`usejul/minicpm5-2b-decision-mlx-4bit`](https://huggingface.co/usejul/minicpm5-2b-decision-mlx-4bit)
(MLX, 1.3 GB) and [`usejul/minicpm5-2b-decision`](https://huggingface.co/usejul/minicpm5-2b-decision)
(PyTorch, bf16).

Two differences with the presets above: `autotune(...)` does not apply (its heads are trained on the
vectors of the other method, and such a model needs a full fine-tune instead), and a state longer than
the limit in its `decision.json` is truncated rather than stretched.

## Contrastive models (CLM-8B)

[CLM-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) is not a model of its own: two small
projection heads (19 M parameters, 75 MB) trained with InfoNCE on top of a **frozen** Qwen3-8B, read at
its last token. The state, with the question's instructions after a blank line, goes through the state
head; each option through the action head; the score is `scale * cosine` and a softmax gives the answer.
The heads mean nothing without that exact backbone, so `jul` runs Qwen3-8B itself and reads the heads
in numpy (no torch at inference, no vLLM server):

```bash
jul models add clm-8b --repo Contrastive-LM/CLM-v0.1-8B    # converts the .pt once (needs torch)
jul ask choice "Which team should handle this?" -o "billing:Charges, invoices, refunds" \
    -o "technical:Bugs and outages" --state "my invoice was charged twice" --model clm-8b
```

`jul models add` recognises a CLM repo by its `config.json` (`"model_type": "clm"`), converts the
checkpoint into `~/.jul/heads/<name>/` (`contrastive.json` + `heads.npz`) and writes a preset whose
repos are the backbone's: `Qwen/Qwen3-8B` (bf16, 16 GB) on torch, `mlx-community/Qwen3-8B-8bit`
(8.7 GB) on MLX. `python -m jul.contrastive convert <.pt or repo> <dir> --backbone-mlx <repo>` picks
another backbone; `jul models add <name> --repo <dir>` then registers that directory.

The texts are CLM's own (`schema.build_pairs`), not jul's formulations: objects rendered as `key: value`
fields, options verbatim, a Noul without descriptions read as CLM's `"true: Yes. This is true: <question>"`.
On CLM's reference requests the MLX 8-bit backbone answers as its bf16 vLLM server does (M1 Pro):

| Request | CLM, bf16 vLLM | jul, MLX 8-bit | jul, MLX 4-bit |
| --- | ---: | ---: | ---: |
| "charged twice…", Noul urgent | 0.840-0.848 | 0.865 | 0.713 |
| same, Choice billing | 0.987-0.989 | 0.991 | 0.985 |
| same, Score frustration (0-2) | 2.000 | 2.000 | 2.000 |
| "I'm very calm.", Score frustration | 1.999 ([issue #3](https://github.com/Contrastive-LM/CLM/issues/3)) | 1.998 | 1.999 |
| tides, Choice "the Moon" | 0.993 | 0.993 | 0.994 |

4-bit keeps every argmax but moves the Noul by 0.13, hence 8-bit by default. A warm call with the
options cached takes ~210 ms on an M1 Pro: every question embeds `state + instructions` once, since the
state head reads the question too.

`autotune` works: the head is trained on the encoder embedding of `state + instructions` (the vector the
state head reads), judged against the CLM heads' own answers, and a question keeps CLM's answer when the
head does not beat it. `jul pack` does not apply (vector method only).

On the Jev bench (`scripts/bench_jul.py clm-8b zero-shot,tuned`, M1 Pro, MLX 8-bit):

| | AG News | Banking77 | Emotion | Mean acc | Mean ECE | p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clm-8b` zero-shot | 0.55 | 0.08 | 0.22 | 0.283 | 0.208 | 217 ms |
| `clm-8b` + autotune (1000 labels) | 0.92 | 0.58 | 0.45 | 0.650 | 0.117 | 217 ms |

Zero-shot it collapses onto a few labels (Banking77's 77 options, Emotion), as it does behind its own
vLLM server (0.53 / 0.08 / 0.23 measured there, torch bf16 on CPU): the port is faithful, the heads are
what they are. Jev scores 0.753, `wemm-4b-4bit` 0.857.

## The built-in presets

| Preset                          | Model                               | Layers  |    tau | p50, M4 Pro | p50, M5 Max | Jev bench, zero-shot |
| ------------------------------- | ----------------------------------- | ------- | -----: | ----------: | ----------: | -------------------- |
| `wemm-4b-4bit` (alias `accurate`, default) | `usejul/WeMM-Embedding-4B-mlx-4bit` | 31 / 31 | 0.0553 |      146 ms |       55 ms | **0.857**            |
| `minicpm5-2b` (alias `fast`)    | `openbmb/MiniCPM5-2B-MLX`           | 39 / 40 | 0.0413 |   **64 ms** |             | 0.617                |

`wemm-4b-4bit` is the default because it is the most accurate: 10 points above Jev with no training.
On the same M4 Pro it is 2.3 times slower than `minicpm5-2b`, which stays the fast option.

A third option does not read a general model at all: `minicpm5-2b-decision` is MiniCPM5-2B *trained*
to answer typed questions (a merged LoRA and a pointer head). It has no layer and no tau — it brings
its own format — and it is 6 points ahead on the development sets, at a latency that depends on how
many options a question has. It is not built in: `jul models add` registers it in a second.

## How it answers

For each formulation of the preset, the state and every option go through the same prompt; the answer
is the option whose vector is closest (cosine, after subtracting a center). The scores of the two
formulations are averaged, then `softmax(cosine / tau)`.

```
1.  This text: "{state}" means in one word: "
2.  {instructions}\nPossible answers: {options}.\nText: "{state}"\nIn one word, the answer is: "
```

The model never writes anything. Everything state-independent — the prompt prefix, the option vectors,
the center — is computed once, so a call only pays for its own tokens.

## Every reading, and every setting

Four ways to read a model. A preset picks one; a call may override it.

| Reading | What it compares | Chosen by | Available on |
| --- | --- | --- | --- |
| **vector** (default) | cosine between the state's hidden state and each option's, `softmax(cos / tau)` | preset `method: "vector"` | any model |
| **letters** | the logits of the option letters (A, B, C…) at the next position | `method="letters"`, per call or per client | any model; a tuned head overrides it |
| **pointer** | a trained pointer head, at the delimiter tokens of the format in `decision.json` | preset `method: "pointer"` | decision models only |
| **tuned head** | a logistic head fitted by `autotune` on vector features | `autotune()` plus a `Context` | pins the reading to vectors |

A decision model may also **route by option count**: below the threshold the pointer head answers, above
it the vector reading does. The pointer reads every option on every call, so its cost grows with the option
list (65 ms at 3 options, 868 ms at 59) while the vector reading is flat; past ~30 options it stops earning
that latency. Routing is per question, so one call can mix both. Measured on `massive`, 59 options, same
weights: **868 ms → 77 ms at equal accuracy**.

| Setting | Where it lives | Default | What it does |
| --- | --- | --- | --- |
| `formulations` | preset | 2 built in | the prompts, and the layer each is read at |
| `tau` | preset | per model | temperature of `softmax(cos / tau)` |
| `center` | preset | `"options"` | what is subtracted before the cosine: `options`, `generic`, `none` |
| `one_word` | preset | — | layer and tau of the single-formulation variant (`one_word_only=True`) |
| `head.temperature` | `decision.json` | 1.954 for ours | divides the pointer logits; never changes an answer |
| `limits.max_state_tokens` / `max_branch_tokens` | `decision.json` | 384 / 1024 | where a too-long state or question is cut |
| `routing.above_options` | preset, measured by `jul models add` | measured | option count above which the vector reading answers |
| `routing` formulations, `tau`, `center` | preset, fitted by `jul models add` | — | the fallback reading, fitted on these very weights |
| `route_above=N` | per call | the model's value | overrides that threshold; `0` disables routing |
| `method=` | per call or client | `"vector"` | `vector` or `letters`; ignored on a pointer preset |
| `one_word_only=` | client | `False` | one formulation instead of two: faster, a little less accurate |
| `backend=` | client, or `JUL_BACKEND` | auto | `mlx` or `torch`; `onnx` only when asked |
| `JUL_BATCH_TOKENS` / `JUL_BATCH_SIZE` | environment | per backend | how many rows the backbone batches at once |
| `features=` | `autotune` | `"vector"` | what a tuned head reads: `vector`, `lexical` (TF-IDF) or `hybrid` |
| `formulations=` | `autotune` | the preset's | the prompts a question is read with, by name, per question if a dict |
| `JUL_HOME` | environment | `~/.jul` | where presets, contexts and calibration data live |
| `JUL_ONNX_MODEL` | environment | the export's graph | another graph for the onnx backend: a path or `s3://bucket/key` |
| `JUL_ONNX_THREADS` | environment | ONNX Runtime's | intra-op threads (on Lambda: one per whole vCPU) |
| `JUL_ONNX_MAX_TOKENS` / `JUL_ONNX_BATCH_TOKENS` | environment | 2048 / 2048 | longest row, and rows × longest per run, on onnx |

**Routing trades accuracy for speed, and the trade is not free.** Measured on massive by subsampling one
set's own options — so the option count is not confounded with the task — the pointer head is better at
*every* count, by about 5 points, while costing 105 ms at 3 options and 832 ms at 59. There is no count
above which it stops earning its answer; there is only a count above which its speed stops being worth
those points.

So `jul models add` **measures** the threshold rather than guessing one: it takes the smallest option
count at which the vector reading is three times faster, and records in the preset what that costs in
accuracy. A model where that never happens gets no routing at all. `--route-above N` sets it by hand and
`--no-routing` skips the whole fitting; the model's own `decision.json` may also carry a threshold.

**For maximum accuracy, pass `route_above=0`** on the call: every question then goes to the pointer head,
whatever its option count, and you pay the latency in the table above. The default is a compromise, and
`jul models add` prints exactly what it costs on the dev set it measured.
