# The hub

JuL is model-agnostic. Your code asks typed questions through one API; behind it you plug the model,
the runtime and the provider that fit your constraints, and change them without touching that code.

```text
                         ┌──────────────────────────────────────────┐
  Python  ─ TypeSafeClient│                                          │
  shell   ─ jul ask / run │   one API: Choice · Noul · Score         │
  any lang ─ jul serve    │   one answer shape: Jev's System One     │
  Lambda  ─ Bundle        │                                          │
                         └───────────────────┬──────────────────────┘
                                             │
           ┌──────────────┬──────────────────┼──────────────────┬──────────────────┐
           ▼              ▼                  ▼                  ▼                  ▼
     embedding LLMs   decision models     encoders         embeddings APIs   System One servers
     WeMM, F2LLM,     jul-decision-*,     e5, BERT,        Ollama, OpenAI,   Jev, Clef (Cloudflare),
     Qwen3-Emb,       minicpm5-2b-        XLM-R            Mistral, Voyage,  Nimble (Ollama), Kev,
     Harrier, any     decision, CLM-8B                     any OpenAI-       another jul serve
     HF decoder                                            compatible        (escalation tiers)
           │              │                  │                  │                  │
           └──────── mlx · torch · onnx ─────┘                api              HTTP
```

Five families, four backends, one call. The rest of this page says what each brings and how to pick.

## Models you run

Anything on the Hugging Face Hub, or in a local directory, that JuL can load on one of its backends.
`jul models add` fits it once on the development sets (layer, center, temperature) and from then on
`--model <name>` uses it everywhere: Python, CLI, server, bundles.

| Family | What it is | Examples | Best for |
| --- | --- | --- | --- |
| **Embedding LLMs** | decoders fine-tuned for embeddings, read at a middle layer | `wemm-4b-4bit` (built in), `wemm-9b`, `f2llm-1.7b`, `qwen3-embedding-0.6b`, `harrier-0.6b` | sorting one text into options, zero-shot |
| **Any decoder** | a general LLM read the same way | `minicpm5-2b` (built in), `bitnet-2b`, any instruct model | trying a model you already use |
| **Decision models** | trained to answer typed questions, bring their own format in a `decision.json` | `jul-decision-minicpm5-2b`, Kev checkpoints, [Strands Decider](models.md#strands-decider-pointer-head-at-answer) | questions that read two things together |
| **Letter-readout models** | LLMs trained to answer with an option letter after their own prompt, read as their runtime reads them | [JevK5, plumb-4b, Quyet, spark-s1](models.md#letter-readout-decision-models-jevk5-plumb-quyet-spark-s1) | open decision models published with their own runtime |
| **Laya** | a decision model on its own runtime (`laya` package): JuL hands it the questions | `laya`, `laya:multilingual` ([Laya](models.md#laya)) | Laya's answers through JuL's API, as a tier or next to the others |
| **Cross models (adapters)** | LoRA adapters that read the question and the text together, on the same weights | `jul-decision-wemm-4b` (the default), `jul-decision-wemm-4b-4bit` | yes/no and choices (the default), scores (`-4bit`), one model in memory |
| **Encoders** | small bidirectional models, read as their sentence embedding | `e5-small`, `jul-decision-e5-small` | milliseconds, CPU only, Lambda |
| **Contrastive heads** | projection heads on a frozen backbone | `clm-8b` (CLM's heads on Qwen3-8B), your own with `--train-heads` | reusing heads trained elsewhere |

```bash
jul models add harrier-0.6b --repo majentik/harrier-oss-v1-0.6b-MLX-4bit     # an embedding LLM
jul models add my-model --repo org/Some-Instruct-3B                          # any decoder
jul models add jul-decision-minicpm5-2b --repo usejul/jul-decision-minicpm5-2b-mlx-4bit   # a decision model
jul models add clm-8b --repo Contrastive-LM/CLM-v0.1-8B                       # contrastive heads
jul models add jevk5 --repo alibiserikbay/JevK5                              # a letter-readout model
jul models                                                                   # what you have
```

What `jul models add` recognizes by itself: a repo with a `decision.json` is a decision model (a
`"method": "letters"` one, or the runtime config of JevK5, Quyet or spark-s1, is a letter-readout model), one
with a CLM `config.json` or a `contrastive.json` gets contrastive heads, one carrying a `cross/` folder gets its
cross model attached. Everything else is fitted as a vector preset. A Strands Decider archive is converted
first (`scripts/convert_strands_decider.py`); Laya needs no `add` (`--model laya`). The 18 models we measured, with their
scores: [Models](models.md#every-model-measured).

## Backends

The same preset name can be fitted on several backends; the backend decides where it runs.

| Backend | Install | Runs on | Reads | Use it for |
| --- | --- | --- | --- | --- |
| `mlx` | `pip install "jul[mlx]"` | Apple Silicon | every reading | a Mac, the default there |
| `torch` | `pip install "jul[torch]"` | CUDA, CPU, MPS | every reading | Linux, Windows, GPUs, the default elsewhere |
| `onnx` | `pip install "jul[onnx]"` | CPU, no torch | vector, cross (encoders) | deploying: Lambda, containers, 225 MB packages |
| `api` | nothing extra | wherever the API is | vector only | no weights at all, a model you already serve |

Pick one with `backend=` in Python, `--backend` on the CLI, or `JUL_BACKEND` for everything. Never picked by
default: `onnx` (it reads a model exported for it) and `api` (it calls a server). Details:
[Installation](installation.md#backend-device-and-batching).

**Vectors are tied to their weights.** MLX presets are 4-bit, PyTorch loads bf16: the same model reads
slightly different vectors on each (cosine ~0.95). Contexts, heads and bundles are therefore keyed per
backend: a head trained on MLX is never applied to PyTorch vectors. Tune and pack on the backend you deploy on.

## Embeddings APIs

When the model already runs somewhere (an Ollama on the box, a hosted provider), JuL reads its embeddings
over HTTP instead of loading weights. `jul models add` fits the center and temperature as for a local
model.

| `--repo` | Endpoint | Key |
| --- | --- | --- |
| `ollama:MODEL` | `$OLLAMA_HOST` or `http://localhost:11434` | none |
| `openai:MODEL` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `mistral:MODEL` | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` |
| `voyage:MODEL` | `https://api.voyageai.com/v1` | `VOYAGE_API_KEY` |
| `http(s)://host/v1#MODEL` | any OpenAI-compatible `/embeddings` | none |

```bash
jul models add qwen3-emb --repo ollama:qwen3-embedding:0.6b --backend api
jul ask choice "Which team?" -o billing -o technical --state "I was charged twice" --model qwen3-emb --backend api
```

Trade-offs: one layer (the API's output), no prefix cache, no logits, and with a hosted provider the state
leaves the machine. See [Models › Embeddings APIs](models.md#embeddings-apis---backend-api).

## System One servers

JuL speaks Jev's HTTP protocol both ways.

**As a server**, `jul serve` exposes any of the models above at `POST /v1/systemone`: any Jev client, SDK or
`curl`, talks to it by changing its base URL. See [Serving over HTTP](serve.md).

**As a client**, `SystemOneHTTP` calls any server that speaks it, and `Escalation` chains them: answer
locally, and send only the questions answered below a confidence bar to the next tier.

| Tier | `--escalate-to` | Key |
| --- | --- | --- |
| Jev (TypeSafe) | `typesafe` | `TYPESAFE_API_KEY` |
| Nimble on Ollama | `ollama`, `ollama:MODEL` | none |
| Clef on Cloudflare Workers AI | `cloudflare`, `cloudflare:clef` | `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` |
| Kev, another `jul serve`, any server | its URL | `--escalate-key-env VAR` |

The same targets go in `jul bench --models` (a URL as `URL#model`, its key in `--remote-key-env`), to measure
them against JuL's models on your own rows before choosing: see [Bench on your own data](cli.md#bench-on-your-own-data).

```python
from jul import TypeSafeClient, Escalation, SystemOneHTTP

client = Escalation(tiers=[("local", TypeSafeClient()),
                           ("ollama", SystemOneHTTP("http://localhost:11434", model="nimble"))],
                    min_confidence=0.8)
r = client.system_one(state="…", questions={…})
r.escalation["team"]     # {"tried": ["local"], "tier": "local", "confidence": 0.93, "met_bar": True}
```

See [Serving › Escalation](serve.md#escalation-local-first-a-bigger-decider-when-unsure) and
[Python API › Escalation](python-api.md#escalation).

## Choosing

Start from your constraint, not from the model.

| Your constraint | Start with | Why |
| --- | --- | --- |
| best accuracy on a laptop, no labels | the default, `jul-decision-wemm-4b` | 0.857 zero-shot on Jev's benchmark (Jev 0.753), 55 ms, 2.6 GB on MLX |
| speed, many options, batch jobs | `f2llm-1.7b` or `harrier-0.6b` | 24 ms / 1 GB, 13 ms / 0.31 GB; tune them with labels |
| CPU only, serverless | `jul-decision-e5-small` on onnx + `autotune` | 4 ms per text, 90 MB, $0.59 per million in Lambda |
| yes/no about how two things relate | a preset with a cross model (the default on PyTorch) | the vectors alone answer those near chance |
| no GPU, no weights, a model already served | an embeddings API | nothing to download |
| local first, but some questions are hard | `Escalation` to Jev, Clef or Nimble | only the unsure ones leave |
| a model you already trust | `jul models add` | one command, measured on dev sets |

Then measure on a few hundred of your own texts: the benchmark tells you where to start, your data tells
you where to stop. [Benchmarks](benchmarks.md) has every number and how it was measured.
