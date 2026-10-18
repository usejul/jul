# FAQ and troubleshooting

## Errors

### `No backend installed. Run: jul setup`

No mlx or torch in this environment. `jul setup` installs the right one; or `pip install "jul[mlx]"` on
Apple Silicon, `pip install "jul[torch]"` elsewhere. In Python the same thing raises `ImportError`.

### `MLX needs Apple Silicon; use --backend torch`

MLX only runs on M-series Macs. On Linux, Windows or an Intel Mac use `--backend torch` (or set
`JUL_BACKEND=torch`).

### `<model> is not downloaded for <backend> (...). Run: jul setup --model <model>`

The CLI does not download weights behind your back. `jul setup --model <model>` fetches them once and runs a
check. The Python API downloads them on the first call instead.

### `Unknown model '<name>'. Available: …`

Not a built-in preset and never added. `jul models` lists what you have;
`jul models add <name> --repo <hf-repo>` fits a new one.

### `'<name>' was fitted for mlx only` / `has no torch weights`

Presets are fitted per backend. Fit it for this backend: `jul models add <name> --backend torch` (the repo
is remembered for known names).

### `PyYAML is needed for YAML question files`

`pip install "jul[yaml]"`, or write the questions file as JSON (same structure).

### `A Score needs at least two levels, lowest first`

A `Score`'s `criteria` lists the levels; give at least two. On the CLI, one `-o` per level.

### `question 'x' has fewer than 2 labeled examples`

`autotune` found almost no line answering that question. Check the names in `answers` match the question
names, and that each line has a `state` (or `text`).

### `is a decision model with no vector reading fitted on its weights`

`autotune` trains a decision model's heads on its vector reading (the preset's `routing` block). A model added
with `jul models add --no-routing` has none: add it again without `--no-routing`.

### `only the vector method packs`

`jul pack` needs a vector preset (embedding models, encoders, with or without a cross model). Decision
models and contrastive heads are served with the client or `jul serve` instead.

### `jul serve` warns "bound to 0.0.0.0 WITHOUT an API key"

Anyone on the network can query it. Set `JUL_API_KEY` (or `--api-key`); callers then send
`Authorization: Bearer <key>` or `x-api-key: <key>`.

### `401` / `403` from `jul serve`

`403`: the server has a key and the request carries none. `401`: the request carries the wrong one.
`/health` needs the key too.

### `input cut: <reading> reads at most N tokens, M tokens past it were dropped`

A warning, not an error: the state was longer than that reading's limit, so its end was not read and the
answer comes from its beginning. `usage.truncated_tokens` says how much was dropped. Shorten the state (put
what decides first), pick a model with a longer limit, or pass `on_long="error"` to refuse such calls
instead. The limit of each reading: [input limits](models.md#input-limits).

### `the state is over the input limit of <reading> (N tokens, M more)`

`jul.truncation.InputTooLong`, raised with `on_long="error"` (HTTP 400 from `jul serve`). The call was not
answered. Shorten the state, or pass `on_long="cut"` to answer on its beginning.

## Accuracy

### It picks a wrong option with high confidence

Read the option descriptions as if you had never seen the task: the model compares the text with them.
Most fixes are there: [Writing good questions](questions.md). Then a [Context](tuning.md#context--what-the-data-looks-like)
with ~50 real texts, then [`autotune`](tuning.md#autotune--when-it-is-off-key) with labels.

### Every answer sits around the same probability

Options too close to each other, or a catch-all that matches everything. Make the options distinct, or
split one `Choice` into several questions.

### A yes/no about how two things relate is near 50%

"Is this a paraphrase?", "does it follow?", "was it paid on time?": an embedding model encodes the text and
the question apart, and answers those near chance. Use a preset with a cross model (the default model on
PyTorch, `jul-decision-wemm-4b-4bit` on MLX, `jul-decision-e5-small` on ONNX): see
[cross models](models.md#cross-models-reading-the-question-and-the-text-together).

### `autotune` says "not used"

The head did not beat zero-shot in cross-validation, so it was not activated: your zero-shot answers are
already as good as what those labels can teach. More labels (at least `max(20, 3 × options)`), `features="hybrid"`,
or better descriptions.

### The same model answers slightly differently on my Mac and my server

MLX presets read 4-bit weights, PyTorch reads bf16: the vectors differ a little (cosine ~0.95), and a close
call can flip. Tune and pack on the backend you deploy on.

### Is it any good in French, German, …?

The models answer other languages; every setting was fitted on English. On the decision bench the default
model scores 0.829 in French against 0.862 in English. Measure on your data.

## Speed and memory

### The first call takes seconds

The model loads on the first call (a few seconds for 2.6 GB). The next ones take tens of milliseconds.
`jul serve` loads it at startup (`--no-warmup` defers it).

### It is slower than the 55 ms in the README

55 ms is the default model on an M5 Max; an M4 Pro takes ~146 ms. Faster options, from the same API:
`f2llm-1.7b` (24 ms), `harrier-0.6b` (13 ms), `jul-decision-e5-small` on ONNX (4 ms). A question with
many options costs little (option vectors are cached), many questions cost one pass each unless they share
the `one_word` prompt.

### Out of memory

Only one model is held at a time. Lower `JUL_BATCH_TOKENS` / `JUL_BATCH_SIZE` for batch work, pick a
smaller model, or the 4-bit one on MLX. On ONNX, `JUL_ONNX_MAX_TOKENS` caps the longest row.

### Can it run without network?

Yes, once the weights are downloaded: set `HF_HUB_OFFLINE=1`. Nothing else calls out, except what you ask
for explicitly (an embeddings API backend, an escalation tier).

## Privacy

### Does my text leave the machine?

Not with a local backend (mlx, torch, onnx): the text never leaves the process, and `jul serve` does not log it
(only its first 200 characters at DEBUG level, off by default). It leaves only when you choose so: `--backend api` with a
hosted provider, or an escalation tier.

### Can I use it commercially?

JuL is Apache-2.0. Each model has its own license, on its Hugging Face page: check the one you ship.

## Getting help

Open an issue on [GitHub](https://github.com/usejul/jul/issues) with the version
(`python -c "import jul; print(jul.__version__)"`), the backend, the command or code, and the full error.
