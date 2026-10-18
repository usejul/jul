# Python API

Everything `from jul import …` exposes, with every argument. The names, fields and shapes of the first
two sections are Jev's SDK's: code written for `typesafe_sdk` runs unchanged. The others are what JuL adds.

```python
from jul import (TypeSafeClient, AsyncTypeSafeClient,            # the client
                 Choice, Noul, NoulCriteria, Score,              # questions
                 SystemOneResponse, ChoiceAnswer, NoulAnswer, ScoreAnswer, Usage,   # answers
                 Context, TuningReport,                          # adapting to your data
                 Escalation, EscalatedResponse, SystemOneHTTP,   # the hub: other deciders
                 Bundle, pack,                                   # deploying a fixed need
                 Preset, PRESETS, __version__)
```

## TypeSafeClient

```python
TypeSafeClient(model=None, context=None, method=None, one_word_only=False, backend=None,
               context_home=None, api_key=None, base_url=None, timeout=None, max_retries=None,
               on_long=None, **ignored)
```

| Argument | Default | What it does |
| --- | --- | --- |
| `model` | `"jul-decision-wemm-4b"` | a preset name, an alias (`accurate`, `fast`), a model added with `jul models add`, or `"laya"` / `"laya:<checkpoint>"` ([Laya](models.md#laya)) |
| `context` | `None` | a `Context`, or the name of a saved one, used by every call |
| `method` | per question type | force a reading for every call: `"vector"`, `"cross"`, `"letters"` |
| `one_word_only` | `False` | one prompt instead of two: faster, a little less accurate |
| `backend` | `$JUL_BACKEND`, else `mlx` on Apple Silicon, else `torch` | `"mlx"`, `"torch"`, `"onnx"`, `"api"` |
| `context_home` | `$JUL_HOME/contexts` | where named contexts are read and saved |
| `on_long` | `$JUL_ON_LONG`, else `"cut"` | a state over a reading's limit: `"cut"` answers on what was read, `"error"` refuses the call ([input limits](models.md#input-limits)) |
| `api_key`, `base_url`, `timeout`, `max_retries`, … | — | Jev's remote arguments: accepted and ignored |

Constructing a client loads nothing and needs no backend installed. The model loads on the first call and
stays in memory; only one model is held at a time (asking another one drops the first).

| Member | What it does |
| --- | --- |
| `system_one(state, questions, context=None, model=None, method=None, route_above=None, on_long=None, **ignored)` | answers every question about one state, returns a `SystemOneResponse` |
| `autotune(context, questions, labeled, model=None, save=True, features="vector", formulations=None)` | trains a head per question, returns `{name: TuningReport}`; see [below](#autotune) |
| `model` | the preset name in use |
| `backend` | the backend, resolved on first access |
| `context` | the client's default context |
| `close()` | releases the model; also `with TypeSafeClient() as client:` |

### system_one

| Argument | What it is |
| --- | --- |
| `state` | the text: a `str` as is, anything else serialized as JSON (a dict, a list, a dataclass…) |
| `questions` | `{name: Choice | Noul | Score}`, at least one |
| `context` | overrides the client's context for this call |
| `model` | another preset for this call (loads it, drops the previous one) |
| `method` | overrides the reading for this call |
| `route_above` | decision models only: above this many options, read as vectors; `0` disables it |
| `on_long` | overrides the client's `on_long` for this call: `"cut"` or `"error"` |
| `response_model`, `retry`, `extra_body`, … | Jev's arguments: accepted and ignored |

Raises `ValueError` for an empty `questions`, an unknown `model=` passed to the call (the constructor raises
it for its own) or a `Score` with fewer than two levels, `jul.truncation.InputTooLong` (a `ValueError`) for a
state over a reading's limit with `on_long="error"`, `TypeError` for a question that is not one of the three types.

## AsyncTypeSafeClient

The same constructor and members, awaitable: `await client.system_one(...)`, `await client.autotune(...)`,
`await client.close()`, `async with AsyncTypeSafeClient() as client:`. The work runs in a worker thread, so
the event loop keeps turning.

## Questions

### Choice

```python
Choice(instructions="", criteria={})
```

Pick one option. `criteria` is `{key: description}`, or a plain list of keys (then the key is also what the
model reads). At least one option. See [Writing good questions](questions.md).

### Noul

```python
Noul(instructions="", criteria=None)
```

Yes or no. `criteria` is optional: `NoulCriteria(true=…, false=…)`, or a dict with `true`/`false` (or
`yes`/`no`) keys. Without it, yes and no read as `"Yes."` and `"No."`.

### Score

```python
Score(instructions="", criteria=[])
```

A level on a scale. `criteria` lists the level descriptions, **lowest first**, at least two.

## Answers

### SystemOneResponse

| Field | Type | What it is |
| --- | --- | --- |
| `answers` | `dict[str, ChoiceAnswer | NoulAnswer | ScoreAnswer]` | every answer, by question name, in the order asked |
| `choices` | `dict[str, ChoiceAnswer]` | the `Choice` answers only |
| `nouls` | `dict[str, NoulAnswer]` | the `Noul` answers only |
| `scores` | `dict[str, ScoreAnswer]` | the `Score` answers only |
| `model` | `str` | the preset that answered |
| `usage` | `Usage` | tokens read |
| `request_id` | `str` | a UUID per call |
| `as_dict()` | `dict` | the Jev HTTP response shape, ready for `json.dumps` |

### ChoiceAnswer, NoulAnswer, ScoreAnswer

| Class | Fields |
| --- | --- |
| `ChoiceAnswer` | `choice` (the key with the highest probability), `probabilities` (`{key: p}`, summing to 1), `confidence` (the chosen key's probability) |
| `NoulAnswer` | `noul`: the probability of yes, in [0, 1] |
| `ScoreAnswer` | `score` (the expected level, from 0 to n−1), `probabilities` (`{"0": p, "1": p, …}`), `legend` (`{"0": "Calm", …}`), `confidence` (1 − the spread of the distribution, scaled to [0, 1]: 1 when one level takes everything) |

Probabilities are rounded to 4 decimals. Each answer has `as_dict()`, which adds `"type"`.

### Usage

`input_tokens` (tokens fed to the model for this call), `output_tokens` (always 0: nothing is generated),
`total_tokens`, and `truncated_tokens`: the largest cut of the call, the tokens a reading dropped to fit its
limit (0 when everything was read; each cut is also logged on the `jul.truncation` logger, see
[input limits](models.md#input-limits)). With `--backend api`, the counts are characters.

## Context

```python
Context(description="", examples=[], labeled=[], name=None, use_description=False)
```

| Argument | What it does |
| --- | --- |
| `description` | one sentence about the data; reaches the prompts only with `use_description=True` (measured harmful on average) |
| `examples` | unlabeled texts of the task: their mean vector becomes the center. Warns under 10; ~50 is the sweet spot |
| `name` | needed to save it and to reuse it by name |
| `use_description` | put `description` in the prompts |

| Member | What it does |
| --- | --- |
| `save(name=None, home=None)` | writes `~/.jul/contexts/<name>/`: centers, heads, calibrations, examples |
| `Context.load(name, home=None)` | reads it back (`FileNotFoundError` if absent) |
| `Context.list_saved(home=None)` | the saved names |
| `Context.delete(name, home=None)` | removes it, returns whether it existed |

Anywhere a context is expected, its saved name works too: `client.system_one(state, q, context="tickets")`.
See [Adapting to your data](tuning.md#context--what-the-data-looks-like).

## autotune

```python
reports = client.autotune(context, questions, labeled, model=None, save=True,
                          features="vector", formulations=None)
```

| Argument | What it is |
| --- | --- |
| `context` | a `Context` or a name: where the heads are stored (created if new) |
| `questions` | the same `{name: question}` you will ask |
| `labeled` | `[(state, {question_name: answer}), …]`: the option key for a `Choice`, `True`/`False` for a `Noul`, the level index for a `Score` |
| `features` | what the head reads: `"vector"`, `"lexical"` (TF-IDF, needs `jul[tune]`), `"hybrid"` (both) |
| `formulations` | the prompts read, by name (`"one_word"`, `"question_options"`, `"question"`), a list for all questions or a dict per question |
| `save` | save the context when it has a name |

Returns `{name: TuningReport}`. A head is **activated only if it beats zero-shot** in cross-validation;
otherwise only a calibration is kept. Then `client.system_one(..., context=...)` uses it with no other
change. Not available on decision models (pointer reading). See [`autotune`](tuning.md#autotune--when-it-is-off-key).

### TuningReport

`question`, `n_examples`, `n_evaluated`, `n_options`, `zero_shot_accuracy`, `head_accuracy` (`None` if no
head was trained), `activated`, `reason`, `temperature`, `features`. `print(report)` gives the summary,
`as_dict()` the fields.

## Escalation

```python
Escalation(tiers, min_confidence=0.8, default=0.8)
```

Several deciders in order. The first answers every question; a question answered below the bar goes to the
next tier, with the other unsure ones only, and so on.

| Argument | What it is |
| --- | --- |
| `tiers` | `[(name, decider), …]`, names unique. A decider is anything with `system_one(state, questions)`: a `TypeSafeClient`, a `SystemOneHTTP`, another `Escalation` |
| `min_confidence` | one bar for every question, or `{question_name: bar}` |
| `default` | the bar of a question missing from that dict |

How sure an answer is: `confidence` for a `Choice` or a `Score`, `max(noul, 1 − noul)` for a `Noul`. The
answer kept is the last tier's that answered, even if less sure (two models' confidences do not compare).
A tier that fails (network, HTTP error) is skipped and its error recorded; a malformed request
(`ValueError`, `TypeError`) is raised, never sent on. `system_one` returns an `EscalatedResponse`: a
`SystemOneResponse` plus `escalation`, per question:

```python
{"tried": ["local", "jev"], "tier": "jev", "confidence": 0.91, "met_bar": True}
```

### SystemOneHTTP

```python
SystemOneHTTP(base_url, model="jev-latest", api_key=None, timeout=30.0, exact_url=False)
```

A decider over HTTP: any server speaking `POST /v1/systemone` (Jev at `https://api.typesafe.ai`, Ollama's
Nimble at `http://localhost:11434`, Kev, another `jul serve`). `base_url` is the server root, with or without
`/v1`. Redirects are never followed, so a key never reaches another host. Raises `jul.escalate.RemoteError`
on any failure. `jul.escalate.remote_tier("typesafe" | "ollama:nimble" | "cloudflare:clef-flash" | URL)` and
`jul.escalate.cloudflare_tier()` build one with the provider's usual key variable.

## Bundle and pack

```python
pack(client, questions, out, context=None, model=None, reading="auto") -> Path
Bundle.load(path, backend=None, model=None, cross_model=None) -> Bundle
```

`pack` computes everything that does not depend on the state (prompts, option vectors, centers, heads)
into a directory. `Bundle` answers with the state as its only input:

| Member | What it does |
| --- | --- |
| `bundle.system_one(state)` | a `SystemOneResponse`, as the client would answer |
| `bundle.system_one_batch(states)` | the same for many states, read in batches |
| `bundle.question_names` | the packed questions |
| `bundle.models` | the models it needs: `["vector"]`, `["cross"]` or both |

`reading="vector"` packs every question on the vectors (one model to ship). `Bundle.load(model=...)` points
to the weights if they moved. Vector presets only. See [Deploying a fixed need](deployment.md).

## Presets

`PRESETS` is the dict of built-in presets (`jul-decision-wemm-4b`, `wemm-4b-4bit`, `minicpm5-2b`), each a
`Preset` with its `name`, `repos` per backend, `formulations`, `tau`, `center`, `latency_ms`, `quality`.
Presets you add live in `~/.jul/presets/<name>@<backend>.json`; `jul models` lists them all.
`jul.presets.resolve(name, backend)` returns the one a client would use.

## Errors you may catch

| Exception | When |
| --- | --- |
| `ValueError` | unknown model or backend, empty questions, a `Score` with one level, a model not fitted for this backend |
| `jul.truncation.InputTooLong` | a `ValueError`: the state is over a reading's limit and `on_long="error"`; `.reading`, `.limit`, `.over` say which and by how much |
| `TypeError` | a question that is not `Choice`, `Noul` or `Score`, a context of the wrong type |
| `ImportError` | no backend installed (`pip install "jul[mlx]"` or `"jul[torch]"`) |
| `FileNotFoundError` | a context name that was never saved |
| `jul.escalate.RemoteError` | a remote tier failed (only raised by `SystemOneHTTP` used alone) |
| `jul.backends.api.EmbeddingsError` | an embeddings API failed or refused |
