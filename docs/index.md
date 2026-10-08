# JuL documentation

JuL is a hub for **typed decisions**. You ask a question about a text: pick an option (`Choice`), yes or no
(`Noul`), a level on a scale (`Score`). You get back an answer and a probability per option. One API, the
one of Jev's SDK, and behind it whatever model fits your constraints: a 4B model on your Mac, a 90 MB encoder
in an AWS Lambda, an embeddings API, or a bigger decider only for the questions the small one is unsure about.

```python
from jul import TypeSafeClient, Choice

client = TypeSafeClient()
r = client.system_one(
    state="I was charged twice this month.",
    questions={"team": Choice(instructions="Which team should handle this?",
                              criteria={"billing": "payments, refunds", "technical": "bugs, errors",
                                        "sales": "pricing, plans"})},
)
r.choices["team"].choice           # "billing"
r.choices["team"].probabilities    # {"billing": 0.88, "technical": 0.11, "sales": 0.01}
```

## Where to start

<div class="cards" markdown="1">

**New here?** [Quickstart](quickstart.md): install, a first answer, a first batch, in ten minutes. Then
[Concepts](concepts.md): the six words that explain the rest.

**Already on Jev?** Change `from typesafe_sdk import` to `from jul import`: same classes, same calls, same
response. [Python API](python-api.md) lists what is identical and what JuL adds.

**Picking a model?** [The hub](hub.md) maps every model, backend and provider JuL plugs into, and how to
choose. [Models](models.md) has the details, [Benchmarks](benchmarks.md) the numbers.

**Going to production?** [Serving over HTTP](serve.md), [Deploying a fixed need](deployment.md),
[AWS Lambda](aws-lambda.md), and [Configuration](configuration.md) for every variable.

</div>

## Every page

| I want to… | Read |
| --- | --- |
| install it, pick MLX, PyTorch or ONNX | [Installation](installation.md) |
| understand state, question, option, reading, context, head | [Concepts](concepts.md) |
| see every model, backend and provider JuL connects | [The hub](hub.md) |
| write options the model gets right | [Writing good questions](questions.md) |
| answer a file of texts from the shell | [Command line](cli.md) |
| improve accuracy on my own data | [Adapting to your data](tuning.md) |
| call it from another language | [Serving over HTTP](serve.md) |
| send unsure answers to a bigger model | [Escalation](serve.md#escalation-local-first-a-bigger-decider-when-unsure) |
| ship a fixed set of questions | [Deploying a fixed need](deployment.md), [AWS Lambda](aws-lambda.md) |
| use my own model | [Models › Adding a model](models.md#adding-a-model) |
| look up a class, argument or field | [Python API](python-api.md) |
| look up a command or flag | [CLI reference](cli-reference.md) |
| look up an environment variable or a file on disk | [Configuration](configuration.md) |
| send metrics and events to my OpenTelemetry collector | [Telemetry](telemetry.md) |
| fix an error | [FAQ and troubleshooting](troubleshooting.md) |
| look up a word | [Glossary](glossary.md) |

Keys on every page: <code>/</code> search, <code>D</code> docs home, <code>Q</code> quickstart, <code>A</code>
Python API, <code>L</code> CLI reference, <code>G</code> GitHub.
