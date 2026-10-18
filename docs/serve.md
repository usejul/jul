# Serving over HTTP: `jul serve`

For a caller that is not Python (a native app, a script in another language), `jul serve` wraps the
same client in a small HTTP server that speaks the Jev HTTP protocol. Any Jev client, SDK or plain
`curl`, talks to it by changing only its base URL: the test suite runs the official Python SDK
(`typesafe-sdk`) against it, `system_one`, `models.list()` and a wrong key included. It uses the standard library alone, so
`pip install jul` is enough.

```bash
jul serve                              # 127.0.0.1:8577, default model
jul serve --model minicpm5-2b --port 8577
```

The model is loaded and warmed up before the server accepts requests (`--no-warmup` defers it to the
first call). One model is held in memory and calls run one at a time.

## The protocol

```bash
curl -s http://127.0.0.1:8577/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": "I was charged twice for my subscription this month.",
    "questions": {
      "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
               "criteria": {"billing": "payments, invoices", "technical": "bugs, errors"}},
      "is_bug": {"type": "noul", "instructions": "Does this report a software bug?"}
    }
  }'
```

The response is `SystemOneResponse.as_dict()`, the shape of the Jev API:

```json
{"request_id": "...", "model": "wemm-4b-4bit",
 "usage": {"input_tokens": 41, "output_tokens": 0, "total_tokens": 41, "truncated_tokens": 0},
 "answers": {"team": {"type": "choice", "choice": "billing", "probabilities": {...}, "confidence": 0.93},
             "is_bug": {"type": "noul", "noul": 0.08}},
 "jul": {"latency_ms": 54.2}}
```

| Route | What it does |
| --- | --- |
| `POST /v1/systemone` | one state, a map of typed questions, one answer per question |
| `GET /v1/models` | `{"models": [{"name", "description", "release_date"}]}`: `jev-latest`, the built-in presets and the ones added with `jul models add` |
| `GET /health` | `{"status": "ok", "model": ..., "ready": true}` |
| `POST /v1/classify` | an alias of `/v1/systemone` |

Questions are written as in Jev: a `noul` takes optional `criteria` `{"true": ..., "false": ...}`, a
`choice` a map of option key to description (a description may be `null`), a `score` the ordered
levels, lowest first. A `jev-*` model name means the server's own model; a JuL model name picks that
one.

Errors follow Jev too: 400 `api_usage_error` for an unknown question type or model, 422 with a
`detail` list for a missing field (`state`, `questions`, a choice's `criteria`), 401 for a wrong key
and 403 for a missing one.

## What JuL adds

All optional, and ignored by a Jev client:

- in the request, `context` (a saved context, with its tuned heads), `method`, `route_above` and `on_long`,
  as in `system_one`; a choice's `criteria` may also be a plain list of keys, as in the library.
  `"on_long": "error"` refuses a state over the model's input limit with a 400 that names the reading and
  its limit, instead of answering on its beginning ([input limits](models.md#input-limits));
- in `usage`, `truncated_tokens`: the tokens dropped to fit a reading's limit (0 when nothing was cut). The
  server also prints each cut;
- in the response, a `jul` object with the latency;
- the `/health` route and the `/v1/classify` alias.

## Escalation: local first, a bigger decider when unsure

`--escalate-to` hands the questions answered below `--min-confidence` to another System One server,
and only those. Keys are read from the provider's usual variable, never from the command line:

| `--escalate-to` | Server | Key |
| --- | --- | --- |
| `typesafe` | Jev, `api.typesafe.ai` | `TYPESAFE_API_KEY` |
| `ollama`, `ollama:clef-flash` | Ollama on localhost (Nimble by default) | none |
| `cloudflare`, `cloudflare:clef` | Clef on Cloudflare Workers AI (`clef-flash` by default) | `CLOUDFLARE_API_TOKEN` (or `CLOUDFLARE_AUTH_TOKEN`), plus `CLOUDFLARE_ACCOUNT_ID` |
| a URL | any `/v1/systemone` server: Kev, another `jul serve`... | the variable named by `--escalate-key-env` |

```bash
TYPESAFE_API_KEY=... jul serve --escalate-to typesafe --min-confidence 0.8
jul serve --escalate-to ollama:nimble
CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... jul serve --escalate-to cloudflare:clef-flash
```

Workers AI serves Clef at its own route (`/accounts/{id}/ai/run/@cf/cloudflare/clef-flash`) with the same
body as Jev; JuL posts there and unwraps Cloudflare's `result` envelope. Clef's image input, a Clef
extension to the protocol, is not sent. A missing account or token stops `jul serve` at startup.

The response says, per question, which tier answered:

```json
"jul": {"escalation": {"team": {"tried": ["local", "remote"], "tier": "remote", "confidence": 0.91, "met_bar": true}}}
```

To choose the server and the bar before serving, measure the pair on your own labeled rows:
`jul bench test.jsonl --models accurate --escalate-to typesafe --min-confidence 0.8` gives the accuracy of
the cascade and the share of rows that would go to Jev (see [Bench on your own data](cli.md#bench-on-your-own-data)).

A Choice or Score counts its `confidence`, a Noul `max(noul, 1 - noul)`. The tier escalated to keeps
the question even when it is less sure (two models' confidences are not comparable); `met_bar` says
whether the kept answer cleared the bar. A tier that fails is skipped and its error recorded; a
malformed request (an unknown model, a Score with one level) is refused with a 400, never escalated.
Redirects are not followed, so a key never reaches another host. In
Python, `jul.Escalation` chains any number of tiers, with a bar per question if needed.

The state of an escalated question leaves the machine. The bar is a policy, not a measurement: check
what it lets through on labeled examples of your own.

## Security and privacy

The server binds to 127.0.0.1 by default: local only, the same on-device promise as the library. To
bind it elsewhere, use `--host` and set a key so that only authorised callers get through:

```bash
JUL_API_KEY=secret jul serve --host 0.0.0.0        # or: jul serve --api-key secret
curl ... -H "Authorization: Bearer $JUL_API_KEY" http://<host>:8577/v1/systemone
```

With a key set, every request (`/health` included) must carry it, as `Authorization: Bearer` like
Jev or as `x-api-key`. Without one, `jul serve` warns when it is bound beyond loopback. The request
`state` is never logged, only the question count and the latency.
