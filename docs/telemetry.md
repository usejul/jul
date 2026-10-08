# Telemetry (OpenTelemetry)

Off by default. When switched on, every `system_one` call sends metrics and events to the OTLP
collector you name, and nowhere else. The names follow
[Claude Code's telemetry](https://code.claude.com/docs/en/monitoring-usage), so a team that already
collects `claude_code.*` can put `jul.*` next to it with the same collector and dashboards.

```bash
pip install "jul[otel]"
export JUL_ENABLE_TELEMETRY=1
export OTEL_METRICS_EXPORTER=otlp          # otlp, console, none
export OTEL_LOGS_EXPORTER=otlp             # otlp, console, none
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318
export OTEL_RESOURCE_ATTRIBUTES="team.id=risk,department=eng"   # optional, lands on every record
```

The standard `OTEL_*` variables apply (per-signal endpoints, headers, `OTEL_METRIC_EXPORT_INTERVAL`).
The default protocol is `http/protobuf`; for `grpc`, also install `opentelemetry-exporter-otlp-proto-grpc`.

Metrics, meter `com.usejul.jul`, resource `service.name=jul`:

| Metric | What it counts | Attributes |
|---|---|---|
| `jul.session.count` | clients that made their first call | standard |
| `jul.decision.count` | questions answered | `question.type`, `method` |
| `jul.token.usage` | input tokens of a call: read by the local model, or the server's `usage` for a remote one (unit `tokens`) | `type=input` |
| `jul.request.duration` | one `system_one` call, histogram (unit `ms`) | standard |
| `jul.decision.confidence` | confidence of each Choice and Score answer, histogram | `question.type` |
| `jul.request.error.count` | calls that raised | `error_type` |

The standard attributes are `session.id`, `app.version`, `app.entrypoint` (`sdk-py` or `cli`),
`model` and `backend`. `method` is how the question was actually read: `vector`, `letters`, `cross`, `contrastive`, `head`
(a tuned head from `autotune`), `pointer` (a decision model), `letter-readout` (a letter-readout decision
model), `laya` or `remote` (a `SystemOneHTTP` server).

Events (OTel logs), each with `event.name` (the same value as the record's `event_name`, e.g. `jul.decision`),
`event.timestamp` and `event.sequence`:

- `jul.request`: one per call. `request_id`, `duration_ms`, `input_tokens`, `question_count`,
  `state_length`, `state`.
- `jul.decision`: one per question, joined to its call by `request_id`. `question.index`,
  `question.type`, `method`, `answer`, `confidence`, `option_count`.
- `jul.request_error`: `error_type`, `duration_ms`, `question_count`.

## Choosing what leaves

By default only metadata leaves the process: the state is sent as `<REDACTED>` with its length, and
question names and instructions are not sent at all. Each kind of content has its own switch:

| Variable | Default | Lets in | Claude Code equivalent |
|---|---|---|---|
| `JUL_OTEL_LOG_STATE` | off | the state itself | `OTEL_LOG_USER_PROMPTS` |
| `JUL_OTEL_LOG_QUESTION_DETAILS` | off | question name, instructions, option keys, context name, error message | `OTEL_LOG_TOOL_DETAILS` |
| `JUL_OTEL_LOG_PROBABILITIES` | off | the full distribution over options | `OTEL_LOG_ASSISTANT_RESPONSES` |
| `JUL_OTEL_LOG_ANSWERS` | **on** | the chosen option key, Noul probability or score | none |
| `JUL_OTEL_CONTENT_MAX_LENGTH` | 61440 | truncation of content attributes | `CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH` |

The answer is on by default because it is an option key you wrote, and it is what a routing or
audit dashboard needs; set `JUL_OTEL_LOG_ANSWERS=0` to send `<REDACTED>` instead. The switches have
their own `JUL_` names so that a machine configured to log Claude Code prompts does not start
sending JuL states without being asked.

A telemetry failure (collector down, OpenTelemetry missing) logs one warning and never fails the
decision. With telemetry off, OpenTelemetry is not imported. `jul[otel]` needs OpenTelemetry 1.38 or later
(`Logger.emit(event_name=)`); CI runs the telemetry tests at that floor.

A session is one client: `session.id` is drawn when a `TypeSafeClient` (or a `SystemOneHTTP`) makes its
first call, and `jul.session.count` counts clients. Under `jul serve` every HTTP request goes through the
server's one client, so a server process is one session, whoever calls it.

Remote deciders report too. A `SystemOneHTTP` call (Jev via `typesafe`, Ollama, Clef on Workers AI, a remote
`jul serve`) is recorded like a local one, with `model` the model the server answered with (e.g.
`jev-1.13.0`), `backend` and `method` set to `remote`, `input_tokens` from the server's `usage`, and the same
switches and redaction. Under `Escalation` / `--escalate-to`, each tier's call is recorded under its own
model, so a dashboard can put the local model and the remote one side by side. An HTTP error is a
`jul.request_error` (`error_type=RemoteError`).

The two histograms have explicit buckets, since OpenTelemetry's default ones (`0, 5, 10, 25 … 10000`) fit
neither:

| Histogram | Buckets | Why |
|---|---|---|
| `jul.request.duration` | `5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 30000, 60000, 120000, 300000` ms | a slow local call (a cold first load takes about a minute) would read as 10 s |
| `jul.decision.confidence` | `0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0` | every confidence would land in `≤ 5`; finer at the top, where escalation bars sit |

`backend` is always set: `torch`, `mlx` or `onnx` for a local model, `torch` for Laya (its own PyTorch
runtime), `remote` for a `SystemOneHTTP` server.
