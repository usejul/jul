"""OpenTelemetry export: metrics and events for every decision, sent to a collector you choose.

Off by default, and nothing is imported until it is switched on:

    pip install "jul[otel]"
    export JUL_ENABLE_TELEMETRY=1
    export OTEL_METRICS_EXPORTER=otlp        # otlp, console, none
    export OTEL_LOGS_EXPORTER=otlp           # otlp, console, none
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318

The names follow Claude Code's telemetry (https://code.claude.com/docs/en/monitoring-usage), so a
team that already collects `claude_code.*` can put `jul.*` next to it: same standard OTEL_* variables,
same metric/event split, same rule that content stays out unless a variable lets it in.

What leaves the process by default is metadata: model, backend, question type, the chosen answer
and its confidence, token count, latency, and the length of the state. The answer is an option key
the developer wrote (or a number), and it is what a routing dashboard needs, so it is on unless
JUL_OTEL_LOG_ANSWERS=0. Content is behind gates, all off by default:

    JUL_OTEL_LOG_STATE=1               the state itself (Claude Code: OTEL_LOG_USER_PROMPTS)
    JUL_OTEL_LOG_QUESTION_DETAILS=1    question names, instructions, option keys, error messages
                                       (Claude Code: OTEL_LOG_TOOL_DETAILS)
    JUL_OTEL_LOG_PROBABILITIES=1       the full distribution over options
                                       (Claude Code: OTEL_LOG_ASSISTANT_RESPONSES)
    JUL_OTEL_CONTENT_MAX_LENGTH=61440  truncation of content attributes
                                       (Claude Code: CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH)

The gates carry a JUL_ prefix on purpose: a machine that lets Claude Code log prompts must not start
shipping JuL states without being asked.

A telemetry failure never breaks a decision: every export error is swallowed after one warning.
"""

from __future__ import annotations

import atexit
import itertools
import logging
import os
import time
import uuid
from typing import Any, Mapping

log = logging.getLogger(__name__)

SERVICE_NAME = "jul"
METER_NAME = "com.usejul.jul"
DEFAULT_CONTENT_MAX_LENGTH = 61440
REDACTED = "<REDACTED>"

#: How the process was started: "sdk-py" for the library, "cli" for the `jul` command.
ENTRYPOINT = "sdk-py"

_UNSET = object()
_state: Any = _UNSET  # _Telemetry, or None when disabled
_warned = False


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _flag("JUL_ENABLE_TELEMETRY")


def _exporters(signal: str) -> list[str]:
    raw = os.environ.get(f"OTEL_{signal}_EXPORTER", "")
    return [e.strip().lower() for e in raw.split(",") if e.strip() and e.strip().lower() != "none"]


def _protocol(signal: str) -> str:
    return (os.environ.get(f"OTEL_EXPORTER_OTLP_{signal}_PROTOCOL")
            or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") or "http/protobuf").strip().lower()


def _warn_once(message: str, *args: Any) -> None:
    global _warned
    if not _warned:
        _warned = True
        log.warning(message, *args)


#: bounds of the jul.request.duration histogram, in ms
DURATION_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 30000, 60000, 120000, 300000)
#: bounds of the jul.decision.confidence histogram (0 to 1), finer at the top where escalation bars sit
CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)


class _Telemetry:
    """The providers, meter instruments and logger for one process."""

    def __init__(self, metric_readers: list | None = None, log_processors: list | None = None):
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk._logs import LoggerProvider
        from . import __version__

        # Resource.create also merges OTEL_RESOURCE_ATTRIBUTES (team.id, department...) and OTEL_SERVICE_NAME.
        resource = Resource.create({"service.name": SERVICE_NAME, "service.version": __version__})
        self.version = __version__

        readers = metric_readers if metric_readers is not None else self._metric_readers()
        processors = log_processors if log_processors is not None else self._log_processors()
        from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
        # OpenTelemetry's default buckets stop at 10 s: a slow local call (a cold first load is ~60 s) would
        # land in +Inf and read as 10 s on a dashboard. These cover a remote call (~300 ms) up to 5 min.
        # The default buckets (0, 5, 10, 25 ... 10000) put every confidence in "<= 5" too.
        views = [View(instrument_name="jul.request.duration",
                      aggregation=ExplicitBucketHistogramAggregation(DURATION_BUCKETS_MS)),
                 View(instrument_name="jul.decision.confidence",
                      aggregation=ExplicitBucketHistogramAggregation(CONFIDENCE_BUCKETS))]
        self.meter_provider = MeterProvider(resource=resource, metric_readers=readers, views=views)
        self.logger_provider = LoggerProvider(resource=resource)
        for p in processors:
            self.logger_provider.add_log_record_processor(p)

        meter = self.meter_provider.get_meter(METER_NAME, __version__)
        self.session_count = meter.create_counter(
            "jul.session.count", description="Clients that made their first call")
        self.decision_count = meter.create_counter(
            "jul.decision.count", description="Questions answered")
        self.token_usage = meter.create_counter(
            "jul.token.usage", unit="tokens", description="Input tokens of a call (local model, or the server's usage)")
        self.request_duration = meter.create_histogram(
            "jul.request.duration", unit="ms", description="Wall-clock time of one system_one call")
        self.decision_confidence = meter.create_histogram(
            "jul.decision.confidence", description="Confidence of each Choice and Score answer (0 to 1)")
        self.error_count = meter.create_counter(
            "jul.request.error.count", description="system_one calls that raised")
        self.logger = self.logger_provider.get_logger(METER_NAME, __version__)
        self.sequence = itertools.count()  # next() is atomic under the GIL: a client shared across threads

    # --- exporters from the environment -----------------------------------------------------------

    @staticmethod
    def _metric_readers() -> list:
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
        readers = []
        for name in _exporters("METRICS"):
            if name == "console":
                readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
            elif name == "otlp":
                if _protocol("METRICS") == "grpc":
                    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
                else:
                    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
                # OTEL_METRIC_EXPORT_INTERVAL is read by the reader itself (default 60000 ms).
                readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
            else:
                _warn_once("jul telemetry: unknown OTEL_METRICS_EXPORTER %r ignored", name)
        return readers

    @staticmethod
    def _log_processors() -> list:
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogRecordExporter
        interval = os.environ.get("OTEL_LOGS_EXPORT_INTERVAL")  # Claude Code's name for the batch delay
        delay = float(interval) if interval else None  # None: OTEL_BLRP_SCHEDULE_DELAY, else 5000 ms
        processors = []
        for name in _exporters("LOGS"):
            if name == "console":
                processors.append(BatchLogRecordProcessor(ConsoleLogRecordExporter(), schedule_delay_millis=delay))
            elif name == "otlp":
                if _protocol("LOGS") == "grpc":
                    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
                else:
                    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
                processors.append(BatchLogRecordProcessor(OTLPLogExporter(), schedule_delay_millis=delay))
            else:
                _warn_once("jul telemetry: unknown OTEL_LOGS_EXPORTER %r ignored", name)
        return processors

    def shutdown(self) -> None:
        """Flush and stop. Safe to call twice: reset() and atexit may both get here."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for provider in (self.meter_provider, self.logger_provider):
            try:
                provider.shutdown()
            except Exception:  # noqa: BLE001 - exiting anyway
                pass

    # --- emission -------------------------------------------------------------------------------

    def event(self, name: str, attributes: Mapping[str, Any]) -> None:
        attrs = {k: v for k, v in attributes.items() if v is not None}
        attrs["event.name"] = f"jul.{name}"     # the same value as the record's event_name
        attrs["event.timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
        attrs["event.sequence"] = next(self.sequence)
        self.logger.emit(event_name=f"jul.{name}", body=f"jul.{name}", attributes=attrs)


def configure(metric_readers: list | None = None, log_processors: list | None = None) -> None:
    """Build the providers now. Tests pass in-memory readers; everyone else relies on the environment."""
    global _state
    if _state not in (_UNSET, None):
        _state.shutdown()
    _state = _Telemetry(metric_readers, log_processors)


def reset() -> None:
    """Forget the providers, so the next call re-reads the environment."""
    global _state, _warned
    if _state not in (_UNSET, None):
        _state.shutdown()
    _state, _warned = _UNSET, False


def _get() -> _Telemetry | None:
    global _state
    if _state is _UNSET:
        if not enabled():
            _state = None
        else:
            try:
                _state = _Telemetry()
                atexit.register(_state.shutdown)  # flush before a short CLI run exits
            except ImportError:
                _warn_once('JUL_ENABLE_TELEMETRY is set but OpenTelemetry is missing: pip install "jul[otel]"')
                _state = None
    return _state


def active() -> bool:
    return _get() is not None


# --- what one call reports -------------------------------------------------------------------------

def _truncate(text: str) -> str:
    try:
        limit = int(os.environ.get("JUL_OTEL_CONTENT_MAX_LENGTH", DEFAULT_CONTENT_MAX_LENGTH))
    except ValueError:
        limit = DEFAULT_CONTENT_MAX_LENGTH
    if len(text) <= limit:
        return text
    marker = f"...[TRUNCATED {len(text) - limit} chars]"
    return text[:max(0, limit - len(marker))] + marker


def _answer_value(answer: Any) -> tuple[Any, float | None]:
    """(answer, confidence). A Noul's answer is already its probability of being true."""
    if hasattr(answer, "choice"):
        return answer.choice, answer.confidence
    if hasattr(answer, "score"):
        return answer.score, answer.confidence
    return getattr(answer, "noul", None), None


class Session:
    """One per client: carries `session.id` and counts the session on its first call."""

    def __init__(self) -> None:
        self.id = str(uuid.uuid4())
        self.started = False

    def standard(self, t: _Telemetry, model: str, backend: str | None) -> dict:
        return {"session.id": self.id, "app.version": t.version, "app.entrypoint": ENTRYPOINT,
                "model": model, "backend": backend}


def record_request(session: Session, *, model: str, backend: str | None, state_text: str,
                   questions: Mapping[str, Any], kinds: Mapping[str, str], methods: Mapping[str, str],
                   response: Any, duration_ms: float, context_name: str | None) -> None:
    t = _get()
    if t is None:
        return
    try:
        _record_request(t, session, model, backend, state_text, questions, kinds, methods, response,
                        duration_ms, context_name)
    except Exception as exc:  # noqa: BLE001 - telemetry must never break a decision
        _warn_once("jul telemetry: could not record a request (%s)", exc)


def _record_request(t, session, model, backend, state_text, questions, kinds, methods, response,
                    duration_ms, context_name) -> None:
    base = session.standard(t, model, backend)
    metric_base = {k: v for k, v in base.items() if v is not None}
    if not session.started:
        session.started = True
        t.session_count.add(1, metric_base)

    details = _flag("JUL_OTEL_LOG_QUESTION_DETAILS")
    tokens = response.usage.input_tokens
    t.token_usage.add(tokens, {**metric_base, "type": "input"})
    t.request_duration.record(duration_ms, metric_base)

    t.event("request", {
        **base,
        "request_id": response.request_id,
        "duration_ms": round(duration_ms, 3),
        "input_tokens": tokens,
        "question_count": len(questions),
        "state_length": len(state_text),
        "state": _truncate(state_text) if _flag("JUL_OTEL_LOG_STATE") else REDACTED,
        "context.name": context_name if details else None,
    })

    probabilities = _flag("JUL_OTEL_LOG_PROBABILITIES")
    answers = os.environ.get("JUL_OTEL_LOG_ANSWERS", "1").strip().lower() not in {"0", "false", "no", "off"}
    for position, (name, question) in enumerate(questions.items()):
        answer = response.answers[name]
        kind, method = kinds[name], methods.get(name)
        value, confidence = _answer_value(answer)
        t.decision_count.add(1, {**metric_base, "question.type": kind, "method": method or "unknown"})
        if confidence is not None:
            t.decision_confidence.record(confidence, {**metric_base, "question.type": kind})
        options = getattr(answer, "probabilities", None)
        attrs = {
            **base,
            "request_id": response.request_id,
            "question.index": position,
            "question.type": kind,
            "method": method,
            "answer": value if answers else REDACTED,
            "confidence": confidence,
            "option_count": len(options) if options else (2 if kind == "noul" else None),
        }
        if details:
            attrs["question.name"] = name
            attrs["question.instructions"] = _truncate(getattr(question, "instructions", "") or "")
            if options:
                attrs["question.options"] = list(options)
        if probabilities and options:
            attrs["probabilities"] = [float(options[k]) for k in options]
            if details:
                attrs["probabilities.keys"] = list(options)
        t.event("decision", attrs)


def record_error(session: Session, *, model: str, backend: str | None, error: BaseException,
                 duration_ms: float, question_count: int) -> None:
    t = _get()
    if t is None:
        return
    try:
        base = session.standard(t, model, backend)
        t.error_count.add(1, {**{k: v for k, v in base.items() if v is not None},
                              "error_type": type(error).__name__})
        t.event("request_error", {
            **base,
            "error_type": type(error).__name__,
            "error": _truncate(str(error)) if _flag("JUL_OTEL_LOG_QUESTION_DETAILS") else None,
            "duration_ms": round(duration_ms, 3),
            "question_count": question_count,
        })
    except Exception as exc:  # noqa: BLE001
        _warn_once("jul telemetry: could not record an error (%s)", exc)
