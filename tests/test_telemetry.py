"""OpenTelemetry export: off unless asked, content gated, and a telemetry failure never breaks a call.

No model is loaded: `_system_one` is replaced by a stub that returns a fixed response, so these tests
check what is exported, not how the answer was computed.
"""

import subprocess
import sys

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk._logs import export as _logs_export  # noqa: E402
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402

import jul  # noqa: E402

# 1.38 (the floor of jul[otel]) names it InMemoryLogExporter; 1.39 added InMemoryLogRecordExporter
InMemoryLogRecordExporter = getattr(_logs_export, "InMemoryLogRecordExporter", None) or _logs_export.InMemoryLogExporter
from jul import Choice, Noul, Score, TypeSafeClient, telemetry  # noqa: E402
from jul.types import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage  # noqa: E402

STATE = {"ticket": "I was charged twice, card ending 4242"}
QUESTIONS = {
    "team": Choice(instructions="Which team should handle this ticket?",
                   criteria={"billing": "payments", "technical": "bugs"}),
    "is_bug": Noul(instructions="Does the message report a software bug?"),
    "anger": Score(instructions="How angry is the customer?", criteria=["Calm", "Angry"]),
}


def _canned(self, state, questions, context, model, method, route_above, methods):
    methods.update({"team": "head", "is_bug": "vector", "anger": "letters"})
    return SystemOneResponse(
        answers={"team": ChoiceAnswer(choice="billing", probabilities={"billing": 0.9, "technical": 0.1},
                                      confidence=0.9),
                 "is_bug": NoulAnswer(noul=0.12),
                 "anger": ScoreAnswer(score=0.3, legend={"0": "Calm", "1": "Angry"},
                                      probabilities={"0": 0.7, "1": 0.3}, confidence=0.4)},
        model="minicpm5-2b", usage=Usage(input_tokens=42), request_id="req-1")


@pytest.fixture
def otel(monkeypatch):
    """Telemetry switched on, exporting to memory. Returns (logs exporter, metric reader)."""
    for name in ("JUL_OTEL_LOG_STATE", "JUL_OTEL_LOG_QUESTION_DETAILS", "JUL_OTEL_LOG_PROBABILITIES",
                 "JUL_OTEL_LOG_ANSWERS", "JUL_OTEL_CONTENT_MAX_LENGTH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JUL_ENABLE_TELEMETRY", "1")
    monkeypatch.setattr(TypeSafeClient, "_system_one", _canned)
    logs, metrics = InMemoryLogRecordExporter(), InMemoryMetricReader()
    telemetry.configure(metric_readers=[metrics], log_processors=[SimpleLogRecordProcessor(logs)])
    yield logs, metrics
    telemetry.reset()


def _events(logs):
    return [dict(r.log_record.attributes) for r in logs.get_finished_logs()]


def _metrics(reader):
    out = {}
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                out[m.name] = [dict(p.attributes) | {"_value": getattr(p, "value", None),
                                                     "_count": getattr(p, "count", None)}
                               for p in m.data.data_points]
    return out


def _ask(client=None):
    client = client or TypeSafeClient(model="minicpm5-2b")
    return client.system_one(state=STATE, questions=QUESTIONS)


def test_one_request_event_and_one_decision_event_per_question(otel):
    logs, _ = otel
    _ask()
    events = _events(logs)
    assert [e["event.name"] for e in events] == ["jul.request", "jul.decision", "jul.decision", "jul.decision"]
    assert [e["event.sequence"] for e in events] == [0, 1, 2, 3]
    request = events[0]
    assert request["model"] == "minicpm5-2b"
    assert request["input_tokens"] == 42
    assert request["question_count"] == 3
    assert request["request_id"] == "req-1"
    assert request["duration_ms"] >= 0
    assert len({e["session.id"] for e in events}) == 1


def test_decisions_carry_type_method_answer_and_confidence(otel):
    logs, _ = otel
    _ask()
    team, is_bug, anger = _events(logs)[1:]
    assert (team["question.type"], team["method"], team["answer"], team["confidence"]) == \
        ("choice", "head", "billing", 0.9)
    assert (is_bug["question.type"], is_bug["answer"]) == ("noul", 0.12)
    assert "confidence" not in is_bug                # a Noul's answer is already a probability
    assert (anger["question.type"], anger["method"], anger["answer"]) == ("score", "letters", 0.3)


def test_content_is_redacted_by_default(otel):
    logs, _ = otel
    _ask()
    events = _events(logs)
    assert events[0]["state"] == "<REDACTED>"
    assert events[0]["state_length"] == len(jul.types.serialize_state(STATE))
    blob = repr(events)
    for secret in ("4242", "charged twice", "Which team", "is_bug", "technical"):
        assert secret not in blob, secret


def test_each_gate_lets_in_only_its_own_content(otel, monkeypatch):
    logs, _ = otel
    monkeypatch.setenv("JUL_OTEL_LOG_STATE", "1")
    _ask()
    events = _events(logs)
    assert "4242" in events[0]["state"]
    assert "question.name" not in events[1] and "probabilities" not in events[1]

    logs.clear()
    monkeypatch.delenv("JUL_OTEL_LOG_STATE")
    monkeypatch.setenv("JUL_OTEL_LOG_QUESTION_DETAILS", "1")
    monkeypatch.setenv("JUL_OTEL_LOG_PROBABILITIES", "1")
    _ask()
    events = _events(logs)
    assert events[0]["state"] == "<REDACTED>"
    team = events[1]
    assert team["question.name"] == "team"
    assert team["question.instructions"] == "Which team should handle this ticket?"
    assert list(team["question.options"]) == ["billing", "technical"]
    assert list(team["probabilities"]) == [0.9, 0.1]


def test_answers_can_be_withheld(otel, monkeypatch):
    logs, _ = otel
    monkeypatch.setenv("JUL_OTEL_LOG_ANSWERS", "0")
    _ask()
    assert {e["answer"] for e in _events(logs)[1:]} == {"<REDACTED>"}


def test_content_is_truncated(otel, monkeypatch):
    logs, _ = otel
    monkeypatch.setenv("JUL_OTEL_LOG_STATE", "1")
    monkeypatch.setenv("JUL_OTEL_CONTENT_MAX_LENGTH", "40")
    _ask()
    state = _events(logs)[0]["state"]
    assert len(state) == 40 and "TRUNCATED" in state


def test_metrics(otel):
    _, reader = otel
    client = TypeSafeClient(model="minicpm5-2b")
    _ask(client)
    _ask(client)
    m = _metrics(reader)
    assert m["jul.session.count"][0]["_value"] == 1           # once per client, not per call
    assert m["jul.token.usage"][0]["_value"] == 84
    assert m["jul.token.usage"][0]["type"] == "input"
    assert m["jul.request.duration"][0]["_count"] == 2
    by_type = {p["question.type"]: p["_value"] for p in m["jul.decision.count"]}
    assert by_type == {"choice": 2, "noul": 2, "score": 2}
    assert {p["question.type"] for p in m["jul.decision.confidence"]} == {"choice", "score"}
    assert "session.id" in m["jul.token.usage"][0]


def test_an_error_is_counted_and_still_raised(otel, monkeypatch):
    logs, reader = otel

    def boom(*args, **kwargs):
        raise RuntimeError("model file missing at /secret/path")

    monkeypatch.setattr(TypeSafeClient, "_system_one", boom)
    with pytest.raises(RuntimeError):
        _ask()
    event = _events(logs)[0]
    assert event["event.name"] == "jul.request_error"
    assert event["error_type"] == "RuntimeError"
    assert "error" not in event                                   # the message can hold paths or data
    assert _metrics(reader)["jul.request.error.count"][0]["_value"] == 1


def test_a_broken_exporter_never_breaks_a_decision(otel, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("collector down")

    monkeypatch.setattr(telemetry._Telemetry, "event", broken)
    assert _ask().choices["team"].choice == "billing"


def test_off_by_default_and_opentelemetry_is_not_even_imported():
    code = ("import sys, os; os.environ.pop('JUL_ENABLE_TELEMETRY', None); "
            "sys.path[:0] = ['lib', 'cli']; import jul; from jul import telemetry; "
            "assert not telemetry.active(); "
            "assert not [m for m in sys.modules if m.startswith('opentelemetry')], 'imported'")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(__import__("pathlib").Path(__file__).parents[1]))


def test_a_reading_that_does_not_say_its_method_is_reported_under_the_presets(otel, monkeypatch):
    """A model family whose branch does not fill `methods` (letter-readout, #41) is not "unknown"."""
    import dataclasses

    logs, reader = otel

    def silent(self, state, questions, context, model, method, route_above, methods):
        return _canned(self, state, questions, context, model, method, route_above, {})

    monkeypatch.setattr(TypeSafeClient, "_system_one", silent)
    client = TypeSafeClient(model="minicpm5-2b")
    client._preset = dataclasses.replace(client._preset, method="letter-readout")
    _ask(client)
    decisions = [e for e in _events(logs) if e["event.name"] == "jul.decision"]
    assert {e["method"] for e in decisions} == {"letter-readout"}
    assert {p["method"] for p in _metrics(reader)["jul.decision.count"]} == {"letter-readout"}


def test_event_sequence_is_unique_across_threads(otel):
    import threading

    logs, _ = otel
    client = TypeSafeClient(model="minicpm5-2b")
    threads = [threading.Thread(target=lambda: [_ask(client) for _ in range(10)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seq = [e["event.sequence"] for e in _events(logs)]
    assert len(seq) == 4 * 10 * 4 and len(set(seq)) == len(seq)


def test_a_letter_readout_model_is_reported_as_such(otel, monkeypatch):
    """The engine.reader branch (#41) runs for real here, on a fake reader: method="letter-readout"."""
    import numpy as np

    logs, reader = otel
    monkeypatch.undo()                 # the real _system_one, not the canned one
    monkeypatch.setenv("JUL_ENABLE_TELEMETRY", "1")

    class Reader:
        def logits(self, state, items):
            return [np.zeros(len(options)) for _, _, options in items], 7

    class Engine:
        reader, pointer, contrastive = Reader(), None, None

    client = TypeSafeClient(model="minicpm5-2b")
    monkeypatch.setattr(client, "_engine_for", lambda model: Engine())
    response = _ask(client)
    assert response.usage.input_tokens == 7
    decisions = [e for e in _events(logs) if e["event.name"] == "jul.decision"]
    assert len(decisions) == 3 and {e["method"] for e in decisions} == {"letter-readout"}
    assert {p["method"] for p in _metrics(reader)["jul.decision.count"]} == {"letter-readout"}


# --- the duration histogram -------------------------------------------------------------------------

def test_a_slow_call_lands_in_a_bucket_below_inf(otel, monkeypatch):
    """OpenTelemetry's default buckets stop at 10 s: a 30 s local call must not read as 10 s."""
    logs, reader = otel
    clock = iter([0.0, 30.0])
    monkeypatch.setattr(jul.client.time, "perf_counter", lambda: next(clock))
    _ask()
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == "jul.request.duration":
                    (point,) = m.data.data_points
    bounds = list(point.explicit_bounds)
    assert bounds == list(telemetry.DURATION_BUCKETS_MS) and bounds[-1] == 300000
    landed = next(i for i, c in enumerate(point.bucket_counts) if c)
    assert landed < len(bounds) and bounds[landed - 1] < 30000 <= bounds[landed]


# --- remote deciders (SystemOneHTTP) ------------------------------------------------------------------

@pytest.fixture
def server():
    """A System One server on localhost: answers as `jev-1.13.0`, or HTTP 401 when the key is "bad"."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.headers.get("Authorization") == "Bearer bad":
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b'{"error": "bad key"}')
                return
            answers = {}
            for name, q in body["questions"].items():
                if q["type"] == "choice":
                    answers[name] = {"type": "choice", "choice": "billing",
                                     "probabilities": {"billing": 0.8, "technical": 0.2}, "confidence": 0.8}
                elif q["type"] == "noul":
                    answers[name] = {"type": "noul", "noul": 0.3}
                else:
                    answers[name] = {"type": "score", "score": 0.6, "legend": {"0": "Calm", "1": "Angry"},
                                     "probabilities": {"0": 0.4, "1": 0.6}, "confidence": 0.6}
            out = json.dumps({"model": "jev-1.13.0", "request_id": "remote-1", "answers": answers,
                              "usage": {"input_tokens": 123}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_a_remote_decider_reports_its_calls(otel, server):
    from jul import SystemOneHTTP

    logs, reader = otel
    response = SystemOneHTTP(server, model="jev-latest", api_key="k").system_one(state=STATE, questions=QUESTIONS)
    assert response.model == "jev-1.13.0"
    events = _events(logs)
    (request,) = [e for e in events if e["event.name"] == "jul.request"]
    assert (request["model"], request["backend"], request["input_tokens"]) == ("jev-1.13.0", "remote", 123)
    assert request["state"] == "<REDACTED>", "same default redaction as a local call"
    decisions = [e for e in events if e["event.name"] == "jul.decision"]
    assert len(decisions) == 3 and {e["method"] for e in decisions} == {"remote"}
    assert {e["model"] for e in decisions} == {"jev-1.13.0"}
    m = _metrics(reader)
    assert {(p["model"], p["backend"], p["method"]) for p in m["jul.decision.count"]} == \
        {("jev-1.13.0", "remote", "remote")}
    assert [p["_value"] for p in m["jul.token.usage"]] == [123]
    assert m["jul.session.count"][0]["_value"] == 1


def test_a_remote_http_error_is_a_request_error(otel, server):
    from jul import SystemOneHTTP
    from jul.escalate import RemoteError

    logs, reader = otel
    with pytest.raises(RemoteError, match="HTTP 401"):
        SystemOneHTTP(server, model="jev-latest", api_key="bad").system_one(state=STATE, questions=QUESTIONS)
    (error,) = [e for e in _events(logs) if e["event.name"] == "jul.request_error"]
    assert (error["model"], error["backend"], error["error_type"]) == ("jev-latest", "remote", "RemoteError")
    assert "error" not in error, "the message stays out without JUL_OTEL_LOG_QUESTION_DETAILS"
    assert _metrics(reader)["jul.request.error.count"][0]["error_type"] == "RemoteError"


def test_escalation_records_each_tier_under_its_own_model(otel, server, monkeypatch):
    """Local tier unsure on everything: every question goes on to the remote tier. Two requests, two models."""
    from jul import Escalation, SystemOneHTTP

    logs, reader = otel
    local = TypeSafeClient(model="minicpm5-2b")
    client = Escalation(tiers=[("local", local), ("jev", SystemOneHTTP(server, api_key="k"))], min_confidence=0.99)
    client.system_one(state=STATE, questions=QUESTIONS)
    requests = [e for e in _events(logs) if e["event.name"] == "jul.request"]
    assert {(e["model"], e.get("backend")) for e in requests} == {("minicpm5-2b", local._backend),
                                                                  ("jev-1.13.0", "remote")}
    methods = {(p["model"], p["method"]) for p in _metrics(reader)["jul.decision.count"]}
    assert ("jev-1.13.0", "remote") in methods and ("minicpm5-2b", "head") in methods


def test_a_remote_decider_with_telemetry_off_sends_nothing_and_answers_the_same(server, monkeypatch):
    from jul import SystemOneHTTP

    monkeypatch.delenv("JUL_ENABLE_TELEMETRY", raising=False)
    telemetry.reset()
    try:
        assert not telemetry.active()
        response = SystemOneHTTP(server, api_key="k").system_one(state=STATE, questions=QUESTIONS)
        assert response.model == "jev-1.13.0" and response.answers["team"].choice == "billing"
    finally:
        telemetry.reset()


def _histogram(reader, name):
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == name:
                    return m.data.data_points
    return []


def test_confidences_land_in_buckets_between_0_and_1(otel, monkeypatch):
    """The default buckets put every confidence in "<= 5": a 0.35 must land in "<= 0.4", a 0.92 in "<= 0.95"."""
    logs, reader = otel

    def two(self, state, questions, context, model, method, route_above, methods):
        methods.update(dict.fromkeys(questions, "vector"))
        return SystemOneResponse(
            answers={"team": ChoiceAnswer(choice="billing", probabilities={"billing": 0.35, "technical": 0.65},
                                          confidence=0.35),
                     "anger": ScoreAnswer(score=0.9, legend={"0": "Calm", "1": "Angry"},
                                          probabilities={"0": 0.08, "1": 0.92}, confidence=0.92)},
            model="minicpm5-2b", usage=Usage(input_tokens=1), request_id="r")

    monkeypatch.setattr(TypeSafeClient, "_system_one", two)
    TypeSafeClient(model="minicpm5-2b").system_one(
        state=STATE, questions={"team": QUESTIONS["team"], "anger": QUESTIONS["anger"]})
    landed = {}
    for point in _histogram(reader, "jul.decision.confidence"):
        bounds = list(point.explicit_bounds)
        assert bounds == list(telemetry.CONFIDENCE_BUCKETS)
        i = next(i for i, c in enumerate(point.bucket_counts) if c)
        landed[point.attributes["question.type"]] = bounds[i] if i < len(bounds) else float("inf")
    assert landed == {"choice": 0.4, "score": 0.95}


def test_laya_reports_the_torch_backend(otel, monkeypatch):
    """Laya runs on its own PyTorch runtime: its metrics carry backend=torch like every other model's."""
    from jul.laya_model import LayaModel

    logs, reader = otel
    monkeypatch.undo()   # the real TypeSafeClient._system_one, which hands Laya models to LayaModel
    monkeypatch.setenv("JUL_ENABLE_TELEMETRY", "1")
    monkeypatch.setattr(LayaModel, "system_one", lambda self, state, questions, **_: _canned(
        None, state, questions, None, None, None, None, {}))
    client = TypeSafeClient(model="laya")
    _ask(client)
    m = _metrics(reader)
    assert {p["backend"] for p in m["jul.decision.count"]} == {"torch"}
    assert {p["method"] for p in m["jul.decision.count"]} == {"laya"}
    (request,) = [e for e in _events(logs) if e["event.name"] == "jul.request"]
    assert request["backend"] == "torch"


def test_event_name_attribute_is_the_records_event_name(otel, monkeypatch):
    """Grafana maps the attribute and the record's event_name onto one label: they must agree (jul.*)."""
    logs, _ = otel
    _ask()

    def boom(*args, **kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr(TypeSafeClient, "_system_one", boom)
    with pytest.raises(RuntimeError):
        _ask()
    records = [r.log_record for r in logs.get_finished_logs()]
    seen = set()
    for record in records:
        name = record.attributes["event.name"]
        assert name == record.event_name and name.startswith("jul.")
        seen.add(name)
    assert seen == {"jul.request", "jul.decision", "jul.request_error"}
