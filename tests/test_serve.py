"""`jul serve` speaks the Jev HTTP protocol. A fake client stands in for the model, so no weights load."""

import json
import threading
import urllib.error
import urllib.request

import pytest
from jul.types import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage

from jul_cli import serve as S


class FakeClient:
    model = "fake-model"

    def __init__(self):
        self.calls = []

    def system_one(self, state, questions, model=None, **extras):
        self.calls.append({"state": state, "questions": questions, "model": model, **extras})
        if model == "nope":
            raise ValueError("Unknown model 'nope'")
        answers = {}
        for name, q in questions.items():
            kind = type(q).__name__
            if kind == "Choice":
                keys = list(q.criteria)
                answers[name] = ChoiceAnswer(keys[0], {k: 1.0 if k == keys[0] else 0.0 for k in keys}, 1.0)
            elif kind == "Noul":
                answers[name] = NoulAnswer(0.9)
            else:
                answers[name] = ScoreAnswer(1.0, {"0": "low", "1": "high"}, {"0": 0.0, "1": 1.0}, 1.0)
        return SystemOneResponse(answers=answers, model=model or self.model, usage=Usage(input_tokens=5),
                                 request_id="rid")


@pytest.fixture
def server(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(S, "_client", fake)
    monkeypatch.setattr(S, "_api_key", None)
    srv = S.make_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", fake
    srv.shutdown()
    srv.server_close()


def call(url, body=None, headers=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})},
                                 method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


JEV_REQUEST = {
    "model": "jev-latest",
    "state": "I was charged twice for my annual plan this morning.",
    "questions": {
        "wants_refund": {"type": "noul", "instructions": "Is the customer asking for money back?",
                         "criteria": {"true": "asks for a refund", "false": "does not"}},
        "queue": {"type": "choice", "instructions": "Which queue?",
                  "criteria": {"billing": "money problems", "technical": None}},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "Today"]},
    },
}


def test_jev_request_gets_a_jev_response(server):
    url, fake = server
    status, out = call(url + "/v1/systemone", JEV_REQUEST)
    assert status == 200
    assert set(out) >= {"model", "answers", "usage", "request_id"}
    assert out["answers"]["wants_refund"] == {"type": "noul", "noul": 0.9}
    assert out["answers"]["queue"]["type"] == "choice" and out["answers"]["queue"]["choice"] == "billing"
    assert out["answers"]["urgency"]["type"] == "score" and "legend" in out["answers"]["urgency"]
    assert out["usage"]["input_tokens"] == 5 and out["usage"]["output_tokens"] == 0
    assert "latency_ms" in out["jul"]
    # a jev-* model name means the server's own model
    assert fake.calls[-1]["model"] is None


def test_noul_criteria_and_null_descriptions_reach_the_library(server):
    url, fake = server
    call(url + "/v1/systemone", JEV_REQUEST)
    qs = fake.calls[-1]["questions"]
    assert qs["wants_refund"].criteria.true == "asks for a refund"
    assert qs["queue"].criteria == {"billing": "money problems", "technical": ""}


def test_choice_list_keeps_its_keys(server):
    url, fake = server
    body = {"state": "x", "questions": {"q": {"type": "choice", "criteria": ["billing", "tech"]}}}
    status, out = call(url + "/v1/systemone", body)
    assert status == 200 and out["answers"]["q"]["choice"] == "billing"


def test_jul_extras_are_passed_through(server):
    url, fake = server
    body = {**JEV_REQUEST, "model": "minicpm5-2b", "context": "tickets", "route_above": 0}
    assert call(url + "/v1/systemone", body)[0] == 200
    assert fake.calls[-1]["model"] == "minicpm5-2b"
    assert fake.calls[-1]["context"] == "tickets" and fake.calls[-1]["route_above"] == 0


def test_classify_alias(server):
    url, _ = server
    assert call(url + "/v1/classify", JEV_REQUEST)[0] == 200


@pytest.mark.parametrize("body,status", [
    ({"state": "x", "questions": {"q": {"type": "boolean"}}}, 400),
    ({"state": "x", "model": "nope", "questions": {"q": {"type": "noul"}}}, 400),
    ({"state": "x"}, 422),
    ({"questions": {"q": {"type": "noul"}}}, 422),
    ({"state": "x", "questions": {"q": {"type": "choice"}}}, 422),
])
def test_errors_follow_jev(server, body, status):
    url, _ = server
    code, out = call(url + "/v1/systemone", body)
    assert code == status
    assert ("detail" in out) if status == 422 else (out["error"]["type"] == "api_usage_error")


def test_models_and_health(server):
    url, _ = server
    status, out = call(url + "/v1/models")
    names = [m["name"] for m in out["models"]]
    assert status == 200 and "jev-latest" in names and "minicpm5-2b" in names
    assert all(set(m) == {"name", "description", "release_date"} for m in out["models"])
    status, out = call(url + "/health")
    assert status == 200 and out == {"status": "ok", "model": "fake-model", "ready": True}


def test_api_key_bearer_or_x_api_key(server, monkeypatch):
    url, _ = server
    monkeypatch.setattr(S, "_api_key", "k3y")
    assert call(url + "/v1/systemone", JEV_REQUEST)[0] == 403
    assert call(url + "/v1/systemone", JEV_REQUEST, {"Authorization": "Bearer wrong"})[0] == 401
    assert call(url + "/v1/systemone", JEV_REQUEST, {"Authorization": "Bearer k3y"})[0] == 200
    assert call(url + "/v1/systemone", JEV_REQUEST, {"x-api-key": "k3y"})[0] == 200
    assert call(url + "/health")[0] == 403


def test_official_jev_sdk_talks_to_it(server, monkeypatch):
    """The official Python SDK, pointed at jul serve by base_url only. Skipped when it is not installed."""
    T = pytest.importorskip("typesafe_sdk")
    url, _ = server
    monkeypatch.setattr(S, "_api_key", "k3y")
    client = T.TypeSafeClient(api_key="k3y", base_url=url)
    r = client.system_one(
        state={"ticket": "I was charged twice"},
        questions={"team": T.Choice(instructions="Which team?", criteria={"billing": "payments", "technical": None}),
                   "is_bug": T.Noul(instructions="Bug?"),
                   "frustration": T.Score(instructions="How frustrated?", criteria=["low", "high"])},
        model="jev-latest",
    )
    assert r.choices["team"].choice == "billing" and r.nouls["is_bug"].noul == 0.9
    assert r.scores["frustration"].score == 1.0
    assert "jev-latest" in [m.name for m in client.models.list().models]
    with pytest.raises(T.TypeSafeAuthenticationError):
        T.TypeSafeClient(api_key="wrong", base_url=url).system_one(state="x", questions={"q": T.Noul(instructions="?")})


def test_a_single_option_choice_is_accepted(server):
    """Jev sets no minimum on a choice's options (a CLICK action has one)."""
    url, fake = server
    body = {"state": "click the button", "questions": {
        "action": {"type": "choice", "instructions": "Which action?", "criteria": {"click": "press the button"}}}}
    status, out = call(url + "/v1/systemone", body)
    assert status == 200 and out["answers"]["action"]["choice"] == "click"
    assert list(fake.calls[-1]["questions"]["action"].criteria) == ["click"]


def test_escalation_trace_survives_the_latency(monkeypatch):
    """With --escalate-to, the server's client is an Escalation; its trace stays next to the latency."""
    from jul import Escalation

    fake = Escalation([("local", FakeClient())], min_confidence=0.5)
    monkeypatch.setattr(S, "_client", fake)
    out = S.classify(JEV_REQUEST)
    assert out["jul"]["escalation"]["queue"]["tier"] == "local" and "latency_ms" in out["jul"]
