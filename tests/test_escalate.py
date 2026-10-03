"""Escalation: unsure answers go to the next decider, and the trace says who answered."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from jul import Choice, Escalation, Noul, NoulCriteria, Score, SystemOneHTTP
from jul.escalate import certainty
from jul.types import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage


class Fixed:
    """A decider answering every question with the same confidence; records what it was asked."""

    def __init__(self, confidence, choice="a", noul=None, fail=False):
        self.confidence, self.choice, self.noul, self.fail = confidence, choice, noul, fail
        self.asked = []

    def system_one(self, state, questions, **kwargs):
        self.asked.append((sorted(questions), kwargs))
        if self.fail:
            raise ConnectionError("down")
        answers = {}
        for name, q in questions.items():
            if isinstance(q, Noul):
                answers[name] = NoulAnswer(self.noul if self.noul is not None else self.confidence)
            elif isinstance(q, Score):
                answers[name] = ScoreAnswer(1.0, {}, {"0": 0.0, "1": 1.0}, self.confidence)
            else:
                answers[name] = ChoiceAnswer(self.choice, {self.choice: self.confidence}, self.confidence)
        return SystemOneResponse(answers=answers, model="fixed", usage=Usage(input_tokens=3), request_id="r")


Q = {"team": Choice("Which team?", {"a": "A", "b": "B"}), "urgent": Noul("Urgent?")}


def test_a_sure_answer_stays_local():
    local, remote = Fixed(0.95), Fixed(0.99, choice="b")
    r = Escalation([("local", local), ("remote", remote)], min_confidence=0.8).system_one("s", Q)
    assert r.choices["team"].choice == "a"
    assert remote.asked == []
    assert r.escalation["team"] == {"tried": ["local"], "tier": "local", "confidence": 0.95, "met_bar": True}


def test_only_unsure_questions_escalate():
    local = Fixed(0.6, noul=0.97)            # unsure on the choice, sure the noul is true
    remote = Fixed(0.9, choice="b")
    r = Escalation([("local", local), ("remote", remote)], min_confidence=0.8).system_one("s", Q)
    assert remote.asked[0][0] == ["team"]
    assert r.choices["team"].choice == "b" and r.escalation["team"]["tier"] == "remote"
    assert r.escalation["urgent"]["tier"] == "local"
    assert r.usage.input_tokens == 6


def test_noul_certainty_is_the_likelier_side():
    assert certainty(NoulAnswer(0.03)) == pytest.approx(0.97)
    assert certainty(NoulAnswer(0.5)) == 0.5


def test_the_last_tier_answers_even_when_unsure_and_says_so():
    r = Escalation([("local", Fixed(0.5)), ("remote", Fixed(0.6))], min_confidence=0.8).system_one("s", Q)
    assert r.escalation["team"]["tier"] == "remote" and r.escalation["team"]["met_bar"] is False


def test_the_escalated_tier_wins_even_less_confident():
    """Confidences of two models are not comparable: the tier escalated to is the one trusted."""
    r = Escalation([("local", Fixed(0.7)), ("remote", Fixed(0.4, choice="b"))], min_confidence=0.8).system_one("s", Q)
    assert r.choices["team"].choice == "b" and r.escalation["team"]["tried"] == ["local", "remote"]


def test_a_failing_last_tier_keeps_the_earlier_answer():
    r = Escalation([("local", Fixed(0.7)), ("jev", Fixed(0, fail=True))], min_confidence=0.8).system_one("s", Q)
    assert r.choices["team"].choice == "a" and r.escalation["team"]["tier"] == "local"
    assert r.escalation["team"]["met_bar"] is False


def test_a_failing_tier_is_skipped_and_recorded():
    r = Escalation([("local", Fixed(0.5)), ("jev", Fixed(0, fail=True)), ("kev", Fixed(0.9, choice="b"))],
                   min_confidence=0.8).system_one("s", Q)
    assert r.choices["team"].choice == "b"
    assert r.escalation["team"]["tried"] == ["local", "jev", "kev"]
    assert r.escalation["team"]["errors"] == ["jev: ConnectionError"]


def test_no_tier_at_all_raises():
    with pytest.raises(RuntimeError, match="no tier answered"):
        Escalation([("a", Fixed(0, fail=True))]).system_one("s", Q)


def test_per_question_bars():
    local, remote = Fixed(0.85, noul=0.85), Fixed(0.99)
    Escalation([("local", local), ("remote", remote)],
               min_confidence={"urgent": 0.95}, default=0.8).system_one("s", Q)
    assert remote.asked[0][0] == ["urgent"]


def test_local_options_reach_the_first_tier_only():
    local, remote = Fixed(0.5), Fixed(0.9)
    Escalation([("local", local), ("remote", remote)]).system_one("s", Q, method="letters")
    assert local.asked[0][1] == {"method": "letters"} and remote.asked[0][1] == {}


def test_response_serializes_with_the_trace():
    out = Escalation([("local", Fixed(0.95))]).system_one("s", Q).as_dict()
    assert out["jul"]["escalation"]["team"]["tier"] == "local"
    assert json.dumps(out)


# --- SystemOneHTTP against a fake Jev server -------------------------------------------------------

@pytest.fixture
def jev_server():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen["path"], seen["auth"] = self.path, self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = {"model": "jev-1.13.0", "usage": {"input_tokens": 42, "output_tokens": 0},
                   "answers": {"team": {"type": "choice", "choice": "b", "probabilities": {"a": 0.1, "b": 0.9},
                                        "confidence": 0.88},
                               "urgent": {"type": "noul", "noul": 0.12},
                               "level": {"type": "score", "score": 1.7, "legend": {"0": "low", "1": "mid", "2": "high"},
                                         "probabilities": {"0": 0.05, "1": 0.2, "2": 0.75}, "confidence": 0.6}}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    srv.shutdown()
    srv.server_close()


def test_http_tier_speaks_the_jev_protocol(jev_server):
    url, seen = jev_server
    questions = {"team": Choice("Which team?", {"a": "A", "b": ""}),
                 "urgent": Noul("Urgent?", NoulCriteria(true="now", false="later")),
                 "level": Score("How bad?", ["low", "mid", "high"])}
    r = SystemOneHTTP(url + "/v1", api_key="k").system_one({"ticket": "x"}, questions)
    assert seen["path"] == "/v1/systemone" and seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "jev-latest" and seen["body"]["state"] == {"ticket": "x"}
    assert seen["body"]["questions"]["team"] == {"type": "choice", "instructions": "Which team?",
                                                 "criteria": {"a": "A", "b": None}}
    assert seen["body"]["questions"]["urgent"]["criteria"] == {"true": "now", "false": "later"}
    assert seen["body"]["questions"]["level"]["criteria"] == ["low", "mid", "high"]
    assert r.choices["team"].choice == "b" and r.nouls["urgent"].noul == 0.12
    assert r.scores["level"].confidence == 0.6 and r.usage.input_tokens == 42 and r.model == "jev-1.13.0"


def test_a_noul_without_criteria_sends_none(jev_server):
    url, seen = jev_server
    SystemOneHTTP(url).system_one("s", {"urgent": Noul("Urgent?")})
    assert "criteria" not in seen["body"]["questions"]["urgent"]


def test_local_then_http(jev_server):
    url, _ = jev_server
    r = Escalation([("local", Fixed(0.5)), ("jev", SystemOneHTTP(url))]).system_one("s", {"team": Q["team"]})
    assert r.choices["team"].choice == "b" and r.escalation["team"]["tier"] == "jev"
