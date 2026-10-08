"""Unsloth decision models: `TypeSafeClient(model="unsloth:<checkpoint>")` hands the questions to Unsloth's
`FastDecisionModel.predict`.

The fast tests stand a fake `unsloth` module in for the real one, answering in the shape unsloth 2026.10.2's
`predict` returns (recorded from a real call on a checkpoint trained with the guide's recipe); the slow one
loads a real checkpoint (JUL_UNSLOTH_CHECKPOINT, a CUDA GPU).
"""

import os
import sys
import types

import pytest
from jul import Choice, Context, Escalation, Noul, NoulCriteria, Score, TypeSafeClient
from jul import unsloth_model
from jul.types import ChoiceAnswer, NoulAnswer, ScoreAnswer

Q = {"team": Choice("Which team?", {"billing": "invoices, payments, refunds", "technical": "bugs, outages, errors",
                                    "sales": "pricing, new plans"}),
     "urgent": Noul("Is it urgent?", NoulCriteria(true="needs action today", false="can wait")),
     "refund": Noul("Does the customer ask for a refund?"),
     "anger": Score("How angry?", ["calm", "annoyed", "furious"]),
     "topic": Choice("Topic?", ["sport", "finance"])}

#: What unsloth 2026.10.2's `FastDecisionModel.predict` returned for Q (extra fields included), on Qwen3.5-0.8B
#: trained with the guide's recipe cut to 60 steps (smoke run of #52, A10G).
ANSWERS = {
    "team": {"type": "choice", "choice": "billing", "confidence": 0.9883,
             "probabilities": {"billing": 0.9883, "technical": 0.0018, "sales": 0.0099}, "answer": "billing"},
    "urgent": {"type": "noul", "noul": 0.8327, "answer": True, "probabilities": {"true": 0.8327, "false": 0.1673}},
    "refund": {"type": "noul", "noul": 0.5635, "answer": True, "probabilities": {"true": 0.5635, "false": 0.4365}},
    "anger": {"type": "score", "score": 1.3963, "confidence": 0.5969, "legend": {"0": "calm", "1": "annoyed",
                                                                               "2": "furious"},
              "probabilities": {"0": 0.0034, "1": 0.5969, "2": 0.3997}, "answer": 1},
    "topic": {"type": "choice", "choice": "finance", "confidence": 0.9713,
              "probabilities": {"sport": 0.0287, "finance": 0.9713}, "answer": "finance"},
}


class FakeFast:
    """Stands in for `unsloth.FastDecisionModel`."""

    def __init__(self, answers):
        self.answers, self.loads, self.calls, self.inference = answers, [], [], []

    def from_pretrained(self, name, **kw):
        print("Unsloth banner")  # the real one prints on stdout
        self.loads.append((name, kw))
        return f"model:{name}", f"tokenizer:{name}"

    def for_inference(self, model):
        self.inference.append(model)

    def predict(self, model, tokenizer, state, questions):
        self.calls.append((model, tokenizer, state, questions))
        return {n: self.answers[n] for n in questions}


@pytest.fixture
def fake_unsloth(monkeypatch):
    fast = FakeFast(ANSWERS)
    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastDecisionModel=fast))
    monkeypatch.setattr(unsloth_model, "_has_head", lambda checkpoint: True)
    return fast


def test_answers_come_back_typed_with_unsloth_s_own_numbers(fake_unsloth):
    client = TypeSafeClient(model="unsloth:./qwen-decisions")
    r = client.system_one(state={"ticket": "charged twice"}, questions=Q)
    assert client.model == "unsloth:./qwen-decisions" and r.model == "unsloth:./qwen-decisions"
    assert r.usage.input_tokens == 0  # predict does not report token counts
    assert r.choices["team"] == ChoiceAnswer("billing", {"billing": 0.9883, "technical": 0.0018, "sales": 0.0099},
                                             0.9883)
    assert r.nouls["urgent"] == NoulAnswer(0.8327) and r.nouls["refund"] == NoulAnswer(0.5635)
    assert r.scores["anger"] == ScoreAnswer(1.3963, {"0": "calm", "1": "annoyed", "2": "furious"},
                                            {"0": 0.0034, "1": 0.5969, "2": 0.3997}, 0.5969)
    assert r.choices["topic"].choice == "finance"
    assert "answer" not in r.as_dict()["answers"]["team"]
    assert "probabilities" not in r.as_dict()["answers"]["urgent"]


def test_the_questions_reach_unsloth_in_the_system_one_body(fake_unsloth):
    TypeSafeClient(model="unsloth:./qwen-decisions").system_one(state={"ticket": "x"}, questions=Q)
    model, tokenizer, state, sent = fake_unsloth.calls[0]
    assert (model, tokenizer) == ("model:./qwen-decisions", "tokenizer:./qwen-decisions")
    assert state == {"ticket": "x"}  # the state as given: Unsloth renders it itself
    assert sent["team"] == {"type": "choice", "instructions": "Which team?",
                            "criteria": {"billing": "invoices, payments, refunds",
                                         "technical": "bugs, outages, errors", "sales": "pricing, new plans"}}
    assert sent["urgent"]["criteria"] == {"true": "needs action today", "false": "can wait"}
    assert "criteria" not in sent["refund"]  # the defaults stay Unsloth's own
    assert sent["anger"] == {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]}


def test_checkpoints_and_lazy_loading(fake_unsloth, capsys):
    client = TypeSafeClient(model="unsloth:someone/qwen-decisions")
    assert fake_unsloth.loads == []  # constructing loads nothing
    client.system_one(state="s", questions={"topic": Q["topic"]})
    client.system_one(state="s", questions={"topic": Q["topic"]})
    assert fake_unsloth.loads == [("someone/qwen-decisions", {})]
    assert fake_unsloth.inference == ["model:someone/qwen-decisions"]
    out = capsys.readouterr()
    assert "Unsloth banner" not in out.out and "Unsloth banner" in out.err  # stdout stays for jul's JSON
    for bad in ("unsloth:", "unsloth:qwen-decisions"):
        with pytest.raises(ValueError, match="Unknown Unsloth checkpoint"):
            TypeSafeClient(model=bad)


def test_a_model_without_a_decision_head_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastDecisionModel=FakeFast(ANSWERS)))
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="not an Unsloth decision model"):
        TypeSafeClient(model=f"unsloth:{plain}").system_one(state="s", questions={"topic": Q["topic"]})


def test_the_head_files_tell_a_decision_model(tmp_path, monkeypatch):
    for name, head in (("clef", "joint_head_config.json"), ("laya", "rl_agent_config.json")):
        (tmp_path / name).mkdir()
        (tmp_path / name / head).write_text("{}")
        assert unsloth_model._has_head(str(tmp_path / name)) is True
    (tmp_path / "plain").mkdir()
    assert unsloth_model._has_head(str(tmp_path / "plain")) is False

    class Api:
        def list_repo_files(self, repo):
            if repo == "offline/repo":
                raise OSError("no network")
            return {"a/clef": ["config.json", "joint_head_config.json"], "a/lm": ["config.json"]}[repo]
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Api))
    assert unsloth_model._has_head("a/clef") is True
    assert unsloth_model._has_head("a/lm") is False
    assert unsloth_model._has_head("offline/repo") is None  # unknown: Unsloth says what is missing


def test_what_does_not_apply_is_refused(fake_unsloth):
    with pytest.raises(ValueError, match="not on backend 'mlx'"):
        TypeSafeClient(model="unsloth:a/b", backend="mlx")
    client = TypeSafeClient(model="unsloth:a/b")
    with pytest.raises(ValueError, match="Unsloth's own runtime"):
        client.autotune(Context(name="t"), {"topic": Q["topic"]}, [("a", {"topic": "sport"}), ("b", {"topic": "finance"})])
    with pytest.raises(ValueError, match="create another one"):
        client.system_one(state="s", questions=Q, model="minicpm5-2b")
    with pytest.raises(ValueError, match="at least one question"):
        client.system_one(state="s", questions={})


def test_a_wrong_answer_type_is_an_error(fake_unsloth):
    fake_unsloth.answers = {**ANSWERS, "topic": ANSWERS["urgent"]}
    with pytest.raises(RuntimeError, match="wrong type"):
        TypeSafeClient(model="unsloth:a/b").system_one(state="s", questions={"topic": Q["topic"]})


def test_missing_package_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth", None)  # import unsloth -> ImportError
    with pytest.raises(ImportError, match=r"jul\[unsloth\]"):
        TypeSafeClient(model="unsloth:a/b").system_one(state="s", questions={"topic": Q["topic"]})


def test_unsloth_as_an_escalation_tier(fake_unsloth):
    class Unsure:
        def system_one(self, state, questions, **kw):
            from jul.types import SystemOneResponse, Usage
            return SystemOneResponse({n: ChoiceAnswer("technical", {"technical": 0.5}, 0.5) for n in questions},
                                     "local", Usage(), "r")
    r = Escalation([("local", Unsure()), ("unsloth", TypeSafeClient(model="unsloth:a/b"))]).system_one(
        "s", {"team": Q["team"]}, method="vector")
    assert r.choices["team"].choice == "billing" and r.escalation["team"]["tier"] == "unsloth"


def test_bench_reads_unsloth_with_its_default_only(fake_unsloth):
    from jul_cli.bench import readings
    methods, features, skipped = readings(TypeSafeClient(model="unsloth:a/b"), [None, "letters"], ["vector"])
    assert methods == [None] and features == []
    assert skipped == {"zero-shot:letters": "Unsloth: reads with its own runtime only",
                       "autotune:vector": "Unsloth: no autotune"}


def test_cli_preflight_asks_for_the_unsloth_package(monkeypatch):
    import importlib.util
    from jul_cli.setup import require_setup
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "unsloth" else real(n, *a))
    with pytest.raises(SystemExit, match="jul setup --model unsloth:a/b"):
        require_setup("unsloth:a/b", None)
    with pytest.raises(SystemExit, match="Unknown Unsloth checkpoint"):
        require_setup("unsloth:nope", None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: object() if n == "unsloth" else real(n, *a))
    require_setup("unsloth:a/b", None)  # nothing else to check: no backend, no preset


@pytest.mark.slow
def test_real_unsloth():
    checkpoint = os.environ.get("JUL_UNSLOTH_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set JUL_UNSLOTH_CHECKPOINT to an Unsloth decision checkpoint (needs a CUDA GPU)")
    pytest.importorskip("unsloth")
    r = TypeSafeClient(model=f"unsloth:{checkpoint}").system_one(state="I was charged twice, refund me now!",
                                                                 questions=Q)
    assert set(r.answers) == set(Q) and 0 <= r.nouls["urgent"].noul <= 1
    assert abs(sum(r.choices["team"].probabilities.values()) - 1) < 1e-3
