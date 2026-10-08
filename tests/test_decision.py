"""The pointer method (jul/decision.py) against Kev's own code, which trained minicpm5-2b-decision.

fixtures/decision_kev.json was written by Kev (torch, fp32, merged-free LoRA): for three requests, the
prefix and branch token ids its encoder produces and the probabilities its model gives (calibrated,
T = 1.954). The fast tests check that jul encodes every request token for token like Kev; the slow one
that the MLX 4-bit model gives the same answers.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from jul.decision import DecisionSpec, PointerReader, render
from jul.presets import Formulation, Preset
from jul.types import Choice, Noul, NoulCriteria, Score, options_of

FIXTURES = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "decision_kev.json").read_text())
MODELS = Path(os.environ.get("JUL_DECISION_MODELS", Path(__file__).resolve().parents[2] / "models"))


def question(q: dict):
    if q["type"] == "choice":
        c = q["criteria"]
        return Choice(q["instructions"], {k: v or "" for k, v in c.items()} if any(c.values()) else list(c))
    if q["type"] == "noul":
        c = q.get("criteria") or {}
        return Noul(q["instructions"], NoulCriteria(true=c.get("true", ""), false=c.get("false", "")) if c else None)
    return Score(q["instructions"], q["criteria"])


def kind(q: dict) -> str:
    return q["type"]


@pytest.fixture(scope="module")
def reader():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("openbmb/MiniCPM5-2B", local_files_only=True)
    except Exception:
        pytest.skip("the MiniCPM5-2B tokenizer is not in the local Hugging Face cache")

    class Backbone:
        name, tokenizer = "minicpm5-2b-decision", tok

    return PointerReader(Backbone(), DecisionSpec.load(FIXTURES, FIXTURES / "decision_minicpm5-2b.json"))


def test_objects_are_rendered_as_labeled_fields():
    assert render({"subject": "Hi", "items": ["a", "b"]}) == "subject: Hi\nitems:\n  - a\n  - b"


@pytest.mark.parametrize("case", CASES, ids=lambda c: "+".join(c["request"]["questions"]))
def test_requests_are_encoded_token_for_token_like_kev(reader, case):
    req = case["request"]
    assert reader.encode_state(req["state"]) == case["prefix"]
    for q, want in zip(req["questions"].values(), case["branches"]):
        texts, _ = reader.option_texts(kind(q), options_of(question(q)))
        branch, q_idx, opt_idx = reader.encode_question(q["instructions"], texts)
        assert (branch, q_idx, opt_idx) == (want["ids"], want["decide"], want["opts"])


def test_noul_options_follow_the_trained_order_and_map_back(reader):
    options = options_of(Noul("Late?", NoulCriteria(true="the parcel is late", false="anything else")))
    texts, index = reader.option_texts("noul", options)
    assert texts == ["no: anything else", "yes: the parcel is late"]
    assert [options[i].key for i in index] == ["false", "true"]
    # jul's default descriptions are not shown to the model, which was trained without them
    texts, _ = reader.option_texts("noul", options_of(Noul("Late?")))
    assert texts == ["no", "yes"]


@pytest.mark.slow
@pytest.mark.parametrize("weights, tolerance", [("mlx-bf16", 0.03), ("mlx-4bit", 0.3)])
def test_mlx_answers_match_kev(tmp_path, monkeypatch, weights, tolerance):
    """bf16 checks the port itself; 4-bit moves probabilities more (up to ~0.3 measured) but not the answer."""
    model_dir = MODELS / f"minicpm5-2b-decision-{weights}"
    if not (model_dir / "decision.json").exists():
        pytest.skip(f"{model_dir} is not on disk")
    from jul import TypeSafeClient
    from jul.presets import pointer_preset, save_preset
    monkeypatch.setattr("jul.presets.PRESET_HOME", tmp_path)
    save_preset(pointer_preset("decision-test", str(model_dir), "mlx"), home=tmp_path)
    client = TypeSafeClient(model="decision-test", backend="mlx")
    for case in CASES:
        qs = {name: question(q) for name, q in case["request"]["questions"].items()}
        answers = client.system_one(state=case["request"]["state"], questions=qs).answers
        for name, want in case["probabilities"].items():
            a = answers[name]
            got = a.probabilities if hasattr(a, "probabilities") else {"true": a.noul, "false": 1 - a.noul}
            keys = list(want)
            assert np.argmax([got[k] for k in keys]) == np.argmax([want[k] for k in keys]), name
            assert max(abs(got[k] - want[k]) for k in keys) < tolerance, (name, got, want)
    client.close()


def test_mlx_gets_the_rope_base_that_transformers_5_moved(tmp_path):
    """Without it mlx-lm silently uses 10000 instead of the model's base, and answers are wrong."""
    pytest.importorskip("mlx")
    from jul.backends.mlx import _rope_fix
    (tmp_path / "config.json").write_text(json.dumps({"rope_parameters": {"rope_theta": 5000000}}))
    assert _rope_fix(str(tmp_path)) == {"rope_theta": 5000000}
    (tmp_path / "config.json").write_text(json.dumps({"rope_theta": 10000, "rope_parameters": {"rope_theta": 10000}}))
    assert _rope_fix(str(tmp_path)) is None


def test_mlx_reads_a_text_only_qwen3_5_checkpoint(tmp_path):
    """Merged Qwen3_5ForCausalLM fine-tunes (JevK5, Plumb, Quyet, spark-s1) say `qwen3_5_text`, which mlx-lm
    does not list; its `qwen3_5` model reads their flat config and weight names."""
    pytest.importorskip("mlx")
    from jul.backends.mlx import _rope_fix
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text",
                                                      "rope_parameters": {"rope_theta": 10000000}}))
    assert _rope_fix(str(tmp_path)) == {"rope_theta": 10000000, "model_type": "qwen3_5"}


def test_a_decision_model_is_recognised_by_its_spec_file(tmp_path):
    """Without this, `jul models add` would run the vector calibration on a decision model."""
    from jul.decision import spec_source
    assert spec_source(str(tmp_path)) is None
    (tmp_path / "decision.json").write_text("{}")
    assert spec_source(str(tmp_path)) == str(tmp_path.resolve())


def test_a_model_that_says_nothing_routes_nowhere():
    """There is no guessed default: `jul models add` measures the threshold, or there is none."""
    spec = DecisionSpec.load(FIXTURES, FIXTURES / "decision_minicpm5-2b.json")
    assert spec.routing is None and spec.route_above is None


def test_a_fitted_fallback_survives_the_preset_round_trip(tmp_path):
    """The fallback's numbers live in the preset, so `jul models add` can fit them for a Hub repo."""
    from jul.decision import fallback_preset
    from jul.presets import load_preset, routing_from, save_preset
    fitted = Preset(name="m", repo="r", formulations=(Formulation("one_word", 'x "{state}" y', 29),),
                    tau=0.048, center="generic", latency_ms="?", quality="", backend="mlx",
                    calibration={"date": "2026-09-23", "dev_accuracy": 0.6})
    preset = Preset(name="m", repo="r", formulations=(), tau=1.0, latency_ms="?", quality="",
                    backend="mlx", method="pointer", routing=routing_from(fitted, 32))
    reloaded = load_preset(save_preset(preset, tmp_path))
    assert reloaded.method == "pointer" and reloaded.formulations == ()
    assert reloaded.routing["above_options"] == 32 and reloaded.routing["center"] == "generic"

    back = fallback_preset("m", "mlx", reloaded.routing, tmp_path)
    assert back.method == "vector" and back.layers == [29] and back.tau == 0.048
    # the name and directory are the ones the centers were fitted under, or "generic" finds no asset
    assert back.name == "m" and back.asset_dir == tmp_path


def test_a_model_can_route_question_types_to_its_vectors(tmp_path):
    """A pointer head that reads Score worse than its own vectors sends Score there, whatever the option count."""
    d = json.loads((FIXTURES / "decision_minicpm5-2b.json").read_text())
    (tmp_path / "decision.json").write_text(json.dumps({**d, "routing": {"types": ["score"]}}))
    spec = DecisionSpec.load(tmp_path)
    assert spec.route_types == ("score",) and spec.route_above is None


def test_routing_by_count_and_by_type_are_independent():
    from jul.client import routed_questions
    qs = {"team": Choice(instructions="Which team?", criteria=[f"t{i}" for i in range(12)]),
          "urgent": Noul(instructions="Urgent?"),
          "level": Score(instructions="How bad?", criteria=["low", "medium", "high"])}
    assert set(routed_questions(qs, 10, ())) == {"team"}
    assert set(routed_questions(qs, 10, ("score",))) == {"team", "level"}
    assert set(routed_questions(qs, 0, ("score",))) == {"level"}   # route_above=0 keeps the type routing
    assert routed_questions(qs, None, None) == {}


def test_a_routed_question_is_read_at_the_fallback_tau():
    """Reading the fallback at the pointer preset's tau (1.0) made every routed answer near uniform."""
    from types import SimpleNamespace

    from jul.client import TypeSafeClient
    client = TypeSafeClient.__new__(TypeSafeClient)
    client._preset = SimpleNamespace(tau=1.0)                       # the pointer preset
    cos = np.array([0.30, 0.20, 0.10])
    engine = SimpleNamespace(preset=SimpleNamespace(tau=1.0),        # the engine keeps the pointer preset
                             compile=lambda *a: None, read=lambda compiled, text, shared: (cos, None, 7))
    fallback = SimpleNamespace(tau=0.05)                              # the vector fallback, passed along
    client._head = lambda *a: None
    client._head_formulations = lambda head, preset=None: None
    q = Score(instructions="How bad?", criteria=["low", "medium", "high"])
    p, tokens = client._answer_probabilities(engine, "score", "vector", q, options_of(q), "text", None, {},
                                             preset=fallback)
    assert tokens == 7 and p[0] > 0.85                               # softmax(cos / 0.05), not softmax(cos / 1.0)
