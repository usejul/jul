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


STRANDS = json.loads((FIXTURES / "decision_strands_cases.json").read_text())


@pytest.fixture(scope="module")
def strands_reader():
    """Strands Decider's format (fixtures/decision_strands.json) on the Qwen3.5-4B tokenizer.

    fixtures/decision_strands_cases.json was written by Strands Decider's own code (git 9800d14), see
    scripts/strands_decider_fixture.py: the ids its engine feeds the model, and where its pointer head reads.
    """
    transformers = pytest.importorskip("transformers")
    t = STRANDS["tokenizer"]
    try:
        tok = transformers.AutoTokenizer.from_pretrained(t["repo"], revision=t["revision"], local_files_only=True)
    except Exception:
        pytest.skip("the Qwen3.5-4B tokenizer is not in the local Hugging Face cache")

    class Backbone:
        name, tokenizer = "strands-decider", tok

    return PointerReader(Backbone(), DecisionSpec.load(FIXTURES, FIXTURES / "decision_strands.json"))


@pytest.mark.parametrize("case", STRANDS["cases"], ids=lambda c: "+".join(c["request"]["questions"]))
def test_requests_are_encoded_token_for_token_like_strands_decider(strands_reader, case):
    req = case["request"]
    reserve = max(len(b["ids"]) for b in case["branches"])
    assert strands_reader.encode_state(req["state"], reserve) == case["prefix"]
    for q, want in zip(req["questions"].values(), case["branches"]):
        texts, _ = strands_reader.option_texts(kind(q), options_of(question(q)))
        branch, q_idx, opt_idx = strands_reader.encode_question(q["instructions"], texts, kind(q))
        assert (branch, q_idx, opt_idx) == (want["ids"], want["question"], want["options"])


@pytest.mark.parametrize("case", STRANDS["cases"], ids=lambda c: "+".join(c["request"]["questions"]))
def test_a_call_leaves_the_state_what_its_longest_question_does_not_take(strands_reader, case):
    """`logits` encodes the questions first, so a long state is cut where Strands Decider cuts it."""
    seen = []

    class Backbone:
        name, tokenizer = "strands-decider", strands_reader.backbone.tokenizer

        def cache_prefix(self, ids):
            seen.append(ids)

        def last_hidden(self, ids, prefix=None):
            seen.append(ids)
            return np.zeros((len(ids), 2), dtype=np.float32)

    reader = PointerReader(Backbone(), strands_reader.spec)
    reader._head = (np.zeros((4, 2)), np.zeros(4), np.zeros((4, 2)), np.zeros(4), np.ones(2), np.zeros(2))
    req = case["request"]
    reader.logits(req["state"], [(kind(q), q["instructions"], options_of(question(q)))
                                 for q in req["questions"].values()])
    assert seen == [case["prefix"]] + [b["ids"] for b in case["branches"]]


def test_strands_noul_shows_the_models_defaults_and_maps_back(strands_reader):
    options = options_of(Noul("Late?", NoulCriteria(true="the parcel is late")))
    texts, index = strands_reader.option_texts("noul", options)
    assert texts == ["1. false \u2014 the statement does not hold for this state", "2. true \u2014 the parcel is late"]
    assert [options[i].key for i in index] == ["false", "true"]


def test_strands_head_layer_norm_and_temperature_per_type(tmp_path):
    """jul's numpy head against Strands Decider's PointerHead (torch numbers in the fixture) and a numpy
    rewrite of it: one LayerNorm on both sides, q and k, a dot product scaled by dim ** -0.5, then the
    temperature of each question's type."""
    h = STRANDS["head"]
    w = {k: np.asarray(v, dtype=np.float32) for k, v in h["weights"].items()}
    np.savez(tmp_path / "pointer_head.npz", **w)
    d = json.loads((FIXTURES / "decision_strands.json").read_text())
    d["head"]["dim"] = h["dim"]
    (tmp_path / "decision.json").write_text(json.dumps(d))
    spec = DecisionSpec.load(tmp_path)

    def layer_norm(x):
        mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + 1e-5) * w["norm_weight"] + w["norm_bias"]

    def read(kind_: str, hidden: np.ndarray, n: int) -> np.ndarray:
        """The head on `hidden`: its first `n` rows are the options, its last the question."""
        class Backbone:
            name, tokenizer = "synthetic", None

            def cache_prefix(self, ids):
                return None

            def last_hidden(self, ids, prefix=None):
                return hidden

        reader = PointerReader(Backbone(), spec)
        reader.encode_state = lambda state, reserve=0: []
        reader.encode_question = lambda instructions, texts, k=None: (list(range(len(hidden))), len(hidden) - 1,
                                                                      list(range(n)))
        q = (Noul("x?") if kind_ == "noul" else Choice("x?", [f"o{i}" for i in range(n)]) if kind_ == "choice"
             else Score("x?", [f"l{i}" for i in range(n)]))
        (z,), _ = reader.logits("state", [(kind_, "x?", options_of(q))])
        return z

    for kind_, decide, opts, want in zip(h["kinds"], h["decide"], h["options"], h["logits"]):
        decide, opts = np.asarray(decide, dtype=np.float32), np.asarray(opts, dtype=np.float32)
        n = 2 if kind_ == "noul" else len(opts)
        z = read(kind_, np.vstack([opts, decide]), n)                      # options first, the question last
        q_ = layer_norm(decide) @ w["q_weight"].T + w["q_bias"]
        k_ = layer_norm(opts[:n]) @ w["k_weight"].T + w["k_bias"]
        ref = (k_ @ q_) * h["dim"] ** -0.5 / spec.temperature_for(kind_)
        if kind_ == "noul":                                                # read (false, true), reported (true, false)
            ref, want = ref[::-1], want[:n][::-1]
        np.testing.assert_allclose(z, ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(z, want[:n], rtol=1e-4, atol=1e-5)
    assert spec.temperature_for("choice") == 0.6246728003026183 and spec.temperature_for("other") == spec.temperature


def test_a_kev_spec_keeps_one_temperature_and_no_norm():
    spec = DecisionSpec.load(FIXTURES, FIXTURES / "decision_minicpm5-2b.json")
    assert spec.text is None and spec.norm is None and spec.max_length is None and spec.state_render == "jul"
    assert {spec.temperature_for(k) for k in ("choice", "noul", "score")} == {spec.temperature}


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
