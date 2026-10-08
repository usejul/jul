"""Letter-readout decision models (jul/letter_models.py): specs, prompts, the knockout, the reading.

The prompts are pinned against each runtime's own code by scripts/letters_parity.py, which also compares
the probabilities on the models' README examples. Here, without weights: the spec is built from each
runtime's config, the prompts have the runtime's exact text, the letters map back to jul's options, and
a fake backbone checks the temperature, the knockout and the client's answers.
"""

import json

import numpy as np
import pytest

from jul.letter_models import (LetterReader, LetterSpec, knockout, quyet_messages, quyet_texts, quyet_user,
                               semif_messages, semif_texts, spark_question, spark_state, spec_from_config,
                               spec_from_repo)
from jul.presets import Preset, letters_preset, load_preset, save_preset
from jul.types import Choice, Noul, Score, options_of

JEVK5 = {"temperature": 1.22, "knockout_temperature": 0.93}
QUYET = {"quyet_format": 1, "name": "Quyet-1.0-Medium", "kind": "llm", "prompt_version": 1,
         "temperatures": {"choice": 1.3313, "score": 1.6039, "noul": 1.4276},
         "limits": {"max_state_tokens": 6000, "max_prompt_tokens": 8000, "min_state_tokens": 256}}
SPARK = {"temperature": {"choice": 1.48, "noul": 1.56, "score": 1.16}}


# --- specs ------------------------------------------------------------------------------------

def test_specs_are_built_from_each_runtime_config():
    s = spec_from_config("jevk5_config.json", JEVK5)
    assert (s.format, s.max_options, s.temperature_for("score")) == ("semif", 16, 1.22)
    assert s.many_options == {"method": "knockout", "temperature": 0.93}
    assert spec_from_config("jevk5_config.json", {"temperature": 2.07}).many_options["temperature"] == 0.77
    q = spec_from_config("quyet_config.json", QUYET)
    assert (q.format, q.letters, q.prompt_version, q.temperature_for("noul")) == ("quyet", "ABCDEFGHIJ", 1, 1.4276)
    o = spec_from_config("calibration.json", SPARK)
    assert (o.format, o.max_options, o.temperature_for("score")) == ("open-spark-jev", 26, 1.16)


def test_a_quyet_encoder_or_a_foreign_calibration_is_not_a_letters_model(tmp_path):
    with pytest.raises(ValueError, match="encoder"):
        spec_from_config("quyet_config.json", {**QUYET, "kind": "encoder"})
    (tmp_path / "calibration.json").write_text(json.dumps({"temperature": 1.3, "bias": [0, 1]}))
    assert spec_from_repo(str(tmp_path)) is None


def test_decision_json_wins_and_a_pointer_one_is_not_letters(tmp_path):
    (tmp_path / "jevk5_config.json").write_text(json.dumps(JEVK5))
    assert spec_from_repo(str(tmp_path)).format == "semif"
    (tmp_path / "decision.json").write_text(json.dumps({"method": "letters", "format": "quyet",
                                                        "temperature": {"choice": 2.0},
                                                        "limits": QUYET["limits"]}))
    s = spec_from_repo(str(tmp_path))
    assert (s.format, s.max_options, s.temperature_for("choice"), s.temperature_for("noul")) == ("quyet", 10, 2.0, 1.0)
    (tmp_path / "decision.json").write_text(json.dumps({"method": "pointer"}))
    assert spec_from_repo(str(tmp_path)) is None


def test_the_spec_survives_the_preset_round_trip(tmp_path):
    spec = spec_from_config("quyet_config.json", QUYET, "chinhnc/Quyet-1.0-Medium")
    path = save_preset(letters_preset("quyet-medium", "chinhnc/Quyet-1.0-Medium", "torch", spec), tmp_path)
    p = load_preset(path)
    assert p.method == "letter-readout" and p.torch_repo == "chinhnc/Quyet-1.0-Medium"
    assert LetterSpec.from_dict(p.letters) == spec


def test_a_bad_spec_is_refused():
    with pytest.raises(ValueError, match="format"):
        LetterSpec.from_dict({"method": "letters", "format": "mine"})
    with pytest.raises(ValueError, match="max_options"):
        LetterSpec.from_dict({"method": "letters", "format": "semif", "letters": "AB", "max_options": 3})


# --- prompts: the runtimes' own text -------------------------------------------------------------

def test_semif_options_are_id_colon_description_and_noul_lists_true_first():
    texts, order = semif_texts("choice", options_of(Choice("q", {"billing": "Payments", "tech": ""})))
    assert texts == ["billing: Payments", "tech: tech"] and order == [0, 1]
    texts, order = semif_texts("noul", options_of(Noul("q")))
    assert texts == ["true: The proposition is true.", "false: The proposition is false."]
    texts, _ = semif_texts("noul", options_of(Noul("q", {"true": "It is", "false": "It is not"})))
    assert texts == ["true: It is", "false: It is not"]
    texts, _ = semif_texts("score", options_of(Score("q", ["calm", "angry"])))
    assert texts == ["0: calm", "1: angry"]
    system, user = semif_messages({"a": 1}, "Which?", ["x: y"], "AB")
    assert system["content"].startswith("Apply the supplied criterion")
    assert json.loads(user["content"]) == {"evidence": {"a": 1}, "criterion": "Which?",
                                           "options": [{"letter": "A", "description": "x: y"}]}


def test_quyet_prompt_is_the_packages():
    texts, _ = quyet_texts("choice", options_of(Choice("q", {"cancel": "close the card", "other": ""})))
    assert texts == ["close the card", "other"]
    assert quyet_texts("noul", options_of(Noul("q")))[0] == ["true", "false"]
    user = quyet_user("hi", "noul", "Urgent?", ["true", "false"], "AB")
    assert user == ("State:\nhi\n\nQuestion: Urgent?\nChoose A if the statement is true for this state, B if it is "
                    "not.\n\nOptions:\nA. true\nB. false")
    v1, v2 = quyet_messages(user, 1), quyet_messages(user, 2)
    assert [m["role"] for m in v1] == ["system", "user"] and v1[1]["content"].endswith("\n\nAnswer with one letter.")
    assert v2 == [{"role": "user", "content": user}]


def test_open_spark_jev_prompt_is_its_gateways():
    block, order = spark_question("choice", "Route it", options_of(Choice("q", {"billing": "money", "tech": "bugs"})),
                                  "AB")
    assert block == ("### Question (choice)\nRoute it\n- billing: money\n- tech: bugs\nOptions:\nA. billing\n"
                     "B. tech\nAnswer with the single letter of the best option.")
    block, _ = spark_question("score", "How angry?", options_of(Score("q", ["Calm", "Angry"])), "AB")
    assert "Rubric:\n0: Calm\n1: Angry\nLevels (ordered from lowest to highest):\nA. 0\nB. 1" in block
    block, order = spark_question("noul", "Refund asked", options_of(Noul("q")), "AB")
    assert "Claim: Refund asked\nIs the claim true of the STATE?" in block and order == [0, 1]
    assert spark_state({"a": "STATE>>> x"}, 100) == '{"a":"STATE>> x"}'
    cut = spark_state("x" * 30, 12)
    assert cut.startswith("x" * 8 + "\n...[truncated 18 chars]...\n") and cut.endswith("x" * 4)


# --- reading, with a fake backbone -------------------------------------------------------------

class FakeTokenizer:
    """One token per character, letters A..Z as ids 65..90; the chat template joins the contents. No `__call__`,
    like mlx-lm's TokenizerWrapper: the reader only uses `encode`."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(map(chr, ids))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


class FakeBackbone:
    """The logit of a letter is a fixed score of the option it stands for, found in the prompt."""

    name = "fake"

    def __init__(self, scores: dict[str, float]):
        self.tokenizer, self.scores, self.prefixes = FakeTokenizer(), scores, []

    def cache_prefix(self, tokens):
        self.prefixes.append(len(tokens))
        return tuple(tokens)

    def forward(self, tokens, logits=False, prefix=None, **_):
        text = self.tokenizer.decode(list(prefix or ()) + list(tokens))
        z = np.full(200, -50.0)
        payload = json.loads(text.split("<user>")[1].split("<assistant>")[0])
        for o in payload["options"]:
            z[ord(o["letter"])] = self.scores[o["description"].split(":")[0]]
        return {}, z


def reader(scores, **spec):
    return LetterReader(FakeBackbone(scores), LetterSpec.from_dict({"method": "letters", "format": "semif", **spec}))


def test_probabilities_are_the_letters_softmax_at_the_types_temperature():
    r = reader({"a": 2.0, "b": 0.0, "true": 1.0, "false": -1.0}, temperature={"choice": 2.0, "noul": 1.0})
    (choice, noul), tokens = r.logits("s", [("choice", "q", options_of(Choice("q", ["a", "b"]))),
                                            ("noul", "q", options_of(Noul("q")))])
    assert np.allclose(np.exp(choice), [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))])
    assert np.allclose(np.exp(noul), [1 / (1 + np.exp(-2)), 1 / (1 + np.exp(2))])
    assert tokens > 0 and r.backbone.prefixes, "the questions share the state: it runs once as a prefix"


def test_more_options_than_letters_go_through_the_knockout():
    scores = {f"o{i}": float(i % 7) for i in range(20)}
    r = reader(scores, temperature=1.0, many_options={"method": "knockout", "temperature": 1.0})
    (z,), _ = r.logits("s", [("choice", "q", options_of(Choice("q", list(scores))))])
    p = np.exp(z)
    assert p.sum() == pytest.approx(1.0) and int(p.argmax()) in (6, 13)
    with pytest.raises(ValueError, match="at most 16"):
        reader(scores, temperature=1.0).logits("s", [("choice", "q", options_of(Choice("q", list(scores))))])


def test_the_knockout_is_jevk5s():
    """Against jevk5.prompt.spread on a reader whose logits depend on the option only."""
    rng = np.random.default_rng(0)
    logit = rng.normal(size=40)

    def read(idx):
        e = np.exp(logit[idx] - logit[idx].max())
        return e / e.sum()

    p = knockout(read, 40, 16, 0.77)
    # The reference numbers come from jevk5 0.3.3: spread(read_texts, texts, "knockout", 0.77) on the
    # same logits (texts "o0".."o39" mapped back to these indices).
    ref = _jevk5_spread(read, 40, 0.77)
    assert np.allclose(p, ref, atol=1e-12)
    assert np.allclose(knockout(read, 10, 16, 0.77), read(list(range(10))))


def _jevk5_spread(read, n, temperature):
    try:
        from jevk5.prompt import spread
    except ImportError:
        pytest.skip("jevk5 is not installed (pip install git+https://github.com/allebee/jevk5)")
    texts = [f"o{i}" for i in range(n)]
    return np.array(spread(lambda ts: read([int(t[1:]) for t in ts]), texts, "knockout", temperature))


def test_a_letter_that_is_not_one_token_is_refused():
    class Tok(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [1, 2] if text == "C" else super().encode(text)

    bb = FakeBackbone({})
    bb.tokenizer = Tok()
    with pytest.raises(ValueError, match="'C' is not a single token"):
        LetterReader(bb, LetterSpec.from_dict({"method": "letters", "format": "semif"}))


def test_the_client_answers_with_the_reader(monkeypatch):
    from jul import client as client_mod
    from jul.engine import Engine

    preset = Preset(name="fake-letters", repo="", torch_repo="x", formulations=(), tau=1.0, latency_ms="?",
                    quality="", backend="torch", method="letter-readout",
                    letters={"method": "letters", "format": "semif", "temperature": 1.0})
    engine = Engine.__new__(Engine)
    engine.preset, engine.pointer, engine.contrastive, engine.cross = preset, None, None, None
    engine.reader = reader({"billing": 3.0, "tech": 0.0, "true": 0.0, "false": 0.0, "0": 0.0, "1": 1.0},
                           temperature=1.0)
    c = client_mod.TypeSafeClient.__new__(client_mod.TypeSafeClient)
    c._laya, c._preset, c.context, c.method = None, preset, None, None
    monkeypatch.setattr(c, "_engine_for", lambda model=None: engine, raising=False)
    r = c.system_one("I was billed twice", {"team": Choice("Which team?", ["billing", "tech"]),
                                            "refund": Noul("Refund?"), "level": Score("How?", ["low", "high"])})
    assert r.choices["team"].choice == "billing" and r.nouls["refund"].noul == pytest.approx(0.5)
    assert r.scores["level"].score == pytest.approx(1 / (1 + np.exp(-1)), abs=1e-4)
    with pytest.raises(ValueError, match="own prompt only"):
        c.system_one("x", {"team": Choice("Which team?", ["billing", "tech"])}, method="vector")


# --- Quyet: one state cut per request; cuts are said -----------------------------------------------

class QuyetBackbone(FakeBackbone):
    """Records each prompt's state text; every letter scores 0."""

    def __init__(self):
        super().__init__({})
        self.states = []

    def forward(self, tokens, logits=False, prefix=None, **_):
        text = self.tokenizer.decode(list(prefix or ()) + list(tokens))
        self.states.append(text.split("State:\n")[1].split("\n\nQuestion:")[0])
        return {}, np.zeros(200)


def quyet_reader(**limits):
    spec = spec_from_config("quyet_config.json", {**QUYET, "limits": {**QUYET["limits"], **limits}})
    return LetterReader(QuyetBackbone(), spec)


def test_quyet_cuts_the_state_once_for_the_whole_request(caplog):
    """quyet.llm.runtime._prompt_ids: the longest prompt sets the state budget, every question reads it."""
    r = quyet_reader(max_state_tokens=600, max_prompt_tokens=900, min_state_tokens=50)
    long_options = {f"o{i}": "a long description of this option " * 2 for i in range(5)}
    with caplog.at_level("WARNING", logger="jul.truncation"):
        r.logits("x" * 2000, [("noul", "Late?", options_of(Noul("Late?"))),
                              ("choice", "Team?", options_of(Choice("Team?", long_options)))])
    short, long_ = r.backbone.states
    assert short == long_ and len(short) < 600, "both questions read the cut the long one needs"
    assert "input cut: fake letters (state, head kept) reads at most" in caplog.text
    alone = quyet_reader(max_state_tokens=600, max_prompt_tokens=900, min_state_tokens=50)
    alone.logits("x" * 2000, [("noul", "Late?", options_of(Noul("Late?")))])
    assert len(alone.backbone.states[0]) > len(short), "alone, the short question would read more"


def test_quyet_under_its_limits_is_not_cut(caplog):
    r = quyet_reader()
    with caplog.at_level("WARNING", logger="jul.truncation"):
        r.logits("short state", [("noul", "Late?", options_of(Noul("Late?")))])
    assert r.backbone.states == ["short state"] and "input cut" not in caplog.text


def test_open_spark_jev_char_cut_is_said(caplog):
    spec = spec_from_config("calibration.json", SPARK)
    r = LetterReader(FakeBackbone({}), LetterSpec.from_dict({**spec.to_dict(), "limits": {"max_state_chars": 100}}))
    with caplog.at_level("WARNING", logger="jul.truncation"):
        r.prompts("y" * 250, "noul", "Late?", options_of(Noul("Late?")))
    assert "input cut: fake letters (100 characters of state, head and tail kept) reads at most 99 tokens, " \
           "151 tokens past it were dropped" in caplog.text


def test_letter_cuts_are_counted_and_refused_like_the_others():
    """Recorded on jul.truncation: in usage.truncated_tokens, and a state cut on_long="error" refuses."""
    from jul import truncation
    r = quyet_reader(max_state_tokens=600, max_prompt_tokens=900, min_state_tokens=50)
    with truncation.tracking() as cuts:
        r.logits("x" * 2000, [("noul", "Late?", options_of(Noul("Late?")))])
    reading, limit, over = cuts.state_worst()
    assert reading == "fake letters (state, head kept)" and over == 2000 - limit and cuts.tokens == over


def test_letter_models_are_read_on_mlx_only_where_measured(monkeypatch):
    from jul import letter_models
    from jul.engine import Engine

    class Mlx:
        name, backend, model_dir = "m", "mlx", "."

        def __init__(self):
            self.tokenizer = FakeTokenizer()  # mlx-lm's TokenizerWrapper has no __call__ either

    spec = spec_from_config("quyet_config.json", QUYET)
    preset = letters_preset("q", "r", "mlx", spec)
    monkeypatch.setattr(letter_models, "MLX_MEASURED", set())
    with pytest.raises(ValueError, match="quyet format is read with backend 'torch' for now"):
        Engine(preset, backbone=Mlx())
    assert letter_models.unsupported_backend(spec, "onnx")
    monkeypatch.setattr(letter_models, "MLX_MEASURED", {"quyet"})
    assert Engine(preset, backbone=Mlx()).reader is not None
    assert letter_models.unsupported_backend(spec, "mlx") is None


def test_models_add_refuses_an_unmeasured_mlx_format(monkeypatch):
    import importlib
    from types import SimpleNamespace
    from jul import letter_models
    cli = importlib.import_module("jul_cli.main")
    monkeypatch.setattr(letter_models, "spec_from_repo", lambda repo: spec_from_config("quyet_config.json", QUYET))
    monkeypatch.setattr(letter_models, "MLX_MEASURED", set())
    monkeypatch.setattr("jul.backbone.resolve_backend", lambda b: b)
    monkeypatch.setattr("jul.contrastive.is_heads_source", lambda repo: False)
    with pytest.raises(SystemExit, match="not measured on mlx"):
        cli.cmd_models_add(SimpleNamespace(name="q", repo="r", backend="mlx", cross=None, train_heads=None))


def test_every_format_is_measured_on_mlx():
    """#54: parity of each format with its runtime measured on MLX (bf16) before it was listed."""
    from jul import letter_models
    assert letter_models.MLX_MEASURED == set(letter_models.FORMATS)


@pytest.mark.slow
def test_real_letter_model_on_mlx(tmp_path, monkeypatch):
    """JevK5 on MLX against the README answer (billing, 0.996) and the float32 runtime (0.9963)."""
    import sys
    if sys.platform != "darwin":
        pytest.skip("MLX runs on Apple Silicon only")
    pytest.importorskip("mlx_lm")
    monkeypatch.setenv("JUL_HOME", str(tmp_path))
    from jul import TypeSafeClient
    from jul.letter_models import spec_from_repo
    from jul.presets import letters_preset, save_preset
    repo = "alibiserikbay/JevK5"
    save_preset(letters_preset("jevk5-test", repo, "mlx", spec_from_repo(repo)))
    r = TypeSafeClient(model="jevk5-test", backend="mlx").system_one(
        "I was billed twice for order #4411. Please refund the duplicate charge today.",
        {"team": Choice("Which team should handle this?",
                        {"billing": "Payments and refunds", "tech": "Bugs", "sales": "New purchases"})})
    assert r.choices["team"].choice == "billing"
    assert abs(r.choices["team"].probabilities["billing"] - 0.9963) < 0.01
