"""The contrastive method (jul/contrastive.py): CLM-8B's projection heads on a frozen encoder.

The fast tests check the texts against CLM's own `schema.build_pairs` rules, the numpy heads against a
torch reimplementation of CLM's `make_head`, and the client wiring with a fake backbone. The slow one
reproduces the reference answers of CLM's README on the real backbone.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from jul.contrastive import (ContrastiveReader, ContrastiveSpec, Head, load_heads, option_texts,
                             state_text, to_text)
from jul.types import Choice, Noul, NoulCriteria, Score, options_of

NOUL = {"true": "Yes. This is true: {instructions}", "false": "No. This is false: {instructions}"}
HIDDEN, WIDTH, PROJ = 16, 12, 8


def test_state_is_context_then_question_after_a_blank_line():
    assert state_text("charged twice", "Is it urgent?") == "charged twice\n\nIs it urgent?"
    assert state_text("charged twice", None) == "charged twice"
    assert state_text({"a": 1, "b": {"c": [1, {"d": 2}]}}, "Q") == "a: 1\n\nb:\n  c:\n    - 1\n    -\n      d: 2\n\nQ"


def test_objects_render_as_clm_writes_them_not_as_jul_does():
    # CLM separates top-level fields with a blank line; jul's `render` does not
    assert to_text({"x": "1", "y": "2"}) == "x: 1\n\ny: 2"
    assert to_text([True, 3]) == "- true\n- 3"


def test_options_are_written_verbatim_and_noul_in_clm_order():
    texts, index = option_texts("choice", "Team?", options_of(Choice("Team?", {"billing": "Charges", "tech": ""})), NOUL)
    assert texts == ["Charges", "tech"] and index == [0, 1]
    options = options_of(Noul("Urgent?"))
    texts, index = option_texts("noul", "Urgent?", options, NOUL)
    # jul's "Yes."/"No." defaults are not CLM's: its own default, built from the instructions, is used
    assert texts == ["false: No. This is false: Urgent?", "true: Yes. This is true: Urgent?"]
    assert [options[i].key for i in index] == ["false", "true"]
    texts, _ = option_texts("noul", "Late?", options_of(Noul("Late?", NoulCriteria(true="late", false="on time"))), NOUL)
    assert texts == ["false: on time", "true: late"]
    texts, _ = option_texts("score", "Angry?", options_of(Score("Angry?", ["Calm", "Angry"])), NOUL)
    assert texts == ["Calm", "Angry"]


def _arrays(seed=0, depth=3):
    r = np.random.RandomState(seed)
    out = {}
    for side in ("state_head", "action_head"):
        out[f"{side}.inp.weight"], out[f"{side}.inp.bias"] = r.randn(WIDTH, HIDDEN), r.randn(WIDTH)
        for i in range(depth - 2):
            out[f"{side}.hidden.{i}.weight"], out[f"{side}.hidden.{i}.bias"] = r.randn(WIDTH, WIDTH), r.randn(WIDTH)
            out[f"{side}.norms.{i}.weight"], out[f"{side}.norms.{i}.bias"] = r.rand(WIDTH) + .5, r.randn(WIDTH)
        out[f"{side}.out.weight"], out[f"{side}.out.bias"] = r.randn(PROJ, WIDTH), r.randn(PROJ)
    return {k: (v * .3).astype(np.float32) for k, v in out.items()}


CFG = {"width": WIDTH, "depth": 3, "activation": "gelu", "layernorm": True, "residual": False}


def test_numpy_head_matches_torch():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    a = _arrays()
    lin = lambda p: (lambda l: (l.weight.data.copy_(torch.from_numpy(a[p + ".weight"])),
                                l.bias.data.copy_(torch.from_numpy(a[p + ".bias"])), l)[-1])(
        nn.Linear(*a[p + ".weight"].shape[::-1]))
    ln = nn.LayerNorm(WIDTH)
    ln.weight.data.copy_(torch.from_numpy(a["state_head.norms.0.weight"]))
    ln.bias.data.copy_(torch.from_numpy(a["state_head.norms.0.bias"]))
    x = np.random.RandomState(1).randn(4, HIDDEN).astype(np.float32)
    with torch.no_grad():
        h = nn.functional.gelu(lin("state_head.inp")(torch.from_numpy(x)))
        h = nn.functional.gelu(ln(lin("state_head.hidden.0")(h)))
        want = nn.functional.normalize(lin("state_head.out")(h), dim=-1).numpy()
    assert np.abs(Head(a, "state_head", CFG)(x) - want).max() < 1e-5


def _heads_dir(tmp_path: Path) -> Path:
    np.savez(tmp_path / "heads.npz", cfg=np.array(json.dumps(CFG)), **_arrays())
    (tmp_path / "contrastive.json").write_text(json.dumps({
        "method": "contrastive", "backbone": {"mlx": "fake/mlx", "torch": "fake/torch"}, "pooling": "last",
        "max_tokens": 6, "scale": 20.0, "noul": NOUL, "heads": "heads.npz", "source": "test"}))
    return tmp_path


class FakeBackbone:
    """Hidden states from a fixed random table, so an embedding depends on the tokens only."""
    name = "fake"

    class tokenizer:
        eos_token_id = 0

        @staticmethod
        def encode(text, add_special_tokens=False):
            return [ord(c) % 97 for c in text]

    def __init__(self):
        self.table = np.random.RandomState(2).randn(97, HIDDEN).astype(np.float32)
        self.calls = []

    def last_hidden(self, tokens, prefix=None):
        self.calls.append(list(tokens))
        return np.cumsum(self.table[tokens], axis=0)


def test_reader_scores_scale_times_cosine_and_caches_options(tmp_path):
    spec = ContrastiveSpec.load(_heads_dir(tmp_path))
    bb = FakeBackbone()
    reader = ContrastiveReader(bb, spec)
    options = options_of(Choice("Team?", {"a": "alpha", "b": "beta"}))
    z, e, _ = reader.read("hello", "choice", "Team?", options)
    sh, ah = load_heads(spec)
    emb = lambda t: (lambda h: h / np.linalg.norm(h))(np.cumsum(bb.table[[ord(c) % 97 for c in t][-6:]], 0)[-1])
    want = 20.0 * ah(np.stack([emb("alpha"), emb("beta")])) @ sh(emb("hello\n\nTeam?")[None])[0]
    assert np.allclose(z, want, atol=1e-5) and np.allclose(e, emb("hello\n\nTeam?"), atol=1e-6)
    # the tail of a long text is kept, as vLLM's truncate_prompt_tokens does
    assert bb.calls[0] == [ord(c) % 97 for c in "hello\n\nTeam?"][-6:]
    n = len(bb.calls)
    reader.read("other", "choice", "Team?", options)
    assert len(bb.calls) == n + 1          # options come from the cache: only the state is embedded


def test_client_answers_and_autotunes_through_the_contrastive_reader(tmp_path, monkeypatch):
    from jul import Context, TypeSafeClient
    from jul.engine import Engine
    from jul.presets import contrastive_preset, save_preset

    (tmp_path / "heads").mkdir()
    heads = _heads_dir(tmp_path / "heads")
    monkeypatch.setattr("jul.presets.PRESET_HOME", tmp_path / "presets")
    save_preset(contrastive_preset("clm-test", heads, "mlx"), home=tmp_path / "presets")
    fake = FakeBackbone()
    monkeypatch.setattr("jul.client.Engine", lambda preset, backend=None: Engine(preset, backbone=fake))
    client = TypeSafeClient(model="clm-test", backend="mlx", context_home=tmp_path / "ctx")
    assert client._preset.method == "contrastive" and client._preset.torch_repo == "fake/torch"

    qs = {"team": Choice("Team?", {"a": "alpha", "b": "beta"}), "urgent": Noul("Urgent?"),
          "anger": Score("Angry?", ["calm", "angry", "furious"])}
    r = client.system_one(state="hello", questions=qs)
    assert r.choices["team"].choice in ("a", "b")
    assert abs(sum(r.choices["team"].probabilities.values()) - 1) < 1e-3
    assert 0 <= r.answers["urgent"].noul <= 1 and 0 <= r.answers["anger"].score <= 2

    rng = np.random.RandomState(3)
    words = ["alpha", "beta"]
    labeled = [(" ".join(rng.choice(["x", "y", "z"], 3)) + " " + words[i % 2], {"team": "ab"[i % 2]})
               for i in range(40)]
    ctx = Context(name="t")
    reports = client.autotune(ctx, {"team": qs["team"]}, labeled, save=False)
    assert reports["team"].n_examples == 40
    r2 = client.system_one(state=labeled[0][0], questions={"team": qs["team"]}, context=ctx)
    assert abs(sum(r2.choices["team"].probabilities.values()) - 1) < 1e-3
    with pytest.raises(ValueError, match="contrastive"):
        client.autotune(ctx, {"team": qs["team"]}, labeled, save=False, features="hybrid")


def test_preset_round_trip_keeps_the_heads(tmp_path):
    from jul.presets import contrastive_preset, load_preset, save_preset
    heads = _heads_dir(tmp_path)
    p = load_preset(save_preset(contrastive_preset("c", heads, "torch"), home=tmp_path / "p"))
    assert p.method == "contrastive" and p.heads == str(heads.resolve())
    assert p.repos == {"mlx": "fake/mlx", "torch": "fake/torch"}


def test_convert_writes_a_spec_that_loads(tmp_path):
    torch = pytest.importorskip("torch")
    from jul.contrastive import convert
    a = _arrays()
    ck = {"state_head": {k.split(".", 1)[1]: torch.from_numpy(v) for k, v in a.items() if k.startswith("state_head")},
          "action_head": {k.split(".", 1)[1]: torch.from_numpy(v) for k, v in a.items() if k.startswith("action_head")},
          "logit_scale": torch.tensor(np.log(20.0)), "cfg": {**CFG, "model": "Qwen/Qwen3-8B"},
          "hidden_size": HIDDEN, "projection_dim": PROJ}
    torch.save(ck, tmp_path / "h.pt")
    spec = ContrastiveSpec.load(convert(str(tmp_path / "h.pt"), tmp_path / "out"))
    assert abs(spec.scale - 20.0) < 1e-4 and spec.backbone["torch"] == "Qwen/Qwen3-8B"
    sh, _ = load_heads(spec)
    x = np.random.RandomState(1).randn(2, HIDDEN).astype(np.float32)
    assert np.allclose(sh(x), Head(a, "state_head", CFG)(x))


REFERENCE = [  # CLM README / independent runs on vLLM (bf16): state, question, expected answer
    ("Customer: my invoice was charged twice and nobody answers the phone!",
     Noul("Is this urgent?"), lambda a: 0.80 < a.noul < 0.90),
    ("Customer: my invoice was charged twice and nobody answers the phone!",
     Choice("Which team should handle this?", {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}),
     lambda a: a.choice == "billing" and a.probabilities["billing"] > 0.95),
    ("What causes tides on Earth?",
     Choice(None, {"a": "The Moon's gravitational pull.", "b": "Photosynthesis in plants.", "c": "Because the Earth is round."}),
     lambda a: a.choice == "a" and a.probabilities["a"] > 0.95),
]


@pytest.mark.slow
def test_clm_reference_answers():
    """Needs a preset added with `jul models add clm-8b --repo Contrastive-LM/CLM-v0.1-8B` (16 GB bf16 on
    torch, 8.7 GB 8-bit on MLX). Passes on MLX 8-bit; 4-bit gives urgency 0.713 and fails the first case."""
    from jul import TypeSafeClient
    client = TypeSafeClient(model=os.environ.get("JUL_CLM_PRESET", "clm-8b"))
    for state, q, ok in REFERENCE:
        a = client.system_one(state=state, questions={"q": q}).answers["q"]
        assert ok(a), (state, a)
