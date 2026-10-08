"""The pointer method: a model trained to decide, read at its own delimiter tokens.

The format read here — delimiter tokens, one branch per question, a pointer head scoring each option's
last token against the question's — is the one of Kev (https://github.com/jaredpalmer/kev, Jared Palmer,
Apache 2.0), which trained the models this reads. This is an independent implementation; the tests check
it against Kev's own encoder and model (see NOTICE).

A decision model (for example `jul-decision-minicpm5-2b`: MiniCPM5-2B with a merged LoRA and a pointer
head) was trained on one input format. That format lives next to the weights, in `decision.json`, so
the code here knows nothing about any particular model:

    tokens    the delimiter tokens (state, question, option_open, option_close, decide)
    layout    how a request is laid out: a prefix holding the state, then one branch per question
    readout   which hidden states are compared: the question token against each option token
    head      the pointer head's weights file, its dimension and its temperature
    limits    the longest state and branch the model was trained on

The state is encoded once and kept as a cached prefix; every question of a call continues from it, so
questions never see each other. Scores are `k(h_option) . q(h_question) / sqrt(dim)`, divided by the
temperature, then a softmax.

A model trained on plain text instead of delimiter tokens (the format of Strands Decider,
https://github.com/strands-labs/strands-decider, Apache 2.0: tags such as `<state>` and numbered option
lines) has a `text` section in place of `tokens` and `layout`: templates for the prefix, the branch and the
option lines. Its readout is by characters: the question is the branch's last token, and each option the last
token lying inside its line (from the tokenizer's offsets). Its head may also carry a LayerNorm applied to
both sides before `q` and `k` (`head.norm`), and a temperature per question type (`head.temperature_by_kind`).
The state may be rendered as indented JSON (`state_render`), and `limits.max_length` gives the window state
and branch share: the state then gets what the call's longest branch leaves.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import truncation
from .types import NOUL_DEFAULTS, Option

SPEC_FILE = "decision.json"


def render_json(v: Any) -> str:
    """str -> stripped; object | array -> indented JSON, key order and unicode kept (Strands Decider)."""
    if v is None:
        return ""
    return v.strip() if isinstance(v, str) else json.dumps(v, indent=2, ensure_ascii=False)


def render(v: Any, indent: int = 0) -> str:
    """str | object | array -> the text the model was trained on (field names kept as labels)."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
                     for k, x in v.items())


@dataclass(frozen=True)
class DecisionSpec:
    """The content of a model's decision.json."""

    tokens: dict[str, str]
    layout: dict[str, list[str]]
    option_text: dict[str, str]
    noul_options: tuple[str, str]           # names for (false, true)
    escape: tuple[str, str] | None          # (pattern, replacement) applied to user text
    add_special_tokens: bool
    question_token: str
    option_token: str
    head_file: str
    query: str
    key: str
    dim: int
    temperature: float
    max_state_tokens: int
    max_branch_tokens: int
    directory: Path
    #: Optional: above `above_options` options the pointer head stops earning its latency, and the question
    #: `types` it reads worse than the vectors (e.g. ["score"]); both fall back to the vector reading.
    #: None = never route. See `route_above` and `route_types`.
    routing: dict[str, Any] | None
    #: Optional, a model trained on plain text: templates of prefix, branch and option lines (see `PointerReader`).
    text: dict[str, Any] | None = None
    #: Optional: a LayerNorm (npz prefix) applied to question and option hidden states before `q` and `k`.
    norm: str | None = None
    norm_eps: float = 1e-5
    #: Optional: temperature per question type; `temperature` for a type it does not name.
    temperature_by_kind: dict[str, float] | None = None
    #: How a state object becomes text: "jul" (`render`) or "json-indent" (`render_json`).
    state_render: str = "jul"
    #: Optional: the window state and branch share. The state then gets at most what the call's longest
    #: branch leaves, as the model's own runtime does.
    max_length: int | None = None

    @classmethod
    def load(cls, directory: str | Path, file: str | Path | None = None) -> "DecisionSpec":
        """`directory` holds the weights and the head; `file` overrides where decision.json is read."""
        directory = Path(directory)
        d = json.loads(Path(file or directory / SPEC_FILE).read_text())
        if d.get("method") != "pointer":
            raise ValueError(f"{directory / SPEC_FILE}: unsupported method {d.get('method')!r}")
        esc = d.get("escape_user_specials")
        head, readout, limits = d["head"], d["readout"], d["limits"]
        if d.get("text") and (readout["question_token"], readout["option_token"]) != ("last", "option_line"):
            raise ValueError(f"{directory / SPEC_FILE}: a text layout is read at 'last' and 'option_line'")
        if d.get("state_render", "jul") not in RENDERERS:
            raise ValueError(f"{directory / SPEC_FILE}: unknown state_render {d['state_render']!r}")
        by_kind = head.get("temperature_by_kind")
        return cls(tokens=d.get("tokens", {}), layout=d.get("layout", {}), option_text=d.get("option_text", {}),
                   noul_options=tuple(d["noul_options"]),
                   escape=(esc["pattern"], esc["replace"]) if esc else None,
                   add_special_tokens=bool(d.get("add_special_tokens", False)),
                   question_token=readout["question_token"], option_token=readout["option_token"],
                   head_file=head["file"], query=head["query"], key=head["key"], dim=int(head["dim"]),
                   temperature=float(head["temperature"]),
                   max_state_tokens=int(limits["max_state_tokens"]),
                   max_branch_tokens=int(limits["max_branch_tokens"]), directory=directory,
                   routing=d.get("routing"), text=d.get("text"), norm=head.get("norm"),
                   norm_eps=float(head.get("norm_eps", 1e-5)),
                   temperature_by_kind={k: float(t) for k, t in by_kind.items()} if by_kind else None,
                   state_render=d.get("state_render", "jul"),
                   max_length=int(limits["max_length"]) if limits.get("max_length") else None)

    def temperature_for(self, kind: str) -> float:
        return float((self.temperature_by_kind or {}).get(kind, self.temperature))


    @property
    def route_above(self) -> int | None:
        """Option count above which the vector reading answers instead; None when the model routes nowhere."""
        above = (self.routing or {}).get("above_options")
        return int(above) if above is not None else None

    @property
    def route_types(self) -> tuple[str, ...]:
        """Question types the vector reading answers whatever their option count (a model whose pointer head
        reads e.g. Score worse than its own vectors says so here)."""
        return tuple((self.routing or {}).get("types") or ())


RENDERERS = {"jul": render, "json-indent": render_json}


def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps) * gamma + beta


def encode_with_offsets(tok, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """(ids, character offsets) of `text` without special tokens, from the tokenizer a backend holds."""
    raw = getattr(tok, "_tok", None)            # jul's ONNX tokenizer, over a bare `tokenizers.Tokenizer`
    if raw is not None:
        e = raw.encode(text, add_special_tokens=False)
        return list(e.ids), [tuple(o) for o in e.offsets]
    if type(tok).__name__ == "TokenizerWrapper":  # mlx-lm wraps the Hugging Face tokenizer
        tok = tok._tokenizer
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]


def option_token_index(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[int]:
    """For each (start, end) character span, the last non-empty token lying entirely inside it."""
    out = []
    for a, b in spans:
        inside = [j for j, (lo, hi) in enumerate(offsets) if hi > lo and lo >= a and hi <= b]
        if not inside:
            raise ValueError(f"option span ({a},{b}) has no token of its own")
        out.append(inside[-1])
    return out


def fallback_preset(name: str, backend: str | None, fitted: dict, asset_dir: Path):
    """The vector reading a decision model routes its long questions to.

    Only the reading matters: the backbone is already loaded, so this preset carries no repo. `name` and
    `asset_dir` must be the ones the centers were fitted under, or a "generic" center finds no asset and
    degrades to the option mean.
    """
    from .presets import Formulation, Preset
    return Preset(name=name, repo="", backend=backend, asset_dir=asset_dir,
                  formulations=tuple(Formulation(f["name"], f["template"], int(f["layer"]))
                                     for f in fitted["formulations"]),
                  tau=float(fitted["tau"]), center=fitted.get("center", "options"),
                  latency_ms="?", quality="vector fallback of a decision model", method="vector")


def spec_source(repo: str) -> str | None:
    """Where this model's decision.json lives: the directory itself, or the Hub repo holding one.

    Returns None for an ordinary model, which is then fitted by `jul models add` as usual.
    """
    if Path(repo).is_dir():
        return str(Path(repo).resolve()) if (Path(repo) / SPEC_FILE).exists() else None
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo, SPEC_FILE)
    except Exception:
        return None
    return repo


class PointerReader:
    """Encodes requests in the model's format and scores options with its pointer head."""

    def __init__(self, backbone, spec: DecisionSpec):
        self.backbone, self.spec = backbone, spec
        tok = backbone.tokenizer
        self.ids = {name: tok.convert_tokens_to_ids(t) for name, t in spec.tokens.items()}
        for name, t in spec.tokens.items():
            if tok.convert_ids_to_tokens(self.ids[name]) != t:
                raise ValueError(f"delimiter {t!r} is not a single token of {backbone.name}")
        self._escape = (re.compile(spec.escape[0]), spec.escape[1]) if spec.escape else None
        self._head: tuple[np.ndarray, ...] | None = None

    @property
    def head(self) -> tuple[np.ndarray, ...]:
        """(Wq, bq, Wk, bk), then (gamma, beta) of the LayerNorm when the head has one; loaded on first use."""
        if self._head is None:
            w = np.load(self.spec.directory / self.spec.head_file)
            s = self.spec
            self._head = (w[f"{s.query}_weight"], w[f"{s.query}_bias"], w[f"{s.key}_weight"], w[f"{s.key}_bias"])
            if s.norm:
                self._head += (w[f"{s.norm}_weight"], w[f"{s.norm}_bias"])
        return self._head

    def _user(self, text: str) -> str:
        return self._escape[0].sub(self._escape[1], text) if self._escape else text

    # --- encoding -----------------------------------------------------------------------------

    def text_tokens(self, text: str) -> list[int]:
        return self.backbone.tokenizer.encode(self._user(text), add_special_tokens=self.spec.add_special_tokens)

    def _pieces(self, layout: list[str], values: dict[str, Any]) -> list[int]:
        """A layout piece is a delimiter name or a {field}: a text, or a token list already built."""
        out: list[int] = []
        for piece in layout:
            if piece.startswith("{"):
                v = values[piece[1:-1]]
                out += v if isinstance(v, list) else self.text_tokens(v)
            else:
                out.append(self.ids[piece])
        return out

    def option_texts(self, kind: str, options: list[Option]) -> tuple[list[str], list[int]]:
        """(texts in the order the model was trained on, and for each the index of the jul option).

        With a text layout a text is a whole option line, numbered from 1."""
        if self.spec.text:
            return self._option_lines(kind, options)
        if kind == "noul":
            by_key = {o.key: o for o in options}
            order = ["false", "true"]
            texts = []
            for name, key in zip(self.spec.noul_options, order):
                desc = by_key[key].description
                texts.append(self._option(name, None if desc == NOUL_DEFAULTS[key] else desc))
            return texts, [next(i for i, o in enumerate(options) if o.key == k) for k in order]
        if kind == "score":
            return [o.description for o in options], list(range(len(options)))
        return [self._option(o.key, o.description or None) for o in options], list(range(len(options)))

    def _option(self, name: str, description: str | None) -> str:
        t = self.spec.option_text
        if not description:
            return t["without_description"].format(name=name)
        return t["with_description"].format(name=name, description=render(description))

    def _option_lines(self, kind: str, options: list[Option]) -> tuple[list[str], list[int]]:
        t = self.spec.text
        if kind == "noul":
            # jul's own default descriptions stand for "none given": the model's defaults are shown instead
            by_key, defaults = {o.key: o for o in options}, t.get("noul_defaults") or {}
            pairs = [(name, defaults.get(key, "") if by_key[key].description == NOUL_DEFAULTS[key]
                      else by_key[key].description) for name, key in zip(self.spec.noul_options, ("false", "true"))]
            index = [next(i for i, o in enumerate(options) if o.key == k) for k in ("false", "true")]
        elif kind == "score":
            pairs = [(t["score_name"].format(index=i), o.description) for i, o in enumerate(options)]
            index = list(range(len(options)))
        else:
            pairs, index = [(o.key, o.description) for o in options], list(range(len(options)))
        lines = []
        for i, (name, desc) in enumerate(pairs):
            desc = render_json(desc)
            if t.get("collapse_whitespace"):
                desc = " ".join(desc.split())
            line = t["option_with_description"] if desc else t["option"]
            lines.append(line.format(number=i + 1, name=self._user(name), description=self._user(desc)))
        return lines, index

    def encode_state(self, state: Any, reserve: int = 0) -> list[int]:
        """`reserve`: tokens the call's longest branch takes of the window, when the spec gives one."""
        text = RENDERERS[self.spec.state_render](state)
        if self.spec.text:
            ids = self.backbone.tokenizer.encode(self.spec.text["prefix"].format(state=self._user(text)),
                                                 add_special_tokens=self.spec.add_special_tokens)
        else:
            ids = self._pieces(self.spec.layout["prefix"], {"state": text})
        limit = self.spec.max_state_tokens
        if self.spec.max_length:
            limit = min(limit, max(1, self.spec.max_length - reserve))
        truncation.record(f"{self.backbone.name} pointer", limit, len(ids) - limit)
        return ids[:limit]

    def encode_question(self, instructions: str, option_texts: list[str],
                        kind: str | None = None) -> tuple[list[int], int, list[int]]:
        """-> branch tokens, offset of the question token, offsets of each option token. `kind` is needed by a
        text layout only (its header)."""
        if self.spec.text:
            return self._encode_text_question(kind, instructions, option_texts)
        spans = [self._pieces(self.spec.layout["option"], {"option": t}) for t in option_texts]
        mark = self.ids[self.spec.option_token]
        branch, opt_idx = [], []
        for piece in self.spec.layout["branch"]:
            if piece == "{options}":
                for sp in spans:
                    opt_idx.append(len(branch) + len(sp) - 1 - sp[::-1].index(mark))
                    branch += sp
            else:
                branch += self._pieces([piece], {"instructions": instructions})
        self._check_branch(branch)
        q = self.ids[self.spec.question_token]
        return branch, len(branch) - 1 - branch[::-1].index(q), opt_idx

    def _encode_text_question(self, kind: str, instructions: str,
                              lines: list[str]) -> tuple[list[int], int, list[int]]:
        """The question is the branch's last token; an option, the last token inside its line."""
        t = self.spec.text
        fields = {"kind": kind, "header": t["headers"][kind],
                  "instructions": self._user(RENDERERS[self.spec.state_render](instructions))}
        before, after = t["branch"].split("{options}")
        head, sep = before.format(**fields), t["option_separator"]
        spans, cursor = [], len(head)
        for line in lines:
            spans.append((cursor, cursor + len(line)))
            cursor += len(line) + len(sep)
        branch, offsets = encode_with_offsets(self.backbone.tokenizer, head + sep.join(lines) + after.format(**fields))
        self._check_branch(branch)
        return branch, len(branch) - 1, option_token_index(offsets, spans)

    def _check_branch(self, branch: list[int]) -> None:
        if len(branch) > self.spec.max_branch_tokens:
            raise ValueError(f"question too long for {self.backbone.name}: {len(branch)} tokens "
                             f"(the model was trained on at most {self.spec.max_branch_tokens})")

    # --- scoring ------------------------------------------------------------------------------

    def logits(self, state: Any, questions: list[tuple[str, str, list[Option]]]) -> tuple[list[np.ndarray], int]:
        """questions: (kind, instructions, options). Returns logits in jul's option order, already divided
        by the model's temperature for each question's type, and the number of tokens run."""
        encoded = []
        for kind, instructions, options in questions:
            texts, index = self.option_texts(kind, options)
            encoded.append((kind, index, *self.encode_question(instructions, texts, kind)))
        prefix_tokens = self.encode_state(state, max((len(e[2]) for e in encoded), default=0))
        prefix = self.backbone.cache_prefix(prefix_tokens)
        out, spent = [], len(prefix_tokens)
        for (kind, index, branch, q_idx, opt_idx), (_, _, options) in zip(encoded, questions):
            h = self.backbone.last_hidden(branch, prefix=prefix)
            Wq, bq, Wk, bk, *norm = self.head
            hq, hk = h[q_idx], h[opt_idx]
            if norm:
                hq, hk = (_layer_norm(x, *norm, self.spec.norm_eps) for x in (hq, hk))
            qv = hq @ Wq.T + bq
            kv = hk @ Wk.T + bk
            z = (kv @ qv) / np.sqrt(self.spec.dim) / self.spec.temperature_for(kind)
            ordered = np.empty(len(options), dtype=np.float32)
            ordered[index] = z
            out.append(ordered)
            spent += len(branch)
        return out, spent
