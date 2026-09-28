"""The contrastive method: projection heads on a frozen encoder, as CLM-8B serves them.

CLM (https://github.com/Contrastive-LM/CLM, `Contrastive-LM/CLM-v0.1-8B` on the Hub) answers a typed
question without generating anything: the state, with the question's instructions appended, and every
option are embedded by a frozen decoder (Qwen3-8B, last-token pooling, L2-normalised); a state head and
an action head, two small MLPs trained with InfoNCE, project them to 512 dimensions; the score of an
option is `scale * cos(state_head(s), action_head(o))` and a softmax over the options is the answer.

The heads are useless without the exact backbone they were trained on, so a contrastive preset is the
backbone (read by jul's own backends, MLX or torch) plus a directory holding:

    contrastive.json   the format: backbone repo per backend, pooling, token limit, how options are
                       written (noul defaults), the scale
    heads.npz          both MLPs as plain arrays, so inference needs neither torch nor the CLM package

`python -m jul.contrastive convert <checkpoint.pt or Hub repo> <directory>` writes them once from a CLM
checkpoint (that step needs torch, to read the .pt); `jul models add clm-8b --repo Contrastive-LM/CLM-v0.1-8B`
does it by itself. The texts are built exactly as CLM's `schema.build_pairs` builds them, and the
encoder call reproduces what vLLM's pooling endpoint returns for Qwen3 (raw tokens, no special tokens,
the last `max_tokens` kept, hidden state of the last token after the final norm).
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import NOUL_DEFAULTS, Option

SPEC_FILE = "contrastive.json"
HEADS_FILE = "heads.npz"
#: The reference head and the backbone it was trained on, per backend. On MLX, 8-bit: on CLM's reference
#: requests (M1 Pro) it answers within 0.02 of bf16 vLLM (urgency 0.865 vs 0.84-0.85), while 4-bit moves
#: the Noul to 0.713 (same argmax). `python -m jul.contrastive convert --backbone-mlx` picks another.
CLM_REPO = "Contrastive-LM/CLM-v0.1-8B"
CLM_FILE = "CLM_v0.1-8B.pt"
CLM_BACKBONE = {"torch": "Qwen/Qwen3-8B", "mlx": "mlx-community/Qwen3-8B-8bit"}


# --- the texts, as CLM's schema.py writes them --------------------------------------------------

def to_text(x: Any, indent: int = 0) -> str:
    """CLM's rendering of a state: an object becomes `key: value` fields (top level separated by a
    blank line), an array one `- item` line per element. Not jul's `render`: the heads were trained on
    this one."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return str(x)
    pad = " " * indent
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{pad}{k}:\n{to_text(v, indent + 2)}")
            else:
                parts.append(f"{pad}{k}: {to_text(v)}")
        return ("\n\n" if indent == 0 else "\n").join(parts)
    if isinstance(x, (list, tuple)):
        parts = []
        for v in x:
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{pad}-\n{to_text(v, indent + 2)}")
            else:
                parts.append(f"{pad}- {to_text(v)}")
        return "\n".join(parts)
    return json.dumps(x, ensure_ascii=False)


def state_text(state: Any, instructions: str | None) -> str:
    """Context first, question last, a blank line between: the layout the state head was trained on."""
    s, i = to_text(state).strip(), to_text(instructions).strip()
    return f"{s}\n\n{i}" if s and i else (s or i)


def option_texts(kind: str, instructions: str | None, options: list[Option],
                 noul: dict[str, str]) -> tuple[list[str], list[int]]:
    """(candidate texts in CLM's order, and for each the index of the jul option).

    choice: the description, else the key, verbatim. score: the level descriptions. noul: CLM's order
    (false, true), each `<key>: <description>`; jul's default "Yes."/"No." counts as no description,
    so CLM's own default (built from the instructions) is used, as a CLM client would get.
    """
    if kind == "noul":
        by_key = {o.key: o for o in options}
        ins = to_text(instructions).strip()
        texts = []
        for k in ("false", "true"):
            d = by_key[k].description
            if not d or d == NOUL_DEFAULTS[k]:
                d = noul[k].format(instructions=ins) if ins else k
            texts.append(f"{k}: {d}")
        return texts, [next(i for i, o in enumerate(options) if o.key == k) for k in ("false", "true")]
    if kind == "score":
        return [to_text(o.description) for o in options], list(range(len(options)))
    return [o.description if o.description not in (None, "") else o.key for o in options], list(range(len(options)))


# --- the spec and the heads ---------------------------------------------------------------------

@dataclass(frozen=True)
class ContrastiveSpec:
    backbone: dict[str, str]
    pooling: str
    max_tokens: int
    add_special_tokens: bool
    noul: dict[str, str]
    scale: float
    heads_file: str
    source: str
    directory: Path

    @classmethod
    def load(cls, directory: str | Path) -> "ContrastiveSpec":
        directory = Path(directory)
        d = json.loads((directory / SPEC_FILE).read_text())
        if d.get("method") != "contrastive":
            raise ValueError(f"{directory / SPEC_FILE}: unsupported method {d.get('method')!r}")
        if d.get("pooling", "last") != "last":
            raise ValueError(f"{directory / SPEC_FILE}: only last-token pooling is supported")
        return cls(backbone=dict(d["backbone"]), pooling="last", max_tokens=int(d["max_tokens"]),
                   add_special_tokens=bool(d.get("add_special_tokens", False)), noul=dict(d["noul"]),
                   scale=float(d["scale"]), heads_file=d.get("heads", HEADS_FILE),
                   source=d.get("source", ""), directory=directory)


def _erf(x: np.ndarray) -> np.ndarray:
    """erf without scipy: Abramowitz-Stegun 7.1.26 is 1.5e-7 off, below float32 noise here."""
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-a * a)
    return (s * y).astype(x.dtype)


#: GELU is the exact (erf) one, torch's default, which CLM's heads use.
_ACT = {"gelu": lambda x: 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0))),
        "relu": lambda x: np.maximum(x, 0.0),
        "silu": lambda x: x / (1.0 + np.exp(-x))}


class Head:
    """`hidden -> width -> ... -> proj` MLP: CLM's `make_head`, in numpy (float32)."""

    def __init__(self, arrays: dict[str, np.ndarray], prefix: str, cfg: dict):
        g = lambda k: np.asarray(arrays[f"{prefix}.{k}"], dtype=np.float32)
        self.inp = (g("inp.weight"), g("inp.bias"))
        self.out = (g("out.weight"), g("out.bias"))
        n_hidden = int(cfg["depth"]) - 2
        self.hidden = [(g(f"hidden.{i}.weight"), g(f"hidden.{i}.bias")) for i in range(n_hidden)]
        self.norms = ([(g(f"norms.{i}.weight"), g(f"norms.{i}.bias")) for i in range(n_hidden)]
                      if cfg.get("layernorm") else [None] * n_hidden)
        self.act = _ACT[cfg.get("activation", "gelu")]
        self.residual = bool(cfg.get("residual", False))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """(n, hidden) -> (n, proj), L2-normalised."""
        x = self.act(np.asarray(x, dtype=np.float32) @ self.inp[0].T + self.inp[1])
        for (w, b), norm in zip(self.hidden, self.norms):
            h = x @ w.T + b
            if norm is not None:
                mu = h.mean(-1, keepdims=True)
                var = ((h - mu) ** 2).mean(-1, keepdims=True)
                h = (h - mu) / np.sqrt(var + 1e-5) * norm[0] + norm[1]
            h = self.act(h)
            x = x + h if self.residual else h
        z = x @ self.out[0].T + self.out[1]
        return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)


def load_heads(spec: ContrastiveSpec) -> tuple[Head, Head]:
    with np.load(spec.directory / spec.heads_file) as f:
        arrays = {k: f[k] for k in f.files}
    cfg = json.loads(str(arrays.pop("cfg")))
    return Head(arrays, "state_head", cfg), Head(arrays, "action_head", cfg)


# --- reading ------------------------------------------------------------------------------------

class ContrastiveReader:
    """Embeds with the backbone, projects with the heads, scores `scale * cos`."""

    def __init__(self, backbone, spec: ContrastiveSpec, cache_size: int = 4096):
        self.backbone, self.spec = backbone, spec
        self.state_head, self.action_head = load_heads(spec)
        self._options: OrderedDict[str, np.ndarray] = OrderedDict()   # option text -> projected vector
        self._cache_size = cache_size

    def embed(self, text: str) -> tuple[np.ndarray, int]:
        """(d,) L2-normalised last-token hidden state of `text`, and its token count."""
        tokens = self.backbone.tokenizer.encode(text, add_special_tokens=self.spec.add_special_tokens)
        tokens = tokens[-self.spec.max_tokens:] or [self.backbone.tokenizer.eos_token_id or 0]
        h = np.asarray(self.backbone.last_hidden(tokens)[-1], dtype=np.float32)
        return h / (np.linalg.norm(h) + 1e-12), len(tokens)

    def _option_vectors(self, texts: list[str]) -> tuple[np.ndarray, int]:
        spent, out = 0, []
        for t in texts:
            z = self._options.get(t)
            if z is None:
                e, n = self.embed(t)
                spent += n
                z = self.action_head(e[None])[0]
                self._options[t] = z
                while len(self._options) > self._cache_size:
                    self._options.popitem(last=False)
            else:
                self._options.move_to_end(t)
            out.append(z)
        return np.stack(out), spent

    def read(self, state: Any, kind: str, instructions: str | None,
             options: list[Option]) -> tuple[np.ndarray, np.ndarray, int]:
        """(logits in jul's option order, the state's encoder embedding, tokens run).

        The embedding is what `autotune` trains a head on: it already holds the question, since the
        state head reads `state + instructions`.
        """
        texts, index = option_texts(kind, instructions, options, self.spec.noul)
        e, spent = self.embed(state_text(state, instructions))
        zc, more = self._option_vectors(texts)
        z = self.spec.scale * (zc @ self.state_head(e[None])[0])
        ordered = np.empty(len(options), dtype=np.float32)
        ordered[index] = z
        return ordered, e, spent + more


# --- converting a CLM checkpoint ----------------------------------------------------------------

def convert(checkpoint: str, out: str | Path, backbone: dict[str, str] | None = None,
            max_tokens: int = 2048) -> Path:
    """Write contrastive.json + heads.npz from a CLM `.pt` (a path, or a Hub repo holding one).

    Needs torch, once: the checkpoint is a pickled torch dict.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("converting a CLM checkpoint reads a torch .pt: pip install torch "
                          "(inference afterwards does not need it)") from exc
    path = Path(checkpoint)
    source = str(checkpoint)
    if not path.is_file():
        from huggingface_hub import HfApi, hf_hub_download
        repo = checkpoint
        files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".pt")]
        if not files:
            raise ValueError(f"{repo}: no .pt checkpoint in the repo")
        info = HfApi().model_info(repo)
        path = Path(hf_hub_download(repo, CLM_FILE if CLM_FILE in files else files[0]))
        source = f"{repo}@{info.sha}/{path.name}"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    for k in ("hidden_size", "projection_dim"):
        if k in ck:
            cfg[k] = ck[k]
    arrays = {f"{side}.{k}": v.float().numpy() for side in ("state_head", "action_head")
              for k, v in ck[side].items()}
    scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))
    if backbone is None:
        base = cfg.get("model", "Qwen/Qwen3-8B")
        backbone = {"torch": base, **({"mlx": CLM_BACKBONE["mlx"]} if base == CLM_BACKBONE["torch"] else {})}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / HEADS_FILE, cfg=np.array(json.dumps(cfg)), **arrays)
    spec = {"method": "contrastive", "source": source, "backbone": backbone, "pooling": "last",
            "max_tokens": max_tokens, "add_special_tokens": False, "scale": scale,
            "noul": {"true": "Yes. This is true: {instructions}", "false": "No. This is false: {instructions}"},
            "heads": HEADS_FILE}
    (out / SPEC_FILE).write_text(json.dumps(spec, indent=1) + "\n")
    return out


def is_clm_repo(repo: str) -> bool:
    """A local directory with a contrastive.json, or a Hub repo whose config.json says `model_type: clm`."""
    if Path(repo).is_dir():
        return (Path(repo) / SPEC_FILE).exists()
    if Path(repo).suffix == ".pt":
        return Path(repo).is_file()
    try:
        from huggingface_hub import hf_hub_download
        config = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    except Exception:
        return False
    return config.get("model_type") == "clm"


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m jul.contrastive",
                                 description="Convert a CLM checkpoint for jul's contrastive method.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("checkpoint", help=f"a .pt file or a Hub repo (e.g. {CLM_REPO})")
    c.add_argument("out")
    c.add_argument("--backbone-torch")
    c.add_argument("--backbone-mlx")
    c.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args(argv)
    backbone = {k: v for k, v in (("torch", a.backbone_torch), ("mlx", a.backbone_mlx)) if v} or None
    print(convert(a.checkpoint, a.out, backbone, a.max_tokens))


if __name__ == "__main__":
    main()
