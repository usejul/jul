"""Backbone: one forward pass, tap hidden states at chosen layers, reuse a cached prompt prefix.

The instruction part of every prompt is identical across calls, so its KV cache is computed once
and each query only pays for its own tokens. When only intermediate layers are needed, the forward
stops right after the deepest one (the remaining layers are never computed).

The framework lives behind `Backbone`: `backends/mlx.py` (Apple Silicon) and `backends/torch.py`
(transformers: CUDA, CPU, MPS) and `backends/onnx.py` (ONNX Runtime on CPU, for deployments without
torch). Everything above this module only sees numpy arrays.
`Backbone(name)` picks the backend from `JUL_BACKEND`, else MLX when available, else torch.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: onnx and api are never picked by default: onnx reads a model exported for it (jul/backends/onnx_export.py),
#: api an embeddings endpoint (jul/backends/api.py).
BACKENDS = ("mlx", "torch", "onnx", "api")

#: Preset name -> repo per backend. A name missing here is used as the repo itself.
MODELS: dict[str, dict[str, str]] = {
    "wemm-4b-4bit": {"mlx": "usejul/WeMM-Embedding-4B-mlx-4bit", "torch": "tencent/WeMM-Embedding-4B"},
    "jul-decision-wemm-4b": {"mlx": "usejul/WeMM-Embedding-4B-mlx-4bit", "torch": "tencent/WeMM-Embedding-4B"},
    "minicpm5-2b": {"mlx": "openbmb/MiniCPM5-2B-MLX", "torch": "openbmb/MiniCPM5-2B"},
    "minicpm5-1b": {"mlx": "openbmb/MiniCPM5-1B-MLX", "torch": "openbmb/MiniCPM5-1B"},
}

#: Size of a group in `PromptTemplate.run_batch`: rows x longest prompt (cached prefix included).
#: `JUL_BATCH_TOKENS` / `JUL_BATCH_SIZE` override them, read at each call like the other JUL_* variables.
BATCH_TOKENS = 16384
BATCH_SIZE = 64

# Layers tapped during extraction, as fractions of the model depth.
DEFAULT_LAYER_FRACTIONS = (0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _mlx_available() -> bool:
    return (platform.system() == "Darwin" and platform.machine() == "arm64"
            and importlib.util.find_spec("mlx") is not None)


def resolve_backend(backend: str | None = None) -> str:
    backend = backend or os.environ.get("JUL_BACKEND")
    if backend:
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. Available: {', '.join(BACKENDS)}")
        return backend
    if _mlx_available():
        return "mlx"
    if importlib.util.find_spec("torch") is not None:
        return "torch"
    raise ImportError("No backend installed: pip install 'jul[mlx]' (Apple Silicon) or 'jul[torch]'")


def model_key(name: str, backend: str) -> str:
    """Identifies vectors computed by a model on a backend (saved centers, heads, calibrations).

    MLX keeps the bare name so that contexts saved before backends existed stay valid.
    """
    return name if backend == "mlx" else f"{name}@{backend}"


def repo_for(name: str, backend: str) -> str:
    repos = MODELS.get(name)
    if repos is None:
        return name
    if backend not in repos:
        raise ValueError(f"{name!r} has no {backend} repo")
    return repos[backend]


class Backbone:
    """A model read by the vector method. `Backbone(name)` returns the resolved backend's subclass.

    Subclasses set `tokenizer` (a Hugging Face tokenizer, for `encode` and the chat template) and
    `n_layers`, and implement `forward` and `cache_prefix`. A causal LM by default; an encoder
    (jul/encoder.py) sets `architecture = "encoder"` and its input convention in `text_prefix`.
    """

    backend: str = ""
    architecture: str = "decoder"
    text_prefix: str = ""

    def __new__(cls, name: str, backend: str | None = None, **kwargs):
        if cls is Backbone:
            backend = resolve_backend(backend)
            if backend == "mlx":
                from .backends.mlx import MLXBackbone as cls
            elif backend == "onnx":
                from .backends.onnx import ONNXBackbone as cls
            elif backend == "api":
                from .backends.api import APIBackbone as cls
            else:
                from .backends.torch import torch_class
                cls = torch_class(repo_for(name, "torch"))  # noqa: PLW0642
        return super().__new__(cls)

    def __init__(self, name: str, backend: str | None = None):
        self.name = name
        self.repo = repo_for(name, self.backend)
        self.key = model_key(name, self.backend)

    @property
    def templates(self) -> dict[str, str]:
        """The prompts `jul models add` fits and `formulations_for` picks from, by formulation name."""
        if self.architecture in ("encoder", "embedding"):
            from .encoder import templates
            return templates(self.text_prefix)
        from .presets import ONE_WORD, QUESTION, QUESTION_OPTIONS
        return {"one_word": ONE_WORD, "question_options": QUESTION_OPTIONS, "question": QUESTION}

    def layer_indices(self, fractions=DEFAULT_LAYER_FRACTIONS) -> list[int]:
        return sorted({max(0, min(self.n_layers - 1, round(f * self.n_layers) - 1)) for f in fractions})

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def forward(self, tokens: list[int], layers=(), logits=False, pool: tuple[int, int] | None = None,
                prefix=None) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
        """Run tokens through the model, after `prefix` (from `cache_prefix`) when given.

        Returns ({layer: (2d,) float32 features}, float32 last-token logits or None). Features are the
        last-token hidden state concatenated with the mean hidden state over positions
        pool=(start, end) of `tokens`. The prefix is left as it was, ready for the next query.
        """
        raise NotImplementedError

    def forward_batch(self, queries: list[list[int]], layers=(), pools: list | None = None,
                      prefix=None) -> list[dict[int, np.ndarray]]:
        """The features of `forward` for each query, all after the same `prefix`. A backend may run
        them in one forward; this default runs them one by one."""
        pools = pools or [None] * len(queries)
        return [self.forward(q, layers=layers, pool=p, prefix=prefix)[0] for q, p in zip(queries, pools)]

    def cache_prefix(self, tokens: list[int]):
        """Run `tokens` once and keep the model state after them, for `forward(prefix=...)`."""
        raise NotImplementedError

    def last_hidden(self, tokens: list[int], prefix=None) -> np.ndarray:
        """(len(tokens), d) float32: the last layer's hidden states after the final norm, for every token
        of `tokens` (run after `prefix` when given). The prefix is left as it was. Used by the pointer
        method (jul/decision.py)."""
        raise NotImplementedError

    @property
    def model_dir(self) -> Path:
        """Local directory of the weights (downloaded on first access for a Hub repo)."""
        if Path(self.repo).is_dir():
            return Path(self.repo)
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(self.repo))


def with_prefix(prefix: tuple, queries, pools):
    """Each query behind its prefix, and its pooled positions moved past the prefix."""
    n = len(prefix)
    seqs = [list(prefix) + list(q) for q in queries]
    return seqs, [(n + p[0], n + p[1]) if p else (n, n + len(q)) for q, p in zip(queries, pools)]


@dataclass
class PromptTemplate:
    """A chat prompt split around the user input: a cached prefix and a per-query suffix."""

    backbone: Backbone
    prefix_text: str
    suffix_text: str
    use_prefix_cache: bool = True

    SENTINEL = "⁣QF_INPUT⁣"

    @classmethod
    def from_user_message(cls, backbone: Backbone, message: str, **kw) -> "PromptTemplate":
        """`message` must contain {input}; it is rendered with the model's chat template."""
        rendered = backbone.tokenizer.apply_chat_template(
            [{"role": "user", "content": message.replace("{input}", cls.SENTINEL)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix, suffix = rendered.split(cls.SENTINEL)
        return cls(backbone, prefix, suffix, **kw)

    def __post_init__(self):
        self.prefix_tokens = self.backbone.encode(self.prefix_text)
        self._n_suffix = len(self.backbone.encode(self.suffix_text))
        self._prefix = None
        if self.use_prefix_cache and self.prefix_tokens:
            self._prefix = self.backbone.cache_prefix(self.prefix_tokens)

    def run(self, text: str, layers=(), logits=False):
        query = self.backbone.encode(text + self.suffix_text)
        n_input = max(1, len(query) - self._n_suffix)
        if self._prefix is None:
            p = len(self.prefix_tokens)
            return self.backbone.forward(self.prefix_tokens + query, layers=layers, logits=logits, pool=(p, p + n_input))
        return self.backbone.forward(query, layers=layers, logits=logits, pool=(0, n_input), prefix=self._prefix)

    def run_batch(self, texts: list[str], layers=()) -> list[dict[int, np.ndarray]]:
        """The features of `run` for each text. Texts are sorted by length and grouped so that a group
        holds at most BATCH_TOKENS tokens (rows x longest prompt) and BATCH_SIZE rows."""
        queries = [self.backbone.encode(t + self.suffix_text) for t in texts]
        n_inputs = [max(1, len(q) - self._n_suffix) for q in queries]
        p = len(self.prefix_tokens)
        if self._prefix is None:
            seqs, pools = [self.prefix_tokens + q for q in queries], [(p, p + n) for n in n_inputs]
        else:
            seqs, pools = queries, [(0, n) for n in n_inputs]
        max_tokens = int(os.environ.get("JUL_BATCH_TOKENS") or BATCH_TOKENS)
        max_rows = int(os.environ.get("JUL_BATCH_SIZE") or BATCH_SIZE)
        out: list = [None] * len(seqs)
        group: list[int] = []

        def flush():
            got = self.backbone.forward_batch([seqs[i] for i in group], layers=layers,
                                              pools=[pools[i] for i in group], prefix=self._prefix)
            for i, features in zip(group, got):
                out[i] = features

        for i in sorted(range(len(seqs)), key=lambda i: len(seqs[i])):
            longest = len(seqs[i]) + (p if self._prefix is not None else 0)
            if group and ((len(group) + 1) * longest > max_tokens or len(group) >= max_rows):
                flush()
                group = []
            group.append(i)
        if group:
            flush()
        return out


def timed(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t
