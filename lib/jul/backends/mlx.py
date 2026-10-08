"""MLX backend (Apple Silicon), through mlx-lm.

`forward_batch` runs several queries in one forward, as the torch backend does: padded on the right,
under the causal mask alone, since a real token never sees the padding after it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache, can_trim_prompt_cache, make_prompt_cache

from ..backbone import Backbone


class _StopForward(Exception):
    pass


#: Text-only model types written by transformers for which mlx-lm only knows the family's type. Its model for
#: the family reads a flat (text-only) config and its weight names as well (e.g. a merged `Qwen3_5ForCausalLM`
#: fine-tune: JevK5, Plumb, Quyet, spark-s1).
_MLX_MODEL_TYPE = {"qwen3_5_text": "qwen3_5"}


def _rope_fix(repo: str) -> dict | None:
    """Overrides for mlx-lm's reading of a transformers config.

    transformers 5 writes the RoPE base under `rope_parameters`; mlx-lm reads `rope_theta` and silently falls
    back to 10000 when it is missing, which gives wrong answers without any error: pass the right value when a
    converted config has only the new key. And a text-only model type mlx-lm does not list is mapped to its
    family's (`_MLX_MODEL_TYPE`)."""
    path = Path(repo) / "config.json"
    if not path.exists():
        # a Hub repo: the cached config, else fetch that one file (1 KB, before the weights)
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        cached = try_to_load_from_cache(repo, "config.json")
        try:
            path = Path(cached if isinstance(cached, str) else hf_hub_download(repo, "config.json"))
        except Exception:
            return None
    config = json.loads(path.read_text())
    fix = {}
    theta = (config.get("rope_parameters") or {}).get("rope_theta")
    if theta and "rope_theta" not in config:
        fix["rope_theta"] = theta
    if config.get("model_type") in _MLX_MODEL_TYPE:
        fix["model_type"] = _MLX_MODEL_TYPE[config["model_type"]]
    return fix or None


class _Tap:
    """Wraps a transformer block to record its features and optionally halt the forward."""

    def __init__(self, block, idx: int, backbone: "MLXBackbone"):
        self.block = block
        self.idx = idx
        self.bb = backbone

    def __call__(self, *args, **kwargs):
        h = self.block(*args, **kwargs)
        if self.idx in self.bb._want:
            pool = self.bb._pool
            last = h[mx.arange(h.shape[0]), self.bb._last]
            # mean in float32, rounded back to the model dtype like `h[:, start:end].mean(1)` was
            mean = (mx.where(pool, h, 0).astype(mx.float32).sum(1) / pool.sum(1)).astype(h.dtype)
            # [last token ; mean over the input tokens]
            self.bb._captured[self.idx] = mx.concatenate([last, mean], axis=-1)
        if self.bb._stop_at == self.idx:
            raise _StopForward
        return h

    def __getattr__(self, name):
        return getattr(self.block, name)


@dataclass
class _Prefix:
    n: int
    cache: list | None = None      # attention KV cache: trimmed back to the prefix after each query
    snapshot: list | None = None   # recurrent state: copied, each query starts from the copy


class MLXBackbone(Backbone):
    backend = "mlx"

    def __init__(self, name: str, backend: str | None = None):
        super().__init__(name)
        self.model, self.tokenizer = load(self.repo, model_config=_rope_fix(self.repo))
        layers = self.model.layers
        for i, block in enumerate(layers):
            layers[i] = _Tap(block, i, self)
        self.n_layers = len(layers)
        self._want: set[int] = set()
        self._stop_at: int | None = None
        self._captured: dict[int, mx.array] = {}
        self._last: mx.array | None = None   # (B,) index of each row's last real token
        self._pool: mx.array | None = None   # (B, T, 1) bool mask of each row's pooled positions
        pad = getattr(self.tokenizer, "pad_token_id", None)
        self._pad = pad if pad is not None else (self.tokenizer.eos_token_id or 0)
        lm = getattr(self.model, "language_model", self.model)  # multimodal wrappers (e.g. Qwen3.5) nest the text model
        self._inner = lm.model
        self._lm_head = lm.lm_head if hasattr(lm, "lm_head") else self._inner.embed_tokens.as_linear

    def forward(self, tokens, layers=(), logits=False, pool=None, prefix: _Prefix | None = None, cache=None):
        """`cache` (an mlx-lm prompt cache, advanced in place) is kept for the dev scripts."""
        if prefix is None:
            return self._first(self._run([tokens], cache, layers, logits, [pool]))
        if prefix.snapshot is not None:
            return self._first(self._run([tokens], self._restored(prefix.snapshot), layers, logits, [pool]))
        try:
            return self._first(self._run([tokens], prefix.cache, layers, logits, [pool]))
        finally:
            for c in prefix.cache:
                if c.offset > prefix.n:
                    c.trim(c.offset - prefix.n)

    def forward_batch(self, queries, layers=(), pools=None, prefix: _Prefix | None = None):
        pools = pools or [None] * len(queries)
        if len(queries) == 1 or (prefix is not None and not _repeatable(prefix)):
            # a recurrent state (or a rotating window) is not repeated over a batch: one query at a time
            return super().forward_batch(queries, layers, pools, prefix)
        cache = None
        if prefix is not None:
            # a fresh cache holding the prefix once per row; the template's own cache is not touched
            cache = make_prompt_cache(self.model)
            for new, c in zip(cache, prefix.cache):
                _repeat_into(new, c, len(queries))
        captured, _ = self._run(queries, cache, layers, False, pools)
        # every group has its own shape (rows x width): MLX would keep the freed buffers of each one
        mx.clear_cache()
        return [{k: v[i] for k, v in captured.items()} for i in range(len(queries))]

    def cache_prefix(self, tokens) -> _Prefix:
        cache = make_prompt_cache(self.model)
        self._run([tokens], cache, logits=True)
        mx.eval([c.state for c in cache])
        if can_trim_prompt_cache(cache):
            return _Prefix(len(tokens), cache=cache)
        # Recurrent state (e.g. Gated DeltaNet in Qwen3.5) cannot be trimmed: keep a copy of the state
        # after the prefix. Each query restores it into fresh containers (`_restore`), so the copy is
        # never modified.
        return _Prefix(len(tokens), snapshot=[tuple(c.state) for c in cache])

    def last_hidden(self, tokens, prefix: _Prefix | None = None) -> np.ndarray:
        self._want, self._stop_at = set(), None
        cache = None
        if prefix is not None:
            cache = self._restored(prefix.snapshot) if prefix.snapshot is not None else prefix.cache
        try:
            h = self._inner(mx.array(tokens)[None], cache=cache)[0].astype(mx.float32)
            mx.eval(h)
            return np.array(h)
        finally:
            if prefix is not None and prefix.cache is not None:
                for c in prefix.cache:
                    if c.offset > prefix.n:
                        c.trim(c.offset - prefix.n)

    def _restored(self, snapshot):
        cache = make_prompt_cache(self.model)
        _restore(cache, snapshot)
        return cache

    @staticmethod
    def _first(result):
        captured, logits = result
        return {k: v[0] for k, v in captured.items()}, (logits[0] if logits is not None else None)

    def _run(self, seqs, cache=None, layers=(), logits=False, pools=None):
        """Right-padded batch of token lists. Returns ({layer: (B, 2d)}, (B, vocab) or None)."""
        lengths = [len(s) for s in seqs]
        width = max(lengths)
        ids = np.full((len(seqs), width), self._pad, dtype=np.int32)
        pool = np.zeros((len(seqs), width, 1), dtype=bool)
        for i, (s, p) in enumerate(zip(seqs, pools or [None] * len(seqs))):
            ids[i, : len(s)] = s
            start, end = p or (0, len(s))
            pool[i, start:end] = True
        self._want = set(layers)
        self._last = mx.array([n - 1 for n in lengths])
        self._pool = mx.array(pool)
        self._stop_at = None if logits else (max(layers) if layers else None)
        self._captured = {}
        out = None
        try:
            h = self._inner(mx.array(ids), cache=cache)
            if logits:
                out = self._lm_head(h[mx.arange(len(seqs)), self._last]).astype(mx.float32)
        except _StopForward:
            pass
        captured = {k: v.astype(mx.float32) for k, v in self._captured.items()}
        mx.eval(list(captured.values()) + ([out] if out is not None else []))
        return ({k: np.array(v) for k, v in captured.items()},
                np.array(out) if out is not None else None)


def _repeat_into(new: KVCache, cache: KVCache, rows: int) -> None:
    """Fill the empty `new` with the keys and values of `cache`, once per row. Read through the attributes,
    not `state`: mlx-lm 0.32 made `state` (keys, values, offset), with the buffers past the offset."""
    keys, values = cache.keys[..., : cache.offset, :], cache.values[..., : cache.offset, :]
    new.update_and_fetch(mx.repeat(keys, rows, axis=0), mx.repeat(values, rows, axis=0))


def _restore(cache: list, snapshot: list) -> None:
    """Load a prefix snapshot into a fresh cache without sharing anything a query writes into. Since mlx-lm
    0.32, an ArraysCache's state holds its list of arrays (layers assign into it: `cache[1] = state`) and a
    KVCache's state its whole buffer (written in place past the offset): both are copied here."""
    for c, state in zip(cache, snapshot):
        c.state = [list(s) if isinstance(s, list) else s for s in state]
        if isinstance(c, KVCache) and c.keys is not None:
            c.keys, c.values = c.keys[..., : c.offset, :], c.values[..., : c.offset, :]


def _repeatable(prefix: _Prefix) -> bool:
    """A plain KV cache can be copied once per row of a batch; a recurrent state or a rotating window
    goes through `forward`, one query at a time."""
    return prefix.cache is not None and all(type(c) is KVCache for c in prefix.cache)
