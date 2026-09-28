"""The vector method: read one hidden state, compare it to the options, never generate a token.

For each formulation of the preset, the state and every option are encoded by the same prompt; the
answer is the option whose vector is closest (cosine, after subtracting a center). The scores of the
formulations are averaged, then turned into probabilities by `softmax(cosine / tau)`.

Everything that does not depend on the state is computed once and cached: the prompt prefix (as a KV
cache inside `PromptTemplate`), the option vectors, and the center. A call therefore pays for its own
tokens only. The "one word" formulation does not mention the question, so its vector is computed once
per state and shared by every question of the call.

`letters` is the older reading, kept for `method="letters"` and the lab's baselines: the options are
listed in the prompt and the answer is read from the logits of the answer tokens. Vectors are the
default for every type, `Noul` and `Score` included, since they beat letters there too (docs/benchmarks.md).
"""

from __future__ import annotations

import logging
import string
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .backbone import MODELS, Backbone, PromptTemplate
from .presets import Formulation, Preset
from .types import Option

LETTERS = string.ascii_uppercase
_log = logging.getLogger(__name__)


def normalize(a: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    # Avoid division by zero for zero vectors (returns NaN, handled downstream by softmax)
    if _log.isEnabledFor(logging.DEBUG) and np.any(norm == 0):  # the stack is only built when it is logged
        import traceback
        _log.debug("zero vector encountered in normalize, shape=%s, norm=%s\n%s",
                   a.shape, norm.ravel()[:5], ''.join(traceback.format_stack()[-5:-1]))
    with np.errstate(invalid="ignore"):
        return a / norm


def softmax(z: np.ndarray) -> np.ndarray:
    # Single option: probability is always 1.0
    if z.size == 1:
        return np.ones_like(z)
    # Reject NaN inputs (e.g., from normalizing a zero vector) — caller must handle this
    if np.any(np.isnan(z)):
        raise ValueError("Cannot compute probabilities: state produced a degenerate vector "
                         "(likely empty or near-identical to the center). "
                         "Ensure the state contains meaningful text content.")
    e = np.exp(z - z.max())
    return e / e.sum()


def short_names(texts: list[str]) -> list[str]:
    """Drop the words shared by every option at the start and end.

    'This example is about sports' / '... about health' -> 'sports' / 'health'. Keeps the options
    listing short, which matters when there are 72 of them.
    """
    words = [t.split() for t in texts]
    pre = 0
    while all(len(w) > pre + 1 for w in words) and len({w[pre] for w in words}) == 1:
        pre += 1
    suf = 0
    while all(len(w) > pre + suf + 1 for w in words) and len({w[-1 - suf] for w in words}) == 1:
        suf += 1
    return [" ".join(w[pre:len(w) - suf]).strip(" .:") for w in words]


@dataclass
class Pass:
    """One formulation compiled for one question: its prompt, its option vectors and its center."""

    formulation: Formulation
    template: PromptTemplate
    centered_options: np.ndarray  # (K, d), centered and L2-normalized
    center: np.ndarray            # (d,)
    prompt_tokens: int            # tokens of the cached prefix, counted once per call

    def scores(self, vector: np.ndarray) -> np.ndarray:
        return self.centered_options @ normalize(vector - self.center)


@dataclass
class CompiledQuestion:
    options: list[Option]
    passes: list[Pass]
    shared_one_word: bool = False
    letters: "LetterPass | None" = None


@dataclass
class LetterPass:
    template: PromptTemplate
    answer_ids: np.ndarray
    prompt_tokens: int


class Engine:
    """Runs a preset's vector method on a backbone, caching everything that is state-independent."""

    def __init__(self, preset: Preset, backbone: Backbone | None = None, max_cached_questions: int = 64,
                 backend: str | None = None):
        self.preset = preset
        if backbone is None:
            MODELS[preset.name] = preset.repos  # presets are the source of truth for repos
            backbone = Backbone(preset.name, backend)
        self.backbone = backbone
        self.pointer = None
        if preset.method == "pointer":
            from .decision import DecisionSpec, PointerReader
            self.pointer = PointerReader(backbone, DecisionSpec.load(backbone.model_dir))
        self.contrastive = None
        if preset.method == "contrastive":
            from .contrastive import ContrastiveReader, ContrastiveSpec
            if not preset.heads:
                raise ValueError(f"{preset.name!r}: a contrastive preset needs its heads directory")
            self.contrastive = ContrastiveReader(backbone, ContrastiveSpec.load(preset.heads))
        self.cross = None
        if preset.cross:
            from . import cross
            self.cross = cross.load(preset.cross, backbone.backend, base=backbone)
        self._questions: OrderedDict[tuple, CompiledQuestion] = OrderedDict()
        self._max_cached = max_cached_questions
        self._one_word_templates: dict[str, PromptTemplate] = {}

    # --- prompts ------------------------------------------------------------------------------

    def _template(self, text: str, description: str | None) -> PromptTemplate:
        """`text` contains {state}; the part before it is the cached prefix."""
        if description:
            text = f"Context: {description}\n" + text
        prefix, suffix = text.split("{state}")
        return PromptTemplate(self.backbone, prefix, suffix)

    def _one_word_template(self, formulation: Formulation, description: str | None) -> PromptTemplate:
        key = description or ""
        if key not in self._one_word_templates:
            self._one_word_templates[key] = self._template(formulation.template, description)
        return self._one_word_templates[key]

    def _render(self, formulation: Formulation, instructions: str, options: list[Option]) -> str:
        if "{instructions}" not in formulation.template:
            return formulation.template
        listing = ", ".join(short_names([o.text for o in options]))
        return formulation.template.replace("{instructions}", instructions).replace("{options}", listing)

    # --- vectors ------------------------------------------------------------------------------

    def vector(self, template: PromptTemplate, layer: int, text: str) -> np.ndarray:
        """Hidden state just before the first generated word. The forward stops after `layer`."""
        h, _ = template.run(text, layers=[layer])
        return h[layer][: h[layer].shape[0] // 2]

    def vectors(self, template: PromptTemplate, layer: int, texts: list[str]) -> np.ndarray:
        """`vector` for each text, (n, d), read in batches when the backend supports it."""
        return np.stack([h[layer][: h[layer].shape[0] // 2] for h in template.run_batch(texts, layers=[layer])])

    def _center(self, template: PromptTemplate, formulation: Formulation, options_matrix: np.ndarray,
                context) -> np.ndarray:
        texts = getattr(context, "examples", None) if context is not None else None
        if texts:
            cached = context.center_for(self.backbone.key, formulation)
            if cached is None:
                cached = self.vectors(template, formulation.layer, texts).mean(0)
                context.set_center(self.backbone.key, formulation, cached)
            return cached
        if self.preset.center == "generic":
            generic = self.preset.generic_center(formulation, self.backbone.backend)
            if generic is not None:
                return generic
        if self.preset.center == "none":
            return np.zeros(options_matrix.shape[-1], dtype=options_matrix.dtype)
        return options_matrix.mean(0)

    def compile(self, kind: str, instructions: str, options: list[Option], context=None,
                formulations: tuple[Formulation, ...] | None = None) -> CompiledQuestion:
        """`formulations` replaces the preset's for this question (see presets.formulations_for)."""
        formulations = tuple(formulations or self.preset.formulations)
        key = (kind, instructions, tuple((o.key, o.description) for o in options), formulations,
               getattr(context, "cache_key", lambda: None)() if context is not None else None)
        if key in self._questions:
            self._questions.move_to_end(key)
            return self._questions[key]

        passes = []
        for f in formulations:
            shared = "{instructions}" not in f.template
            template = (self._one_word_template(f, _description(context)) if shared
                        else self._template(self._render(f, instructions, options), _description(context)))
            L = self.vectors(template, f.layer, [o.text for o in options])
            center = self._center(template, f, L, context)
            centered = L - center
            # Single option: centered vector is always zero (option == center), skip normalization
            if len(options) == 1:
                # Use a dummy normalized vector; softmax will return [1.0] anyway
                normalized = np.zeros_like(centered)
            else:
                norms = np.linalg.norm(centered, axis=-1)
                zero_mask = norms == 0
                if np.any(zero_mask):
                    for i, is_zero in enumerate(zero_mask):
                        if is_zero:
                            _log.error("option %d has zero centered vector: key=%r, text=%r, "
                                       "instructions=%r, kind=%s",
                                       i, options[i].key, options[i].text[:100] if options[i].text else '',
                                       instructions[:80], kind)
                normalized = normalize(centered)
            passes.append(Pass(f, template, normalized, center, len(template.prefix_tokens)))

        compiled = CompiledQuestion(options=options, passes=passes,
                                    shared_one_word=any("{instructions}" not in f.template
                                                        for f in formulations))
        self._questions[key] = compiled
        if len(self._questions) > self._max_cached:
            self._questions.popitem(last=False)
        return compiled

    # --- reading the answer -------------------------------------------------------------------

    def read(self, compiled: CompiledQuestion, state: str,
             shared: dict[int, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        """(mean cosine score per option, concatenated centered vectors, tokens spent).

        The centered vectors are what a tuned head is trained on, so training and inference read the
        state in exactly the same way. `shared` carries the "one word" vector across the questions of
        a call, since that formulation does not mention the question.
        """
        scores, features, tokens = [], [], 0
        for p in compiled.passes:
            is_shared = "{instructions}" not in p.formulation.template
            if is_shared and shared is not None and p.formulation.layer in shared:
                vector = shared[p.formulation.layer]
            else:
                vector = self.vector(p.template, p.formulation.layer, state)
                tokens += p.prompt_tokens + _state_tokens(p.template, self.backbone, state)
                if is_shared and shared is not None:
                    shared[p.formulation.layer] = vector
            centered = vector - p.center
            centered_norm = np.linalg.norm(centered)
            if centered_norm == 0:
                _log.warning("zero centered vector: state_len=%d, vector_norm=%.4f, "
                             "center_norm=%.4f, formulation=%s",
                             len(state), np.linalg.norm(vector), np.linalg.norm(p.center),
                             p.formulation.template[:50])
                _log.debug("zero centered vector state preview: %r", state[:200])
            normalized = normalize(centered)
            scores.append(normalized @ p.centered_options.T)
            features.append(normalized)
        return np.mean(scores, axis=0), np.concatenate(features), tokens

    def read_many(self, compiled: CompiledQuestion, states: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """`read` for many states at once: (scores (n, K), features (n, D)), without the token count."""
        scores, features = [], []
        for p in compiled.passes:
            vectors = self.vectors(p.template, p.formulation.layer, states)
            centered = vectors - p.center
            norms = np.linalg.norm(centered, axis=-1)
            zero_mask = norms == 0
            if np.any(zero_mask):
                for i, is_zero in enumerate(zero_mask):
                    if is_zero:
                        _log.error("zero centered vector in batch[%d]: state_len=%d, "
                                   "vector_norm=%.6f, center_norm=%.6f",
                                   i, len(states[i]),
                                   np.linalg.norm(vectors[i]), np.linalg.norm(p.center))
                        _log.debug("zero centered vector batch[%d] state: %r", i, states[i][:500])
            centered = normalize(centered)
            scores.append(centered @ p.centered_options.T)
            features.append(centered)
        return np.mean(scores, axis=0), np.concatenate(features, axis=-1)

    def vector_probabilities(self, compiled: CompiledQuestion, state: str, tau: float | None = None,
                             shared=None) -> tuple[np.ndarray, np.ndarray, int]:
        """(probabilities, raw cosine scores, tokens)."""
        scores, _, tokens = self.read(compiled, state, shared)
        return softmax(scores / (tau or self.preset.tau)), scores, tokens

    # --- the letters reading, for Noul and Score ----------------------------------------------

    def compile_letters(self, kind: str, instructions: str, options: list[Option]) -> LetterPass:
        markers = ([str(i) for i in range(len(options))] if kind == "score" and len(options) <= 10
                   else list(LETTERS[: len(options)]))
        if len(options) > len(markers):
            raise ValueError(f"The letters reading supports at most {len(LETTERS)} options")
        hint = ("Answer with the number of the level only." if kind == "score"
                else "Answer with the letter of the option only.")
        listing = "\n".join(f"{m}) {o.text}" for m, o in zip(markers, options))
        name = "Levels" if kind == "score" else "Options"
        message = f"{instructions}\n\n{name}:\n{listing}\n\n{hint}\n\n" + "Input: {input}"
        template = PromptTemplate.from_user_message(self.backbone, message)
        ids = []
        for m in markers:
            toks = self.backbone.encode(m)
            if len(toks) != 1:
                raise ValueError(f"Answer marker {m!r} is not a single token for {self.backbone.name}")
            ids.append(toks[0])
        if len(set(ids)) != len(ids):
            raise ValueError("Answer markers collide on this tokenizer")
        return LetterPass(template, np.array(ids), len(template.prefix_tokens))

    def letter_logits(self, letters: LetterPass, state: str) -> tuple[np.ndarray, int]:
        _, logits = letters.template.run(state, logits=True)
        tokens = letters.prompt_tokens + _state_tokens(letters.template, self.backbone, state)
        return logits[letters.answer_ids], tokens


def _description(context) -> str | None:
    if context is None:
        return None
    return context.description if getattr(context, "use_description", False) else None


def _state_tokens(template: PromptTemplate, backbone: Backbone, state: str) -> int:
    return len(backbone.encode(state + template.suffix_text))
