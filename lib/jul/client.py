"""The public API: `TypeSafeClient`, drop-in for the TypeSafe (Jev) Python SDK, plus context and tuning.

    from jul import TypeSafeClient, Choice
    client = TypeSafeClient(model="wemm-4b-4bit")
    response = client.system_one(state={"ticket": "charged twice"},
                                 questions={"team": Choice(instructions="Which team?",
                                                           criteria={"billing": "...", "tech": "..."})})
    response.choices["team"].choice

Everything runs locally: no API key, no network. The model is loaded on first use and only one is
held in memory at a time, since a single model can weigh several GB.
"""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import tuning
from .calibration import fit_temperature_bias
from .backbone import model_key, resolve_backend
from .context import Context, question_digest, resolve_context
from .engine import Engine, softmax
from .decision import fallback_preset
from .presets import Preset, formulations_for, one_word_preset, resolve
from .types import (Choice, ChoiceAnswer, Noul, NoulAnswer, Option, Question, Score, ScoreAnswer,
                    SystemOneResponse, Usage, options_of, serialize_state)

#: How each question type is read by default. Vectors everywhere, measured:
#: on 480 class-balanced yes/no examples, vectors beat letters on accuracy (0.771 vs 0.692), ranking
#: (AUC 0.938 vs 0.904) and calibration (ECE 0.148 vs 0.238); on the ordinal hand set, 7/9 vs 6/9 with
#: letters unable to reach the lowest level at all. `method="letters"` keeps the older reading.
DEFAULT_METHOD = {"choice": "vector", "noul": "vector", "score": "vector"}

_KIND = {Choice: "choice", Noul: "noul", Score: "score"}


class TypeSafeClient:
    """Local, typed decisions. Accepts the Jev SDK's constructor arguments and ignores the remote ones."""

    def __init__(self, model: str | None = None, context: Context | str | None = None,
                 method: str | None = None, one_word_only: bool = False, backend: str | None = None,
                 context_home: Path | None = None, api_key: str | None = None, base_url: str | None = None,
                 timeout: float | None = None, max_retries: int | None = None, **_ignored: Any):
        self._one_word_only = one_word_only
        #: Resolved on the first call, not here: constructing a client must not need a backend
        #: installed, nor load anything. `backend=` is remembered until then.
        self._requested_backend = backend
        self._backend: str | None = None
        self._preset = self._resolve_preset(model)
        self._engine: Engine | None = None
        self._context_home = context_home
        self.context = resolve_context(context, context_home)
        self.method = method

    # --- model handling -----------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """The backend, resolved on first access (and so on the first call)."""
        if self._backend is None:
            self._backend = resolve_backend(self._requested_backend)
        return self._backend

    def _resolve_preset(self, model: str | None) -> Preset:
        """Before the first call there may be no backend at all: fall back to the built-in preset."""
        try:
            backend = self.backend
        except ImportError:
            backend = None
        return (one_word_preset(model, backend) if self._one_word_only
                else resolve(model, backend))

    def _engine_for(self, model: str | None) -> Engine:
        preset = self._resolve_preset(model) if model else self._preset
        if self._engine is None or self._engine.preset.name != preset.name:
            self._engine = None  # drop the previous model before loading another
            self._engine = Engine(preset, backend=self.backend)
        self._engine.preset = preset
        self._preset = preset
        return self._engine

    @property
    def model(self) -> str:
        return self._preset.name

    def close(self) -> None:
        """Release the model. The Jev SDK closes an HTTP session here."""
        self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- the call -----------------------------------------------------------------------------

    def system_one(self, state: Any, questions: Mapping[str, Question], context: Context | str | None = None,
                   model: str | None = None, method: str | None = None, route_above: int | None = None,
                   **_ignored: Any) -> SystemOneResponse:
        """Answer every question about one state, in a single pass per formulation.

        With a cross model in the preset (jul/cross.py), the types it declares are read by it; `method`
        ("vector", "cross") forces one reading for every question of the call.

        `route_above` overrides, for this call, the option count above which a decision model hands a
        question to its vector reading (its decision.json sets the default; 0 disables the routing).

        `_ignored` swallows the Jev arguments that mean nothing locally (`response_model`, `retry`,
        `extra_body`, ...) so existing code keeps running.
        """
        if not questions:
            raise ValueError("system_one needs at least one question")
        ctx = resolve_context(context, self._context_home) if context is not None else self.context
        engine = self._engine_for(model)
        text = serialize_state(state)
        shared: dict[int, np.ndarray] = {}
        answers, tokens = {}, 0

        if engine.pointer is not None:
            # A decision model reads the raw state in its own format, once for all the questions. Beyond
            # `route_above` options the pointer head costs more latency than it earns,
            # so those questions go to the vector reading its decision.json describes.
            # The fallback's fitted numbers live in the preset, written by `jul models add` on these very
            # weights; a model's own decision.json may carry them too (and the default threshold).
            spec_routing = engine.pointer.spec.routing or {}
            fitted = engine.preset.routing or spec_routing.get("vector")
            above = (route_above if route_above is not None else
                     (engine.preset.routing or {}).get("above_options") or engine.pointer.spec.route_above)
            routed = ({n: q for n, q in questions.items() if len(options_of(q)) > above}
                      if above and fitted else {})
            direct = {n: q for n, q in questions.items() if n not in routed}
            if direct:
                items = [(_kind_of(q), q.instructions, options_of(q)) for q in direct.values()]
                logits, tokens = engine.pointer.logits(state, items)
                for (name, question), (kind, _, options), z in zip(direct.items(), items, logits):
                    answers[name] = _format(kind, question, options,
                                            self._calibrated(ctx, kind, question, options, z))
            if routed:
                pointer_preset, engine.preset = engine.preset, fallback_preset(
                    engine.preset.name, self.backend, fitted,
                    engine.preset.asset_dir if engine.preset.routing else engine.pointer.spec.directory)
                try:
                    for name, question in routed.items():
                        kind, options = _kind_of(question), options_of(question)
                        probabilities, spent = self._answer_probabilities(engine, kind, "vector", question,
                                                                         options, text, ctx, shared)
                        tokens += spent
                        answers[name] = _format(kind, question, options, probabilities)
                finally:
                    engine.preset = pointer_preset
            return SystemOneResponse(answers={n: answers[n] for n in questions}, model=self._preset.name,
                                     usage=Usage(input_tokens=tokens), request_id=str(uuid.uuid4()))

        if engine.contrastive is not None:
            # Projection heads on a frozen encoder (CLM-8B): one embedding of state + instructions per
            # question, the option embeddings cached. A head from `autotune` reads that same embedding.
            for name, question in questions.items():
                kind, options = _kind_of(question), options_of(question)
                logits, embedding, spent = engine.contrastive.read(state, kind, question.instructions, options)
                tokens += spent
                head = self._head(ctx, kind, question, options)
                probabilities = (tuning.apply(head, embedding) if head is not None
                                 else self._calibrated(ctx, kind, question, options, logits))
                answers[name] = _format(kind, question, options, probabilities)
            return SystemOneResponse(answers=answers, model=self._preset.name, usage=Usage(input_tokens=tokens),
                                     request_id=str(uuid.uuid4()))

        for name, question in questions.items():
            kind = _kind_of(question)
            options = options_of(question)
            how = method or self.method or self._default_method(engine, ctx, kind, question, options)
            if how == "cross":
                if engine.cross is None:
                    raise ValueError(f"{self._preset.name!r} has no cross model (preset `cross`)")
                logits, spent = engine.cross.logits(state, kind, question.instructions, options)
                tokens += spent
                answers[name] = _format(kind, question, options, softmax(logits))
                continue
            probabilities, spent = self._answer_probabilities(engine, kind, how, question, options, text,
                                                              ctx, shared)
            tokens += spent
            answers[name] = _format(kind, question, options, probabilities)

        return SystemOneResponse(answers=answers, model=self._preset.name, usage=Usage(input_tokens=tokens),
                                 request_id=str(uuid.uuid4()))

    def _default_method(self, engine: Engine, ctx: Context | None, kind: str, question: Question,
                        options: list[Option]) -> str:
        """The cross model for the types it declares, unless this question has a tuned head or a
        calibration: those were fitted on the vector reading, which then keeps answering it."""
        if engine.cross is not None and engine.cross.handles(kind):
            digest = self._digest(kind, question, options)
            if not (ctx and (digest in ctx.heads or digest in ctx.calibration)):
                return "cross"
        return DEFAULT_METHOD[kind]

    def _answer_probabilities(self, engine: Engine, kind: str, how: str, question: Question,
                              options: list[Option], text: str, ctx: Context | None,
                              shared: dict) -> tuple[np.ndarray, int]:
        # A tuned head is trained on vector features, so it pins the reading to vectors whatever
        # `method` says; otherwise a head trained by `autotune` would be silently ignored.
        head = self._head(ctx, kind, question, options)
        if how == "letters" and head is None:
            letters = engine.compile_letters(kind, question.instructions, options)
            logits, tokens = engine.letter_logits(letters, text)
            return self._calibrated(ctx, kind, question, options, logits), tokens

        compiled = engine.compile(kind, question.instructions, options, ctx, self._head_formulations(head))
        scores, features, tokens = engine.read(compiled, text, shared)
        if head is not None:
            return tuning.apply(head, features, text), tokens
        return self._calibrated(ctx, kind, question, options, scores / self._preset.tau), tokens

    def _digest(self, kind: str, question: Question, options: list[Option]) -> str:
        return question_digest(model_key(self._preset.name, self.backend), kind, question.instructions, options)

    def _head_formulations(self, head: dict | None):
        """The formulations a head was trained on (None: the preset's)."""
        names = (head or {}).get("meta", {}).get("formulations")
        return formulations_for(self._preset, names) if names else None

    def _head(self, ctx: Context | None, kind, question, options) -> dict | None:
        return ctx.heads.get(self._digest(kind, question, options)) if ctx else None

    def _calibrated(self, ctx: Context | None, kind, question, options, logits: np.ndarray) -> np.ndarray:
        if ctx:
            fitted = ctx.calibration.get(self._digest(kind, question, options))
            if fitted:
                temperature, bias = fitted
                return softmax(logits / temperature + np.asarray(bias))
        return softmax(logits)

    # --- tuning -------------------------------------------------------------------------------

    def autotune(self, context: Context | str, questions: Mapping[str, Question], labeled: list,
             model: str | None = None, save: bool = True, features: str = "vector",
             formulations: Mapping[str, list[str]] | list[str] | None = None) -> dict[str, tuning.TuningReport]:
        """Fit a per-task head (and a calibration) from labeled examples, and store it in a context.

        `labeled` is a list of `(state, {question_name: answer})`. An answer is the option key for a
        Choice, True/False for a Noul, the level index for a Score. Returns one report per question;
        a head that does not beat the zero-shot method on held-out examples is not activated.
        With a cross model in the preset, a question of a type it reads keeps it unless the head beats
        it on these examples (the head then answers, on the vectors).
        `features` is what the head reads: "vector" (the model's vectors), "lexical" (TF-IDF of the
        text) or "hybrid" (both); see jul/tuning.py. `formulations` picks the prompts the head reads,
        by name ("one_word", "question_options", "question"), for every question (a list) or per
        question (a mapping); the head remembers them, so answering and packing read the same way.
        """
        ctx = resolve_context(context, self._context_home)
        if ctx is None:
            raise ValueError("autotune needs a Context or the name of one")
        if isinstance(context, str) and ctx.name is None:
            ctx.name = context
        engine = self._engine_for(model)
        if engine.pointer is not None:
            raise ValueError(f"{self._preset.name!r} is a decision model (pointer method): autotune trains "
                             "heads on vector features and does not apply to it yet")
        states = [serialize_state(s) for s, _ in labeled]
        reports: dict[str, tuning.TuningReport] = {}
        features_mode = features
        if engine.contrastive is not None:
            if features != "vector" or formulations:
                raise ValueError(f"{self._preset.name!r} is a contrastive model: its heads read the encoder "
                                 "embedding only (features='vector', no formulations)")
            for name, question in questions.items():
                reports[name] = self._autotune_contrastive(engine, ctx, name, question, labeled)
            if save and ctx.name:
                ctx.save(home=self._context_home)
            return reports

        for name, question in questions.items():
            kind = _kind_of(question)
            options = options_of(question)
            keys = [o.key for o in options]
            index = {k: i for i, k in enumerate(keys)}
            rows = [(i, _answer_index(kind, a[name], index)) for i, (_, a) in enumerate(labeled) if name in a]
            if len(rows) < 2:
                raise ValueError(f"question {name!r} has fewer than 2 labeled examples")

            names = formulations.get(name) if isinstance(formulations, Mapping) else formulations
            chosen = formulations_for(self._preset, names) if names else None
            compiled = engine.compile(kind, question.instructions, options, ctx, chosen)
            scores, features = engine.read_many(compiled, [states[i] for i, _ in rows])
            y = np.array([label for _, label in rows])

            digest = self._digest(kind, question, options)
            # A type the cross model reads is answered by it until a head does better: the head is judged
            # against the cross model's zero-shot answers on these very examples, and no vector calibration
            # is kept (it would take the question off the cross model).
            crossed = engine.cross is not None and engine.cross.handles(kind)
            baseline = (np.stack([engine.cross.logits(labeled[i][0], kind, question.instructions, options)[0]
                                  for i, _ in rows]) if crossed else scores)
            head, report = tuning.train(features, y, baseline, keys, name, self._preset.name,
                                        texts=[states[i] for i, _ in rows], mode=features_mode)
            if crossed:
                ctx.calibration.pop(digest, None)
                report.reason += "; zero-shot here is the cross model" + ("" if head else ", which keeps the question")
            else:
                ctx.calibration[digest] = fit_temperature_bias(scores / self._preset.tau, y)
            if head is not None:
                head["meta"]["formulations"] = [f.name for f in chosen] if chosen else None
                ctx.heads[digest] = head
            else:
                ctx.heads.pop(digest, None)
            reports[name] = report

        if save and ctx.name:
            ctx.save(home=self._context_home)
        return reports

    def _autotune_contrastive(self, engine: Engine, ctx: Context, name: str, question: Question,
                              labeled: list) -> tuning.TuningReport:
        """autotune on a contrastive model: the head is trained on the encoder embedding of
        state + instructions (what the state head reads), judged against the CLM heads' own answers."""
        kind, options = _kind_of(question), options_of(question)
        index = {o.key: i for i, o in enumerate(options)}
        rows = [(s, _answer_index(kind, a[name], index)) for s, a in labeled if name in a]
        if len(rows) < 2:
            raise ValueError(f"question {name!r} has fewer than 2 labeled examples")
        read = [engine.contrastive.read(s, kind, question.instructions, options) for s, _ in rows]
        scores = np.stack([z for z, _, _ in read])
        features = np.stack([e for _, e, _ in read])
        y = np.array([label for _, label in rows])
        digest = self._digest(kind, question, options)
        head, report = tuning.train(features, y, scores, [o.key for o in options], name, self._preset.name)
        ctx.calibration[digest] = fit_temperature_bias(scores, y)
        if head is not None:
            ctx.heads[digest] = head
        else:
            ctx.heads.pop(digest, None)
        return report


class AsyncTypeSafeClient(TypeSafeClient):
    """The same API, awaitable. MLX work runs in a worker thread so the event loop keeps turning."""

    async def system_one(self, *args: Any, **kwargs: Any) -> SystemOneResponse:  # type: ignore[override]
        import asyncio
        return await asyncio.to_thread(TypeSafeClient.system_one, self, *args, **kwargs)

    async def autotune(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        import asyncio
        return await asyncio.to_thread(TypeSafeClient.autotune, self, *args, **kwargs)

    async def close(self) -> None:  # type: ignore[override]
        TypeSafeClient.close(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()


# --- helpers ---------------------------------------------------------------------------------

def _kind_of(question: Question) -> str:
    kind = _KIND.get(type(question))
    if kind is None:
        raise TypeError(f"Questions must be Choice, Noul or Score (got {type(question).__name__})")
    return kind


def _answer_index(kind: str, answer: Any, index: Mapping[str, int]) -> int:
    if kind == "noul":
        if isinstance(answer, str):
            answer = answer.strip().lower() in {"true", "yes", "1"}
        return index["true"] if answer else index["false"]
    key = str(answer)
    if key not in index:
        raise ValueError(f"Answer {answer!r} is not one of the options {list(index)}")
    return index[key]


def _format(kind: str, question: Question, options: list[Option], probabilities: np.ndarray):
    keys = [o.key for o in options]
    probs = {k: round(float(p), 4) for k, p in zip(keys, probabilities)}
    if kind == "noul":
        return NoulAnswer(noul=probs["true"])
    if kind == "score":
        levels = np.arange(len(options))
        mean = float((levels * probabilities).sum())
        std = math.sqrt(max(0.0, float((probabilities * (levels - mean) ** 2).sum())))
        max_std = (len(options) - 1) / 2
        return ScoreAnswer(score=round(mean, 4),
                           legend={o.key: o.description for o in options},
                           probabilities=probs,
                           confidence=round(1 - std / max_std, 4) if max_std else 1.0)
    best = int(np.argmax(probabilities))
    return ChoiceAnswer(choice=keys[best], probabilities=probs,
                        confidence=round(float(probabilities[best]), 4))
