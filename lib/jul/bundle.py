"""Bundles: a fixed need, packed once, answered with the state as the only input.

`pack` runs everything that does not depend on the state and writes it to a directory, a bundle:
the rendered prompt of every pass (formulation x question), its layer, its option vectors and its
center, and the tuned head or the calibration a context holds for the question. `Bundle`
loads that directory on a backbone and answers `system_one(state)` / `system_one_batch(states)` in
the format of `TypeSafeClient.system_one`, without preparing anything again. Each distinct prompt
prefix is run once at load and kept as the backbone's prefix cache (a KV cache on torch, MLX and
the onnx exports that have one), so a call pays for the state's tokens only; a prompt shared by
several questions (the "one word" formulation) is read once per state.

Packing is not tied to a backend: the bundle names the backend its vectors were computed with,
and loading it on another one warns, since the vectors differ between weights (MLX 4-bit against
torch bf16 is ~0.95 cosine). Pack on the backend you deploy on, or on one whose vectors match it
(the onnx 8-bit export matches float32 within ~0.9995).

    jul pack bundle/ --questions questions.yaml --context tickets --backend onnx
    bundle = jul.Bundle.load("bundle/")
    bundle.system_one("I was charged twice")

A bundle holds no weights: it names the model (preset and repo) and is loaded on it, wherever that
runs. Nothing here trains or changes a model.

With a cross model in the preset (jul/cross.py), each question is packed with the reading the client
would give it: the types the cross model declares go to it, unless the question has a tuned head or a
calibration. `bundle.json` then lists which models the bundle needs (`models`: "vector", "cross"), and
`Bundle.load` loads those only: a bundle of Noul and Score questions never loads the vector model, so a
deployment ships the cross model alone. `reading="vector"` packs every question as vectors (one model).
"""

from __future__ import annotations

import json
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import tuning
from .backbone import Backbone, PromptTemplate
from .engine import normalize, softmax
from .presets import Preset
from .types import Option, SystemOneResponse, Usage, serialize_state

FORMAT = 1
#: How a question can be packed: `pack(reading=...)`.
READINGS = ("auto", "vector")
MANIFEST = "bundle.json"
ARRAYS = "arrays.npz"


@dataclass
class _Pass:
    prompt: int                    # index into Bundle.prompts
    layer: int
    centered_options: np.ndarray   # (K, d)
    center: np.ndarray             # (d,)


@dataclass
class _Question:
    name: str
    kind: str
    options: list[Option]
    passes: list[_Pass]
    head: dict | None
    calibration: tuple[float, np.ndarray] | None
    reading: str = "vector"
    instructions: str = ""


def pack(client, questions: dict, out: str | Path, context=None, model: str | None = None,
         reading: str = "auto") -> Path:
    """Freeze `questions` as `client` answers them (its preset and backend, `context`'s heads and
    calibration) into the directory `out`. `reading="vector"` reads every question as vectors, even the
    ones the preset's cross model would answer."""
    if reading not in READINGS:
        raise ValueError(f"reading must be one of {READINGS}, not {reading!r}")
    from .client import _kind_of
    from .context import resolve_context
    from .types import options_of

    ctx = resolve_context(context, client._context_home) if context is not None else client.context
    engine = client._engine_for(model)
    if engine.pointer is not None or engine.contrastive is not None:
        raise ValueError(f"{client.model!r} is read with the {client._preset.method} method: "
                         "only the vector method packs")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prompts: list[tuple[str, str]] = []
    arrays: dict[str, np.ndarray] = {}
    manifest_questions = []
    for name, question in questions.items():
        kind = _kind_of(question)
        options = options_of(question)
        if reading == "auto" and client._default_method(engine, ctx, kind, question, options) == "cross":
            manifest_questions.append({
                "name": name, "kind": kind, "instructions": question.instructions,
                "options": [{"key": o.key, "description": o.description} for o in options],
                "reading": "cross", "passes": [], "head": None, "calibration": None})
            continue
        head = client._head(ctx, kind, question, options)
        # a tuned head is read with the formulations it was trained on
        compiled = engine.compile(kind, question.instructions, options, ctx, client._head_formulations(head))
        passes = []
        for i, p in enumerate(compiled.passes):
            key = (p.template.prefix_text, p.template.suffix_text)
            if key not in prompts:
                prompts.append(key)
            arrays[f"{name}.{i}.options"] = p.centered_options
            arrays[f"{name}.{i}.center"] = p.center
            passes.append({"formulation": p.formulation.name, "layer": p.formulation.layer,
                           "prompt": prompts.index(key)})
        if head is not None:
            for k, v in head.items():
                if k != "meta" and not k.startswith("_"):
                    arrays[f"{name}.head.{k}"] = np.asarray(v)
        fitted = ctx.calibration.get(client._digest(kind, question, options)) if ctx else None
        manifest_questions.append({
            "name": name, "kind": kind, "instructions": question.instructions,
            "options": [{"key": o.key, "description": o.description} for o in options],
            "reading": "vector", "passes": passes, "head": head["meta"] if head is not None else None,
            "calibration": [float(fitted[0]), list(map(float, fitted[1]))] if fitted else None,
        })
    preset: Preset = engine.preset
    readings = {q["reading"] for q in manifest_questions}
    manifest = {"format": FORMAT, "preset": preset.to_json(), "backend": engine.backbone.backend,
                "models": [m for m in ("vector", "cross") if m in readings],
                "model_key": engine.backbone.key, "repo": engine.backbone.repo, "tau": preset.tau,
                "prompts": [{"prefix": a, "suffix": b} for a, b in prompts], "questions": manifest_questions,
                "context": getattr(ctx, "name", None), "packed": time.strftime("%Y-%m-%d")}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    np.savez(out / ARRAYS, **arrays)
    return out


class Bundle:
    """A bundle loaded on a backbone. See the module docstring."""

    def __init__(self, manifest: dict, arrays: dict, backbone: Backbone | None, cross=None):
        self.manifest = manifest
        self.backbone = backbone
        self.cross = cross
        self.tau = float(manifest["tau"])
        self.prompts = ([PromptTemplate(backbone, p["prefix"], p["suffix"]) for p in manifest["prompts"]]
                        if backbone is not None else [])
        self.questions: list[_Question] = []
        for q in manifest["questions"]:
            name = q["name"]
            head = None
            if q["head"] is not None:
                prefix = f"{name}.head."
                head = {k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)}
                head["meta"] = q["head"]
            self.questions.append(_Question(
                name=name, kind=q["kind"], options=[Option(o["key"], o["description"]) for o in q["options"]],
                passes=[_Pass(p["prompt"], p["layer"], arrays[f"{name}.{i}.options"], arrays[f"{name}.{i}.center"])
                        for i, p in enumerate(q["passes"])],
                head=head,
                calibration=(q["calibration"][0], np.array(q["calibration"][1])) if q["calibration"] else None,
                reading=q.get("reading", "vector"), instructions=q["instructions"]))
        # (prompt, layer) pairs read per state, each once whatever the number of questions using it
        self._reads = sorted({(p.prompt, p.layer) for q in self.questions for p in q.passes})

    @classmethod
    def load(cls, path: str | Path, backend: str | None = None, model: str | None = None,
             cross_model: str | None = None) -> "Bundle":
        """`model` replaces the repo or directory of the weights the bundle names (same weights, moved),
        `cross_model` the directory of its cross model. Only the models the bundle needs are loaded."""
        from .backbone import MODELS, resolve_backend
        path = Path(path)
        manifest = json.loads((path / MANIFEST).read_text())
        if manifest.get("format") != FORMAT:
            raise ValueError(f"{path}: bundle format {manifest.get('format')!r}, this jul reads {FORMAT}")
        backend = resolve_backend(backend or manifest["backend"])
        if backend != manifest["backend"]:
            warnings.warn(f"{path} was packed on {manifest['backend']}: its option vectors, centers and "
                          f"heads may not match {backend}'s vectors", stacklevel=2)
        repo = model or manifest["preset"]["repos"].get(backend) or manifest["repo"]
        name = manifest["preset"]["name"]
        MODELS[name] = {**MODELS.get(name, {}), backend: repo}
        with np.load(path / ARRAYS) as z:
            arrays = {k: z[k] for k in z.files}
        models = manifest.get("models", ["vector"])
        cross = None
        if "cross" in models:
            from . import cross as cross_model_
            spec = {"repo": cross_model, "subfolder": None} if cross_model else manifest["preset"]["cross"]
            cross = cross_model_.load(spec, backend)
        return cls(manifest, arrays, Backbone(name, backend) if "vector" in models else None, cross)

    @property
    def question_names(self) -> list[str]:
        return [q.name for q in self.questions]

    @property
    def models(self) -> list[str]:
        """The models this bundle reads with ("vector", "cross")."""
        return self.manifest.get("models", ["vector"])

    def _vectors(self, texts: list[str]) -> dict[tuple[int, int], np.ndarray]:
        """(prompt, layer) -> (n, d) last-token vectors of every text."""
        out = {}
        for prompt, layer in self._reads:
            rows = self.prompts[prompt].run_batch(texts, layers=[layer])
            out[prompt, layer] = np.stack([h[layer][: h[layer].shape[0] // 2] for h in rows])
        return out

    def _probabilities(self, q: _Question, vectors: dict, texts: list[str]) -> np.ndarray:
        centered = [normalize(vectors[p.prompt, p.layer] - p.center) for p in q.passes]
        if q.head is not None:
            return tuning.apply(q.head, np.concatenate(centered, axis=-1), texts)
        logits = np.mean([c @ p.centered_options.T for c, p in zip(centered, q.passes)], axis=0) / self.tau
        if q.calibration is not None:
            temperature, bias = q.calibration
            logits = logits / temperature + bias
        return np.stack([softmax(row) for row in logits])

    def system_one_batch(self, states: list) -> list[SystemOneResponse]:
        """`system_one` for many states, each prompt read over all of them in batches."""
        from .client import _format
        texts = [serialize_state(s) for s in states]
        vectors = self._vectors(texts)
        answers = [{} for _ in texts]
        tokens = [sum(len(self.backbone.encode(t + self.prompts[p].suffix_text)) for p in {p for p, _ in self._reads})
                  for t in texts]
        for q in self.questions:
            if q.reading == "cross":
                for i, (row, state) in enumerate(zip(answers, states)):
                    logits, spent = self.cross.logits(state, q.kind, q.instructions, q.options)
                    row[q.name] = _format(q.kind, None, q.options, softmax(logits))
                    tokens[i] += spent
                continue
            for row, probabilities in zip(answers, self._probabilities(q, vectors, texts)):
                row[q.name] = _format(q.kind, None, q.options, probabilities)
        answers = [{q.name: a[q.name] for q in self.questions} for a in answers]   # the packed order
        return [SystemOneResponse(answers=a, model=self.manifest["preset"]["name"], usage=Usage(input_tokens=n),
                                  request_id=str(uuid.uuid4())) for a, n in zip(answers, tokens)]

    def system_one(self, state) -> SystemOneResponse:
        return self.system_one_batch([state])[0]
