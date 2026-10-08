"""Laya (https://huggingface.co/convaiinnovations/laya) as a JuL model: `TypeSafeClient(model="laya")`.

Laya is a decision model of its own (an encoder and a head trained by ConvAI Innovations), in its own
format, run by its own package, `laya` (PyTorch). JuL does not read its weights: it hands the questions
to `laya`'s `system_one`, which already speaks the System One body, and returns its answers as JuL's
typed answers. So everything that takes a model name takes Laya: `jul ask`, `jul run`, `jul serve`,
`jul bench`, a tier of an `Escalation`.

    pip install 'jul[laya]'
    TypeSafeClient(model="laya")                       # the English checkpoint (repo root)
    TypeSafeClient(model="laya:multilingual")          # or laya:typed-decisions, laya:english
    TypeSafeClient(model="laya:/path/to/checkpoint")   # a local directory, or another Hub repo

What JuL's own readings add does not apply to it: no `method`, no heads or calibration from a context,
no `autotune`, no `pack`. The answers are Laya's own, `confidence` included: Laya calibrates it, so it
is not the top probability JuL's readings report. Its extra fields (`answer_confidence`, `action`) are
dropped. Laya picks the device (CUDA, MPS, CPU) and pins the Hub revision it downloads.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Mapping

from .escalate import _answer, _question_payload
from .types import (Choice, ChoiceAnswer, Noul, NoulAnswer, Question, Score, ScoreAnswer, SystemOneResponse,
                    Usage)

__all__ = ["LayaModel", "is_laya", "VARIANTS"]

REPO = "convaiinnovations/laya"
#: `laya:<variant>` -> (repo, subfolder): the checkpoints of `laya.DEFAULT_MODELS`, named the same.
VARIANTS: dict[str, tuple[str, str | None]] = {
    "english": (REPO, None),
    "multilingual": (REPO, "multilingual"),
    "typed-decisions": (REPO, "typed-decisions"),
}
_ANSWER_TYPE = {Choice: ChoiceAnswer, Noul: NoulAnswer, Score: ScoreAnswer}


def is_laya(model: str | None) -> bool:
    return isinstance(model, str) and (model == "laya" or model.startswith("laya:"))


def _checkpoint(model: str) -> tuple[str, str | None]:
    """`laya`, `laya:<variant>` or `laya:<repo or directory>` -> (repo or path, subfolder)."""
    suffix = model.partition(":")[2].strip()
    if not suffix:
        return VARIANTS["english"]
    if suffix in VARIANTS:
        return VARIANTS[suffix]
    if "/" in suffix:  # a Hub repo (owner/name) or a local directory
        return suffix, None
    raise ValueError(f"Unknown Laya checkpoint {model!r}: use laya, "
                     + ", ".join(f"laya:{v}" for v in VARIANTS) + ", or laya:<hub repo or directory>")


class LayaModel:
    """One Laya checkpoint, loaded on the first call (constructing it needs neither `laya` nor torch)."""

    runtime = "Laya"
    package = "laya"

    def __init__(self, model: str = "laya", backend: str | None = None):
        if backend not in (None, "torch"):
            raise ValueError(f"{model!r} runs on Laya's own PyTorch runtime, not on backend {backend!r}")
        self.name = model
        self.repo, self.subfolder = _checkpoint(model)
        self._agent: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._agent is None:
                try:
                    import laya  # noqa: PLC0415 - optional dependency
                except ImportError as e:
                    raise ImportError(f"{self.name!r} needs the laya package: pip install 'jul[laya]'") from e
                kw = {"subfolder": self.subfolder} if self.subfolder else {}
                self._agent = laya.load(self.repo, **kw)
            return self._agent

    def close(self) -> None:
        self._agent = None

    def system_one(self, state: Any, questions: Mapping[str, Question], **_ignored: Any) -> SystemOneResponse:
        if not questions:
            raise ValueError("system_one needs at least one question")
        payload = {n: _question_payload(q) for n, q in questions.items()}  # validates the types too
        data = self._load().system_one(state, payload)
        answers = {}
        for name, question in questions.items():
            try:
                answer = _answer(data["answers"][name])
            except (KeyError, TypeError, ValueError) as e:
                raise RuntimeError(f"{self.name}: malformed answer for {name!r}: {e!r}") from e
            if type(answer) is not _ANSWER_TYPE[type(question)]:
                raise RuntimeError(f"{self.name}: {name!r} was answered with the wrong type")
            answers[name] = answer
        usage = data.get("usage") or {}
        return SystemOneResponse(answers=answers, model=self.name,
                                 usage=Usage(input_tokens=int(usage.get("input_tokens") or 0),
                                             truncated_tokens=int(usage.get("truncated_tokens") or 0)),
                                 request_id=str(uuid.uuid4()))
