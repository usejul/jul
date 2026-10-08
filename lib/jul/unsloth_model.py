"""Unsloth decision models as JuL models: `TypeSafeClient(model="unsloth:<checkpoint>")`.

Unsloth (https://github.com/unslothai/unsloth) trains decision models: an LLM (Qwen3.5, Llama, Gemma...) with a
LoRA and a small Clef-style head that scores every option of every question in one pass, without generating
text (guide: https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth). It also fine-tunes
Laya and Clef. JuL does not read these weights: it hands the questions to Unsloth's
`FastDecisionModel.predict`, which takes the System One body, and returns its answers as JuL's typed answers.
So everything that takes a model name takes them: `jul ask`, `jul run`, `jul serve`, `jul bench`, a tier of an
`Escalation`.

    pip install 'jul[unsloth]'
    TypeSafeClient(model="unsloth:./qwen-decisions")         # a directory written by save_pretrained(_merged)
    TypeSafeClient(model="unsloth:someone/qwen-decisions")   # or a Hub repo

What JuL's own readings add does not apply: no `method`, no heads or calibration from a context, no
`autotune`, no `pack`. The answers are Unsloth's own, `confidence` included (calibrated by
`FastDecisionModel.calibrate`). Its extra fields (`answer`, and the probabilities of a noul) are dropped, and
it does not report token counts, so `usage` stays at 0. Unsloth runs on an NVIDIA, AMD or Intel GPU; it does
not run decision models on a Mac.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

from .escalate import _answer, _question_payload
from .types import (Choice, ChoiceAnswer, Noul, NoulAnswer, Question, Score, ScoreAnswer, SystemOneResponse,
                    Usage)

__all__ = ["UnslothModel", "is_unsloth"]

PREFIX = "unsloth:"
#: The files that say a checkpoint carries a trained decision head: Clef's head (what Unsloth trains on an
#: LLM, and Clef itself) or Laya's. Without one, Unsloth would put a fresh, untrained head on the model.
HEAD_FILES = ("joint_head_config.json", "rl_agent_config.json")
_ANSWER_TYPE = {Choice: ChoiceAnswer, Noul: NoulAnswer, Score: ScoreAnswer}


def is_unsloth(model: str | None) -> bool:
    return isinstance(model, str) and model.startswith(PREFIX)


def _checkpoint(model: str) -> str:
    """`unsloth:<directory or owner/repo>` -> the directory or the repo."""
    suffix = model[len(PREFIX):].strip()
    if not suffix or "/" not in suffix:
        raise ValueError(f"Unknown Unsloth checkpoint {model!r}: use unsloth:<directory> or unsloth:<owner/repo> "
                         "(a decision model saved by Unsloth)")
    return suffix


def _has_head(checkpoint: str) -> bool | None:
    """Whether the checkpoint has a decision head; None when it cannot be told (offline, no access)."""
    root = Path(checkpoint).expanduser()
    if root.is_dir():
        return any((root / name).is_file() for name in HEAD_FILES)
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415 - installed with Unsloth
        files = HfApi().list_repo_files(checkpoint)
    except Exception:  # noqa: BLE001 - offline, gated, private: let Unsloth say what is missing
        return None
    return any(f.rsplit("/", 1)[-1] in HEAD_FILES for f in files)


class UnslothModel:
    """One Unsloth decision checkpoint, loaded on the first call (constructing it needs neither `unsloth` nor
    torch)."""

    runtime = "Unsloth"
    package = "unsloth"

    def __init__(self, model: str, backend: str | None = None):
        if backend not in (None, "torch"):
            raise ValueError(f"{model!r} runs on Unsloth's own PyTorch runtime, not on backend {backend!r}")
        self.name = model
        self.checkpoint = _checkpoint(model)
        self._loaded: tuple[Any, Any, Any] | None = None
        self._lock = threading.Lock()

    def _load(self) -> tuple[Any, Any, Any]:
        with self._lock:
            if self._loaded is None:
                # Unsloth prints its banner and patch notes on stdout: keep them on stderr, so that `jul ask`
                # prints only its JSON there.
                with contextlib.redirect_stdout(sys.stderr):
                    try:
                        from unsloth import FastDecisionModel  # noqa: PLC0415 - optional dependency
                    except ImportError as e:
                        raise ImportError(f"{self.name!r} needs the unsloth package: pip install 'jul[unsloth]'") from e
                    if _has_head(self.checkpoint) is False:
                        raise ValueError(f"{self.checkpoint!r} is not an Unsloth decision model (no decision head: "
                                         f"none of {', '.join(HEAD_FILES)}); train one with Unsloth first")
                    model, tokenizer = FastDecisionModel.from_pretrained(self.checkpoint)
                    FastDecisionModel.for_inference(model)
                self._loaded = (FastDecisionModel, model, tokenizer)
            return self._loaded

    def close(self) -> None:
        self._loaded = None

    def system_one(self, state: Any, questions: Mapping[str, Question], **_ignored: Any) -> SystemOneResponse:
        if not questions:
            raise ValueError("system_one needs at least one question")
        payload = {n: _question_payload(q) for n, q in questions.items()}  # validates the types too
        fast, model, tokenizer = self._load()
        data = fast.predict(model, tokenizer, state, payload)
        answers = {}
        for name, question in questions.items():
            try:
                answer = _answer(data[name])
            except (KeyError, TypeError, ValueError) as e:
                raise RuntimeError(f"{self.name}: malformed answer for {name!r}: {e!r}") from e
            if type(answer) is not _ANSWER_TYPE[type(question)]:
                raise RuntimeError(f"{self.name}: {name!r} was answered with the wrong type")
            answers[name] = answer
        return SystemOneResponse(answers=answers, model=self.name, usage=Usage(), request_id=str(uuid.uuid4()))
