"""Tuned settings per model. Adding a model means adding a preset, not touching the code.

Every number here was fitted on the dev datasets (yahootopics, empathetic, massive,
financialphrasebank) with `scripts/dev_fit_tau.py`, never on the Jev benchmark.

The presets below were fitted on MLX. `jul models add <name> --backend <b>` (jul/calibrate.py) fits a
preset for any model on any backend and saves it as `<name>@<backend>.json` in ~/.jul/presets; such a
file takes precedence over the built-in preset on that backend.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .home import JUL_HOME

ASSETS = Path(__file__).resolve().parent / "assets"
PRESET_HOME = JUL_HOME / "presets"

ONE_WORD = 'This text: "{state}" means in one word: "'
QUESTION_OPTIONS = '{instructions}\nPossible answers: {options}.\nText: "{state}"\nIn one word, the answer is: "'
#: The question without its options: a short prefix whatever the number of options. With a tuned head,
#: listing the options adds nothing on Banking77's 72 (0.910 without, 0.900 with, jul-lambda
#: 2026-09-25) and 3 points on emotion's 6 (0.852 against 0.884).
QUESTION = '{instructions}\nText: "{state}"\nIn one word, the answer is: "'
#: Formulations a question can be read with, by name (see `formulations_for`).
FORMULATION_NAMES = ("one_word", "question_options", "question")


@dataclass(frozen=True)
class Formulation:
    """One prompt formulation, read at one layer. `name` also names its generic-center asset."""

    name: str
    template: str
    layer: int


@dataclass(frozen=True)
class Preset:
    name: str
    repo: str                      # MLX repo ("" when the preset has none)
    formulations: tuple[Formulation, ...]
    tau: float
    latency_ms: str
    quality: str
    #: Fallback center when no context supplies task texts. "options" = mean of the option vectors,
    #: "generic" = the asset fitted on varied texts, "none" = no centering.
    center: str = "options"
    notes: str = ""
    torch_repo: str | None = None  # transformers repo; None: no torch backend for this preset
    onnx_repo: str | None = None   # directory written by jul.backends.onnx_export; None: no onnx
    #: (layer, tau) of the single-formulation "one word" variant; None: see ONE_WORD_ONLY.
    one_word: tuple[int, float] | None = None
    #: The backend the numbers were fitted on; None for the built-in presets (MLX).
    backend: str | None = None
    #: Where the generic centers live.
    asset_dir: Path = field(default=ASSETS, compare=False)
    #: What `jul models add` measured, for `jul models` to show. Not used at inference.
    calibration: dict | None = field(default=None, compare=False, hash=False)
    #: "vector" (formulations, layers, tau), "pointer": a decision model read with the format stored
    #: in its own decision.json (jul/decision.py), or "contrastive": projection heads on a frozen
    #: encoder, CLM-8B's method (jul/contrastive.py). Formulations and tau are unused by the last two.
    method: str = "vector"
    #: On a pointer preset: the vector reading a long question falls back to, fitted by `jul models add`
    #: on these very weights (formulations, tau, center) plus `above_options`. None = no routing.
    routing: dict | None = field(default=None, compare=False, hash=False)
    #: A cross model that answers some question types instead of the vector reading (jul/cross.py):
    #: {"repo": <directory or Hub repo, or {backend: repo}>, "subfolder": <optional>}. None = vectors only.
    cross: dict | None = field(default=None, compare=False, hash=False)
    #: On a "contrastive" preset (jul/contrastive.py): the directory holding contrastive.json and the
    #: heads, written by `jul models add` from a CLM checkpoint. The repos are the frozen backbone's.
    heads: str | None = None

    @property
    def layers(self) -> list[int]:
        return sorted({f.layer for f in self.formulations})

    @property
    def repos(self) -> dict[str, str]:
        return {**({"mlx": self.repo} if self.repo else {}),
                **({"torch": self.torch_repo} if self.torch_repo else {}),
                **({"onnx": self.onnx_repo} if self.onnx_repo else {})}

    def generic_center(self, formulation: Formulation, backend: str = "mlx") -> np.ndarray | None:
        """The asset fitted with this backend's weights, else the MLX one."""
        names = [center_asset_name(self.name, backend, formulation.name)]
        if backend != "mlx":
            names.append(center_asset_name(self.name, "mlx", formulation.name))
        for path in [self.asset_dir / n for n in names]:
            if path.exists():
                return np.load(path)
        return None

    # --- JSON, for the presets written by `jul models add` ------------------------------------

    def to_json(self) -> dict:
        return {"name": self.name, "repos": self.repos, "backend": self.backend,
                "formulations": [dataclasses.asdict(f) for f in self.formulations],
                "tau": self.tau, "center": self.center,
                "one_word": list(self.one_word) if self.one_word else None,
                "latency_ms": self.latency_ms, "quality": self.quality, "notes": self.notes,
                "calibration": self.calibration, "method": self.method, "routing": self.routing,
                **({"cross": self.cross} if self.cross else {}),
                **({"heads": self.heads} if self.heads else {})}

    @classmethod
    def from_json(cls, d: dict, asset_dir: Path) -> "Preset":
        repos = d["repos"]
        return cls(name=d["name"], repo=repos.get("mlx", ""), torch_repo=repos.get("torch"),
                   onnx_repo=repos.get("onnx"),
                   formulations=tuple(Formulation(**f) for f in d["formulations"]),
                   tau=d["tau"], center=d["center"],
                   one_word=tuple(d["one_word"]) if d.get("one_word") else None,
                   latency_ms=d.get("latency_ms", "?"), quality=d.get("quality", ""),
                   notes=d.get("notes", ""), backend=d.get("backend"), asset_dir=asset_dir,
                   calibration=d.get("calibration"), method=d.get("method", "vector"),
                   routing=d.get("routing"), cross=d.get("cross"), heads=d.get("heads"))


def formulations_for(preset: Preset, names) -> tuple[Formulation, ...]:
    """The preset's formulations picked by name. "question" is read at the layer of the preset's
    question_options formulation (the same kind of prompt, without the options)."""
    by_name = {f.name: f for f in preset.formulations}
    out = []
    for name in names:
        if name in by_name:
            out.append(by_name[name])
        elif name == "question" and "question_options" in by_name:
            out.append(Formulation("question", question_template(by_name["question_options"].template),
                                   by_name["question_options"].layer))
        else:
            raise ValueError(f"unknown formulation {name!r} for {preset.name!r}: expected one of "
                             f"{', '.join(sorted(set(by_name) | {'question'}))}")
    return tuple(out)


def question_template(question_options: str) -> str:
    """A "question_options" template without its options: QUESTION for QUESTION_OPTIONS, and the
    same for an encoder's (jul/encoder.py)."""
    from .encoder import OPTIONS_CLAUSES
    for clause in OPTIONS_CLAUSES:
        if clause in question_options:
            return question_options.replace(clause, "")
    raise ValueError(f"no options clause to remove in {question_options!r}")


def repo_fields(backend: str, repo: str) -> dict:
    """The Preset fields that give `repo` to `backend` and no repo to the others."""
    return {"repo": repo if backend == "mlx" else "", "torch_repo": repo if backend == "torch" else None,
            "onnx_repo": repo if backend == "onnx" else None}


def center_asset_name(name: str, backend: str, formulation: str) -> str:
    """MLX keeps the untagged name the built-in assets always had."""
    tag = "" if backend == "mlx" else f".{backend}"
    return f"{name}{tag}.{formulation}.center.npy"


def preset_path(name: str, backend: str, home: Path | None = None) -> Path:
    return (home or PRESET_HOME) / f"{name}@{backend}.json"


def load_preset(path: Path) -> Preset:
    return Preset.from_json(json.loads(path.read_text()), path.parent)


def save_preset(preset: Preset, home: Path | None = None) -> Path:
    path = preset_path(preset.name, preset.backend, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preset.to_json(), indent=2, ensure_ascii=False) + "\n")
    return path


def pointer_preset(name: str, repo: str, backend: str) -> Preset:
    """A decision model's preset: nothing to fit, its format and temperature live in its decision.json."""
    from huggingface_hub import snapshot_download

    from .decision import DecisionSpec
    spec = DecisionSpec.load(repo if Path(repo).is_dir() else snapshot_download(repo, allow_patterns=["*.json", "*.npz"]))
    return Preset(name=name, **repo_fields(backend, repo), backend=backend, formulations=(), tau=1.0,
                  latency_ms="?", quality="decision model (pointer method)", method="pointer",
                  notes=f"format and temperature ({spec.temperature:.3f}) read from {repo}/decision.json")


def contrastive_preset(name: str, heads_dir: str | Path, backend: str) -> Preset:
    """A contrastive model's preset (CLM-8B): the backbone's repos and the directory of its heads.

    Nothing is fitted: the scale comes with the heads. Every backend the heads name a backbone for is
    listed, so one `jul models add` serves MLX and torch alike.
    """
    from .contrastive import ContrastiveSpec
    spec = ContrastiveSpec.load(heads_dir)
    if backend not in spec.backbone:
        raise ValueError(f"{heads_dir}: no {backend} backbone for these heads "
                         f"(has {', '.join(spec.backbone)}); pass --backbone at conversion")
    return Preset(name=name, repo=spec.backbone.get("mlx", ""), torch_repo=spec.backbone.get("torch"),
                  backend=backend, formulations=(), tau=1.0, latency_ms="?",
                  quality="contrastive heads on a frozen encoder (CLM method)", method="contrastive",
                  heads=str(Path(heads_dir).resolve()),
                  notes=f"heads {spec.source}, scale {spec.scale:.2f}, backbone {spec.backbone.get(backend)}")


def routing_from(fitted: Preset, above_options: int) -> dict:
    """The `routing` block of a pointer preset, from a vector preset fitted on the same weights.

    The names stay the same so the generic-center assets `jul models add` just wrote are the ones the
    fallback reads (center_asset_name keys on the preset name).
    """
    return {"above_options": int(above_options),
            "formulations": [dataclasses.asdict(f) for f in fitted.formulations],
            "tau": fitted.tau, "center": fitted.center,
            "fitted": (fitted.calibration or {}).get("date", ""),
            "dev_accuracy": (fitted.calibration or {}).get("dev_accuracy")}


def fitted_presets(home: Path | None = None) -> list[Preset]:
    """The presets written by `jul models add`, then the ones shipped with the package."""
    dirs = [home or PRESET_HOME, ASSETS / "presets"]
    return [load_preset(p) for d in dirs if d.exists() for p in sorted(d.glob("*@*.json"))]


PRESETS: dict[str, Preset] = {
    # Tencent's WeMM-Embedding-4B (Qwen3.5-4B fine-tuned for embeddings), converted to MLX 4-bit, text
    # only. Fitted by `jul models add` on the dev sets: layer 31 / 31 (of 32), tau 0.0553, center
    # "options", dev accuracy 0.735 +/- 0.031. Jev bench, September 2026 (M5 Max): 0.857 zero-shot at
    # 55 ms p50, 0.897 with a tuned head (1000 labeled examples per task).
    "wemm-4b-4bit": Preset(
        name="wemm-4b-4bit",
        repo="usejul/WeMM-Embedding-4B-mlx-4bit",
        torch_repo="tencent/WeMM-Embedding-4B",
        formulations=(Formulation("one_word", ONE_WORD, 31),
                      Formulation("question_options", QUESTION_OPTIONS, 31)),
        tau=0.0553,
        latency_ms="~55",
        quality="0.857 zero-shot on the Jev bench (Jev: 0.753); 0.897 with a tuned head",
        center="options",
        notes="An embedding model read like any LLM. Banking77 and Emotion are in MTEB, which it trained "
              "on; on AG News, which it did not, it scores 0.90 against Jev's 0.91. It sorts one text "
              "into labels; on questions that read two texts together, prefer minicpm5-2b-decision.",
    ),
    # Layer 39 / 40 and both temperatures fitted on the dev sets (the combination
    # is stable over layers 38-41, 0.565-0.578). Fitting the combination's own tau lowered mean dev
    # ECE from 0.179 to 0.155.
    "minicpm5-2b": Preset(
        name="minicpm5-2b",
        repo="openbmb/MiniCPM5-2B-MLX",
        torch_repo="openbmb/MiniCPM5-2B",
        formulations=(Formulation("one_word", ONE_WORD, 39),
                      Formulation("question_options", QUESTION_OPTIONS, 40)),
        tau=0.04133,
        latency_ms="~65",
        quality="0.617 zero-shot on the Jev bench, above GLiNER (0.583); 0.757 with a tuned head",
        center="generic",
        notes="Centering measured on the dev sets: generic 0.560 > options 0.520 "
              "> none 0.500, and the task center of a Context(examples=...) is best at 0.585. The "
              "asset covers 'one_word'; 'question_options' falls back to the mean of the option "
              "vectors, which measured 0.535 overall.",
    ),
}

#: Single-formulation variants, kept because they are what the 'one word' rows of the bench measured.
ONE_WORD_ONLY: dict[str, tuple[int, float]] = {"minicpm5-2b": (39, 0.04554)}

ALIASES = {"fast": "minicpm5-2b", "accurate": "wemm-4b-4bit"}
DEFAULT_MODEL = "wemm-4b-4bit"


def resolve(name: str | None, backend: str | None = None, home: Path | None = None) -> Preset:
    """A preset fitted on `backend` (~/.jul/presets, then the package), else the built-in one."""
    name = name or DEFAULT_MODEL
    key = ALIASES.get(name, name)
    if backend:
        for path in (preset_path(key, backend, home), preset_path(key, backend, ASSETS / "presets")):
            if path.exists():
                return load_preset(path)
    if key in PRESETS:
        return PRESETS[key]
    fitted = [p for p in fitted_presets(home) if p.name == key]
    if fitted:
        if backend is None:
            return fitted[0]
        raise ValueError(f"{name!r} was fitted for {', '.join(p.backend for p in fitted)} only. "
                         f"Fit it for {backend} with: jul models add {key} --backend {backend}")
    names = sorted(set(PRESETS) | {p.name for p in fitted_presets(home)})
    raise ValueError(f"Unknown model {name!r}. Available: {', '.join(names)} "
                     f"(aliases: {', '.join(ALIASES)}). Add one with: jul models add <name> --repo <repo>")


def one_word_preset(name: str | None = None, backend: str | None = None, home: Path | None = None) -> Preset:
    """The cheaper single-pass variant of a preset: one formulation, its own temperature."""
    p = resolve(name, backend, home)
    one_word = p.one_word or ONE_WORD_ONLY.get(p.name)
    if one_word is None:
        raise ValueError(f"{p.name!r} has no fitted one-word variant")
    layer, tau = one_word
    return dataclasses.replace(p, formulations=(Formulation("one_word", ONE_WORD, layer),), tau=tau)
