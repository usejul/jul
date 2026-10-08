"""`jul setup`: from a bare install to a first answer, in one command.

Each step checks before it acts, so running it again only re-runs the final check:

1. backend  — MLX on Apple Silicon, else PyTorch; installs its extra's packages when missing.
2. preset   — the model has tuned settings for that backend (built-in, shipped, or `jul models add`).
3. weights  — downloaded once into the Hugging Face cache.
4. check    — one real decision, timed.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

from jul.backbone import BACKENDS
from jul.presets import DEFAULT_MODEL

#: Modules each backend imports; one missing means the extra is not installed.
BACKEND_MODULES = {"mlx": ("mlx", "mlx_lm"), "torch": ("torch", "transformers", "accelerate"),
                   "onnx": ("onnxruntime", "tokenizers", "jinja2"), "api": ()}
#: The files a model load reads (the mlx_lm list, plus chat templates): no .bin / .gguf / .pth twins.
WEIGHT_PATTERNS = ["*.json", "*.safetensors", "*.py", "tokenizer.model", "*.tiktoken", "*.txt", "*.jinja"]
#: An onnx export (jul/backends/onnx_export.py) holds the graph and its weights instead of safetensors.
ONNX_PATTERNS = ["*.json", "*.onnx", "*.onnx.data", "tokenizer.model", "*.tiktoken", "*.txt", "*.jinja"]


def weight_patterns(backend: str | None) -> list[str]:
    """Only the onnx backend fetches .onnx files: many Hub repos ship an onnx/ twin of their weights."""
    return ONNX_PATTERNS if backend == "onnx" else WEIGHT_PATTERNS


def _step(label: str, msg: str) -> None:
    print(f"  {label:<8} {msg}", flush=True)


def apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def default_backend() -> str:
    """Unlike `resolve_backend`, does not need the backend to be installed yet."""
    backend = os.environ.get("JUL_BACKEND") or ("mlx" if apple_silicon() else "torch")
    if backend not in BACKENDS:
        raise SystemExit(f"JUL_BACKEND={backend!r}: expected one of {', '.join(BACKENDS)}")
    return backend


def missing_modules(backend: str) -> list[str]:
    return [m for m in BACKEND_MODULES[backend] if importlib.util.find_spec(m) is None]


def extra_requirements(extra: str) -> list[str]:
    """The requirements of `jul[extra]`, read from the installed metadata, markers kept.

    `extra == "..."` is dropped from each marker: pip evaluates markers without an extra, so it would
    otherwise skip them all.
    """
    reqs = []
    for req in importlib.metadata.requires("jul") or []:
        if f'extra == "{extra}"' not in req:
            continue
        req = re.sub(rf'\s*and extra == "{extra}"', "", req)
        req = re.sub(rf';\s*extra == "{extra}"', "", req)
        reqs.append(req)
    return reqs


def ensure_backend(backend: str, install: bool) -> None:
    if backend == "mlx" and not apple_silicon():
        raise SystemExit("MLX needs Apple Silicon; use --backend torch")
    missing = missing_modules(backend)
    if not missing:
        _step("backend", f"{backend} (installed)")
        return
    hint = f"pip install -e \".[{backend}]\"  (from the jul checkout)"
    if not install:
        raise SystemExit(f"{backend} is not installed ({', '.join(missing)} missing): {hint}")
    try:
        reqs = extra_requirements(backend)
    except importlib.metadata.PackageNotFoundError:
        reqs = []
    if not reqs:
        raise SystemExit(f"jul is not installed as a package, cannot read its [{backend}] extra: {hint}")
    _step("backend", f"{backend}: installing {', '.join(missing)} ...")
    cmd = [sys.executable, "-m", "pip", "install", *reqs]
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit(f"pip failed; install the backend yourself: {hint}")
    importlib.invalidate_caches()
    if still := missing_modules(backend):
        raise SystemExit(f"{', '.join(still)} still missing after pip install: {hint}")
    _step("backend", f"{backend} (installed now)")


def ensure_preset(model: str, backend: str):
    from jul.presets import resolve
    try:
        preset = resolve(model, backend)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    repo = preset.repos.get(backend)
    if not repo and backend == "api":
        raise SystemExit(f"{preset.name!r} is not fitted on an embeddings API. Add one with: jul models add NAME "
                         f"--repo ollama:qwen3-embedding:0.6b --backend api (or openai:MODEL ...)")
    if not repo:
        raise SystemExit(f"{preset.name!r} has no {backend} weights. "
                         f"Fit it for {backend} with: jul models add {preset.name} --backend {backend}")
    origin = "fitted on " + preset.backend if preset.backend else "built-in"
    settings = ("pointer method, from its decision.json" if preset.method == "pointer"
                else "letter readout, with its own prompt and temperature" if preset.method == "letter-readout"
                else f"tau {preset.tau}, center {preset.center}")
    _step("preset", f"{preset.name} ({origin}): {settings}")
    return preset, repo


def ensure_weights(repo: str, backend: str | None = None) -> None:
    if backend == "api":
        _step("weights", f"{repo} (embeddings API, nothing to download)")
        return
    patterns = weight_patterns(backend)
    if Path(repo).is_dir():
        if not weights_cached(repo, backend):
            raise SystemExit(f"{repo} holds no {'.onnx' if backend == 'onnx' else '.safetensors'} weights")
        _step("weights", f"{repo} (local directory)")
        return
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        snapshot_download(repo, allow_patterns=patterns, local_files_only=True)
        _step("weights", f"{repo} (downloaded)")
        return
    except LocalEntryNotFoundError:
        _step("weights", f"{repo}: downloading ...")
    path = snapshot_download(repo, allow_patterns=patterns)
    _step("weights", f"{repo} -> {path}")


def weights_cached(repo: str, backend: str | None = None) -> bool:
    if backend == "api":   # an embeddings endpoint: no weights to download
        return True
    if Path(repo).is_dir():   # a local model directory (e.g. a decision model added with `jul models add`)
        return any(Path(repo).glob("*.onnx" if backend == "onnx" else "*.safetensors"))
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        return False
    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
    was_disabled = are_progress_bars_disabled()
    disable_progress_bars()   # a cache lookup, not a download: no "Fetching n files" bar
    try:
        snapshot_download(repo, allow_patterns=weight_patterns(backend), local_files_only=True)
    except LocalEntryNotFoundError:
        return False
    finally:
        if not was_disabled:
            enable_progress_bars()
    return True


def setup_command(model: str | None, backend: str | None) -> str:
    return ("jul setup" + (f" --model {model}" if model and model != DEFAULT_MODEL else "")
            + (f" --backend {backend}" if backend else ""))


def require_setup(model: str | None, backend: str | None) -> None:
    """Stops a command that loads a model when `jul setup` has not been run for it. Offline, no load."""
    from jul.backbone import resolve_backend
    runtime = _delegated(model)
    if runtime is not None:  # its own package runs it, and downloads its checkpoint on first use
        if importlib.util.find_spec(runtime.package) is None:
            raise SystemExit(f"{model} needs the {runtime.package} package. Run: {setup_command(model, None)}")
        return
    from jul.presets import resolve
    fix = setup_command(model, backend)
    try:
        resolved = resolve_backend(backend)
    except ImportError:
        raise SystemExit(f"No backend installed. Run: {fix}") from None
    if missing := missing_modules(resolved):
        raise SystemExit(f"The {resolved} backend is incomplete ({', '.join(missing)} missing). Run: {fix}")
    try:
        preset = resolve(model, resolved)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    repo = preset.repos.get(resolved)
    if repo and not weights_cached(repo, resolved):
        raise SystemExit(f"{preset.name} is not downloaded for {resolved} ({repo}). "
                         f"Run: {fix}")


def check(model: str, backend: str) -> None:
    from jul import Choice, TypeSafeClient
    question = {"team": Choice(instructions="Which team should handle this ticket?",
                               criteria={"billing": "payments, invoices, refunds",
                                         "technical": "bugs, errors, crashes"})}
    client = TypeSafeClient(model=model, backend=backend)
    t = time.perf_counter()
    client.system_one(state="The app crashes when I open settings.", questions=question)
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    answer = client.system_one(state="I was charged twice this month.", questions=question).answers["team"]
    ms = (time.perf_counter() - t) * 1000
    client.close()
    if answer.choice != "billing":
        print(f"  warning: the check answered {answer.choice!r} where 'billing' was expected", file=sys.stderr)
    _step("check", f"'I was charged twice' -> {answer.choice} ({answer.confidence:.2f}), "
                   f"{ms:.0f} ms per decision (first call {load_s:.1f}s with the load)")


def run(model: str | None, backend: str | None, install: bool = True, skip_check: bool = False) -> None:
    model = model or DEFAULT_MODEL
    runtime = _delegated(model)
    if runtime is not None:
        return setup_delegated(model, runtime, install, skip_check)
    default = default_backend()
    backend = backend or default
    print(f"jul setup: {model} on {backend}")
    ensure_backend(backend, install)
    preset, repo = ensure_preset(model, backend)
    ensure_weights(repo, backend)
    if not skip_check:
        check(preset.name, backend)
    print("\nReady. Try:\n  jul ask choice \"Which team should handle this ticket?\" "
          "-o billing:\"payments, invoices\" -o technical:\"bugs, errors\" --state \"I was charged twice\""
          + ("" if model == DEFAULT_MODEL else f" --model {model}")
          + ("" if backend == default else f" --backend {backend}"))


def _delegated(model: str | None):
    """The model's own runtime (Laya, Unsloth) when JuL does not read it, else None. Loads nothing."""
    from jul.client import delegated_model
    try:
        return delegated_model(model)
    except ValueError as exc:  # a malformed `laya:` / `unsloth:` name
        raise SystemExit(str(exc)) from None


def setup_delegated(model: str, runtime, install: bool, skip_check: bool) -> None:
    """A model on its own package (jul/laya_model.py, jul/unsloth_model.py): install it, then check one decision."""
    package = runtime.package
    print(f"jul setup: {model} ({runtime.runtime}'s own runtime)")
    hint = f"pip install 'jul[{package}]'"
    if importlib.util.find_spec(package) is None:
        if not install:
            raise SystemExit(f"{package} is not installed: {hint}")
        try:
            reqs = extra_requirements(package)
        except importlib.metadata.PackageNotFoundError:
            reqs = []
        if not reqs:
            raise SystemExit(f"jul is not installed as a package, cannot read its [{package}] extra: {hint}")
        _step("runtime", f"{package}: installing ...")
        if subprocess.run([sys.executable, "-m", "pip", "install", *reqs]).returncode != 0:
            raise SystemExit(f"pip failed; install it yourself: {hint}")
        importlib.invalidate_caches()
    _step("runtime", f"{package} (installed)")
    if not skip_check:
        check(model, None)
    print(f"\nReady. Try:\n  jul ask choice \"Which team should handle this ticket?\" "
          f"-o billing:\"payments, invoices\" -o technical:\"bugs, errors\" --state \"I was charged twice\" --model {model}")
