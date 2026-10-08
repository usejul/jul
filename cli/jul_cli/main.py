"""The `jul` command line. Built only on the public API of the library.

  jul setup                    # backend, weights and a first check for the default model
  jul ask choice "Which team should handle this ticket?" \
      -o billing:"payments, invoices" -o technical:"bugs, errors" \
      --state "I was charged twice" --model wemm-4b-4bit

  jul run questions.yaml --input tickets.jsonl --output answers.jsonl --context tickets
  jul context create tickets --description "Support tickets of an online bank" --examples sample.txt
  jul synth questions.yaml --seeds sample.jsonl --per-option 30 --output synth.jsonl
  jul autotune tickets --questions questions.yaml --labeled labeled.jsonl
  jul bench test.jsonl --train train.jsonl --models fast,accurate --output bench.json
  jul models

Every command prints the same JSON shape as the Jev API response.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from jul import Choice, Context, Noul, NoulCriteria, Score, TypeSafeClient
from jul.backbone import BACKENDS
from jul.presets import ALIASES, DEFAULT_MODEL, PRESETS

TYPES = {"choice": Choice, "noul": Noul, "score": Score}


# --- loading question files ---------------------------------------------------------------------

def load_questions(path: str | Path) -> dict:
    """A YAML or JSON file mapping a question name to `{type, instructions, criteria}`."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("PyYAML is needed for YAML question files: pip install 'jul[yaml]' "
                             "(or use a .json file)") from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: expected a mapping of question name -> question")
    return {name: _question(name, spec) for name, spec in raw.items()}


def _question(name: str, spec: dict):
    kind = str(spec.get("type", "choice")).lower()
    if kind not in TYPES:
        raise SystemExit(f"question {name!r}: unknown type {kind!r} (choice, noul or score)")
    instructions = spec.get("instructions", "")
    criteria = spec.get("criteria")
    if kind == "noul":
        if isinstance(criteria, dict):
            return Noul(instructions=instructions,
                        criteria=NoulCriteria(true=criteria.get("true", ""), false=criteria.get("false", "")))
        return Noul(instructions=instructions)
    if kind == "score":
        return Score(instructions=instructions, criteria=list(criteria or []))
    return Choice(instructions=instructions, criteria=criteria or {})


def read_jsonl(path: str | Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_examples(path: str | Path) -> list[str]:
    """One text per line (.txt) or a `text` field per line (.jsonl)."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [r["text"] if isinstance(r, dict) else str(r) for r in read_jsonl(path)]
    return [l for l in path.read_text().splitlines() if l.strip()]


# --- commands -----------------------------------------------------------------------------------

def cmd_ask(a):
    pairs = [o.split(":", 1) if ":" in o else (o, "") for o in a.option or []]
    if a.kind == "choice":
        if len(pairs) < 1:
            raise SystemExit("choice needs at least one -o option")
        question = Choice(instructions=a.instructions, criteria={k.strip(): v.strip() for k, v in pairs})
    elif a.kind == "noul":
        criteria = {k.strip(): v.strip() for k, v in pairs}
        question = Noul(instructions=a.instructions,
                        criteria=NoulCriteria(true=criteria.get("true", ""), false=criteria.get("false", ""))
                        if criteria else None)
    else:
        levels = [v.strip() or k.strip() for k, v in pairs]
        if len(levels) < 2:
            raise SystemExit("score needs at least two -o levels, lowest first")
        question = Score(instructions=a.instructions, criteria=levels)

    client = TypeSafeClient(model=a.model, backend=a.backend, context=a.context, method=a.method)
    for state in a.state:
        t = time.perf_counter()
        response = client.system_one(state=state, questions={a.kind: question})
        out = response.as_dict()
        out["latency_ms"] = round((time.perf_counter() - t) * 1000, 1)
        print(json.dumps(out, ensure_ascii=False))
    client.close()


def cmd_run(a):
    questions = load_questions(a.questions)
    client = TypeSafeClient(model=a.model, backend=a.backend, context=a.context, method=a.method)
    out = open(a.output, "w") if a.output else sys.stdout
    n, t0 = 0, time.perf_counter()
    try:
        for row in read_jsonl(a.input):
            state = row.get("state", row) if isinstance(row, dict) else row
            response = client.system_one(state=state, questions=questions)
            record = response.as_dict()
            if isinstance(row, dict) and "id" in row:
                record["id"] = row["id"]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            if a.output and n % 50 == 0:
                print(f"  {n} rows", file=sys.stderr, flush=True)
    finally:
        if a.output:
            out.close()
    rate = n / max(time.perf_counter() - t0, 1e-9)
    print(f"{n} rows in {time.perf_counter() - t0:.1f}s ({rate:.1f}/s)"
          + (f" -> {a.output}" if a.output else ""), file=sys.stderr)
    client.close()


def cmd_context(a):
    if a.action == "list":
        names = Context.list_saved()
        print("\n".join(names) if names else "no saved context")
        return
    if a.action == "create":
        examples = read_examples(a.examples) if a.examples else []
        ctx = Context(description=a.description or "", examples=examples, name=a.name,
                      use_description=a.use_description)
        if examples and not a.lazy:
            # Compile the centers now so later calls pay nothing.
            client = TypeSafeClient(model=a.model, backend=a.backend, context=ctx)
            client.system_one(state=examples[0],
                              questions={"_": Choice(instructions="warm up", criteria={"a": "a", "b": "b"})},
                              context=ctx)
            client.close()
        path = ctx.save(a.name)
        print(f"saved context {a.name!r} ({len(examples)} examples) -> {path}")
        return
    if a.action == "show":
        ctx = Context.load(a.name)
        print(json.dumps({"name": ctx.name, "description": ctx.description,
                          "description_in_prompts": ctx.use_description, "examples": len(ctx.examples),
                          "centers": sorted(ctx.centers), "tuned_questions": len(ctx.heads),
                          "calibrated_questions": len(ctx.calibration)}, indent=2))
        return
    if a.action == "delete":
        print(f"deleted {a.name!r}" if Context.delete(a.name) else f"no context named {a.name!r}")


def read_labeled(path: str | Path, questions: dict, require_answers: bool = True) -> list[tuple]:
    """`(state, answers)` pairs from JSONL lines `{state|text, answers: {question: answer}}`."""
    labeled = []
    for row in read_jsonl(path):
        if isinstance(row, str):
            row = {"state": row}
        state = row.get("state", row.get("text"))
        answers = row.get("answers") or {k: v for k, v in row.items() if k in questions}
        if state is None or (require_answers and not answers):
            raise SystemExit("each labeled line needs a `state` (or `text`) and one answer per question")
        labeled.append((state, answers))
    return labeled


def cmd_autotune(a):
    questions = load_questions(a.questions)
    labeled = read_labeled(a.labeled, questions)
    try:
        ctx = Context.load(a.context)
    except FileNotFoundError:
        ctx = Context(name=a.context)
    client = TypeSafeClient(model=a.model, backend=a.backend, context=ctx)
    print(f"tuning on {len(labeled)} labeled examples with {a.model or DEFAULT_MODEL} ...", file=sys.stderr)
    reports = client.autotune(ctx, questions, labeled, features=a.features)
    for report in reports.values():
        print(report)
        print()
    client.close()


def cmd_pack(a):
    from jul.bundle import MANIFEST, pack
    questions = load_questions(a.questions)
    ctx = Context.load(a.context) if a.context else None
    client = TypeSafeClient(model=a.model, backend=a.backend, context=ctx)
    path = pack(client, questions, a.out, context=ctx, reading=a.reading)
    manifest = json.loads((path / MANIFEST).read_text())
    print(f"packed {len(questions)} question(s) on {manifest['preset']['name']} ({manifest['backend']}) "
          f"into {path}: {len(manifest['prompts'])} prompt(s) per state; models to ship: "
          + " + ".join(manifest["models"]))
    for q in manifest["questions"]:
        how = ("cross model" if q.get("reading") == "cross" else
               f"{q['head'].get('features', 'vector')} head" if q["head"] else
               "calibrated zero-shot" if q["calibration"] else "zero-shot")
        print(f"  {q['name']}: {q['kind']}, {len(q['options'])} options, {how}")
    client.close()


def cmd_synth(a):
    from jul.synth import mlx_writer, synthesize
    questions = load_questions(a.questions)
    seeds = read_labeled(a.seeds, questions, require_answers=False) if a.seeds else []
    if not seeds:
        print("no seeds: texts are written from the questions alone", file=sys.stderr)
    print(f"loading writer {a.writer} ...", file=sys.stderr)
    write = mlx_writer(a.writer, temperature=a.temperature)
    t0 = time.perf_counter()
    with open(a.output, "w") as out:
        def save(rows):
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            if rows:
                print(f"  {next(iter(rows[0]['answers'].items()))}: {len(rows)} texts", file=sys.stderr)
        rows = synthesize(questions, seeds, a.per_option, write, batch=a.batch, seed=a.seed, on_option=save)
    print(f"{len(rows)} texts in {time.perf_counter() - t0:.0f}s -> {a.output}", file=sys.stderr)


def cmd_models(a):
    if a.action == "add":
        return cmd_models_add(a)
    from jul.presets import fitted_presets
    try:
        from huggingface_hub import scan_cache_dir
        cached = {r.repo_id for r in scan_cache_dir().repos}
    except Exception:
        cached = set()
    rows = [(name, "mlx", p) for name, p in PRESETS.items()]
    rows += [(p.name, p.backend, p) for p in fitted_presets()]
    print(f"{'preset':<14} {'backend':<8} {'downloaded':<11} {'latency':<10} quality")
    for name, backend, p in rows:
        repo = p.repos.get(backend)
        mark = "yes" if repo in cached else "no"
        print(f"{name:<14} {backend:<8} {mark:<11} {p.latency_ms + ' ms':<10} {p.quality}")
    print("\naliases: " + ", ".join(f"{a} -> {t}" for a, t in ALIASES.items()))
    for name, backend, p in rows:
        print(f"\n{name} ({backend}): " + ", ".join(f"{b} {r}" for b, r in p.repos.items()))
        if p.method == "pointer":
            print("  method: pointer (decision model; format, head and temperature in its decision.json)")
        elif p.method == "letter-readout":
            print("  method: letter readout (decision model; its own prompt, option-letter logits, temperature)")
        elif p.method == "contrastive":
            print(f"  method: contrastive (projection heads on the frozen backbone, in {p.heads})")
        else:
            print("  formulations: " + ", ".join(f"{f.name}@layer{f.layer}" for f in p.formulations)
                  + f", tau={p.tau}, center={p.center}")
        if p.notes:
            print(f"  note: {p.notes}")
    print("\nAdd a model: jul models add <name> --repo <hf repo> [--backend mlx|torch|onnx]")


def _with_cross(preset, cross):
    """The preset with a cross model (a repo or directory, or a preset `cross` entry) for the question
    types its cross.json declares (jul/cross.py)."""
    import dataclasses

    import json

    from jul.cross import SPEC_FILE, CrossSpec, LoraSpec, local_dir
    entry = cross if isinstance(cross, dict) else {"repo": cross}
    directory = local_dir(entry["repo"], entry.get("subfolder"))
    where = entry["repo"] + (f"/{entry['subfolder']}" if entry.get("subfolder") else "")
    if json.loads((directory / SPEC_FILE).read_text()).get("method") == "lora":
        spec = LoraSpec.load(directory)
        print(f"  cross model {where}: LoRA adapters on {spec.base}, read with the preset's own model; "
              f"reads {', '.join(spec.types)} ({spec.max_length} tokens)")
    else:
        spec = CrossSpec.load(directory)
        print(f"  cross model {where}: reads {', '.join(spec.types)} (layer {spec.layer}, {spec.max_length} tokens)")
    return dataclasses.replace(preset, cross=entry)


def _add_contrastive(a):
    """Projection heads on a frozen backbone: trained here on any backbone (--train-heads), converted from
    a published checkpoint (CLM-8B), or a directory that already holds them. Written to
    ~/.jul/heads/<name>; training and conversion need torch, inference does not."""
    from pathlib import Path as _P

    from jul.backbone import resolve_backend
    from jul.contrastive import SPEC_FILE, convert
    from jul.home import JUL_HOME
    from jul.presets import contrastive_preset, save_preset
    backend = resolve_backend(a.backend)
    if a.train_heads:
        from jul.backbone import Backbone
        from jul.contrastive import default_reading, read_rows, rows_from_labeled, train_heads
        if a.questions:
            questions = load_questions(a.questions)
            rows = rows_from_labeled(questions, read_labeled(a.train_heads, questions))
        else:
            rows = read_rows(a.train_heads)
        heads = JUL_HOME / "heads" / a.name
        print(f"{a.name}: training contrastive heads on {a.repo} ({backend}) from {len(rows)} rows into {heads}")
        bb = Backbone(a.repo, backend)
        report = train_heads(bb, rows, heads, {backend: a.repo}, default_reading(bb), epochs=a.epochs)
        print(f"  held-out accuracy {report['heads_accuracy']:.3f} against {report['cosine_accuracy']:.3f} for "
              f"the plain cosine of the same embeddings ({report['held_out']} rows)")
    elif (_P(a.repo) / SPEC_FILE).exists():
        heads = _P(a.repo)
    else:
        heads = JUL_HOME / "heads" / a.name
        print(f"{a.name}: converting the heads of {a.repo} into {heads}")
        convert(a.repo, heads)
    try:
        preset = contrastive_preset(a.name, heads, backend)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    path = save_preset(preset)
    print(f"{a.name}: contrastive heads on a frozen backbone, backbone {preset.repos[backend]} on {backend}"
          f" -> {path}")
    print(f"  {preset.notes}")
    print(f"\nUse it: jul ask ... --model {a.name} --backend {backend}"
          f"   (the backbone downloads on first use)")


def cmd_models_add(a):
    from jul.calibrate import CalibrationError, calibrate
    if not a.name:
        raise SystemExit("models add needs a name")
    if a.cross and not a.repo:
        # attach a cross model to a preset already fitted: nothing to refit
        from jul.backbone import resolve_backend
        from jul.presets import resolve, save_preset
        backend = resolve_backend(a.backend)
        try:
            preset = resolve(a.name, backend)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from exc
        import dataclasses
        # a built-in preset names no backend: the copy saved with its cross model is for this one
        path = save_preset(dataclasses.replace(_with_cross(preset, a.cross), backend=preset.backend or backend))
        print(f"{a.name} on {backend}: cross model attached -> {path}")
        return
    if a.train_heads:
        if not a.repo:
            raise SystemExit("--train-heads needs --repo, the backbone the heads are trained on")
        return _add_contrastive(a)
    from jul.contrastive import is_heads_source
    if a.repo and is_heads_source(a.repo):
        return _add_contrastive(a)
    from jul.letter_models import spec_from_repo
    letters = spec_from_repo(a.repo) if a.repo else None
    if letters is not None:
        from jul.backbone import resolve_backend
        from jul.presets import letters_preset, save_preset
        backend = resolve_backend(a.backend)
        if backend != "torch":
            raise SystemExit(f"error: a letter-readout model is read with --backend torch for now, not {backend} "
                             "(its parity with the model's runtime is measured on torch only)")
        path = save_preset(letters_preset(a.name, a.repo, backend, letters))
        temps = ", ".join(f"{k} {v:.3g}" if k != "default" else f"{v:.3g}" for k, v in letters.temperature.items())
        print(f"{a.name}: letter-readout decision model on {backend}, read with its own {letters.format} prompt "
              f"(letters {letters.letters[:letters.max_options]}, temperature {temps}; from {letters.source}). "
              f"Nothing fitted -> {path}")
        print(f"\nUse it: jul ask ... --model {a.name} --backend {backend}")
        return
    from jul.decision import spec_source
    source = spec_source(a.repo) if a.repo else None
    if source:
        import dataclasses

        from jul.backbone import resolve_backend
        from jul.calibrate import (DATA_HOME, ROUTE_COUNTS, ROUTE_SPEEDUP, fit_route_above, load_data)
        from jul.decision import DecisionSpec
        from jul.presets import pointer_preset, routing_from, save_preset
        backend = resolve_backend(a.backend)
        preset = pointer_preset(a.name, source, backend)
        if a.no_routing:
            path = save_preset(preset)
            print(f"{a.name}: decision model on {backend}, read with its decision.json "
                  f"(nothing fitted, --no-routing) -> {path}")
            print(f"\nUse it: jul ask ... --model {a.name} --backend {backend}")
            return
        # The pointer head reads every option on every call, so a long question is cheaper read as
        # vectors: fit that fallback on these very weights. `calibrate` saves a vector preset under the
        # same name, so the pointer preset is written last and wins.
        print(f"{a.name}: decision model; fitting the vector reading its long questions fall back to")
        try:
            fitted = calibrate(a.name, repo=source, backend=backend, data=a.data,
                               n_dev=a.n_dev, n_generic=a.n_generic)
        except CalibrationError as exc:
            raise SystemExit(f"error: {exc} (pass --no-routing to add it without the fallback)") from exc
        spec = DecisionSpec.load(preset.repos[backend]) if Path(preset.repos[backend]).is_dir() else None
        above = a.route_above if a.route_above is not None else (spec and spec.route_above)
        preset = dataclasses.replace(preset, routing=routing_from(fitted, above or 0))
        path = save_preset(preset)
        c, route = fitted.calibration, None
        if above is None:
            # The pointer head is worth a few points at every option count, so the threshold is not where
            # accuracy stops suffering: it is where the speed is worth them. Measured, never guessed.
            print(f"{a.name}: measuring where reading as vectors becomes {ROUTE_SPEEDUP:.0f}x faster")
            dev, _ = load_data(Path(a.data) if a.data else DATA_HOME, a.n_dev, 0)
            route = fit_route_above(a.name, dev)
            above = route["above_options"]
            preset = dataclasses.replace(preset, routing={**preset.routing, "above_options": above,
                                                          "measured": route})
            path = save_preset(preset)
        print(f"\n{a.name}: decision model on {backend}, pointer head from its decision.json; above "
              f"{above} options it reads as vectors: layers "
              + ", ".join(f"{f.name}@{f.layer}" for f in fitted.formulations)
              + f", center {fitted.center}, tau {fitted.tau}")
        print(f"  the fallback alone scores {c['dev_accuracy']:.3f} ± {c['dev_accuracy_stderr']:.3f} "
              f"on the dev sets (n={c['n_dev']})")
        if route and above:
            print(f"  above {above} options it is {route['speedup']:.1f}x faster and "
                  f"{route['accuracy_cost'] * 100:+.1f} points less accurate, on {route['set']}")
        elif route:
            print(f"  routing off: never {ROUTE_SPEEDUP:.0f}x faster within {ROUTE_COUNTS[-1]} options "
                  f"on {route['set']}")
        print(f"  -> {path}")
        print(f"\nUse it: jul ask ... --model {a.name} --backend {backend}"
              f"   (route_above=0 on a call buys those points back)")
        return
    try:
        preset = calibrate(a.name, repo=a.repo, backend=a.backend, data=a.data,
                           n_dev=a.n_dev, n_generic=a.n_generic)
    except (CalibrationError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    except RuntimeError as exc:   # an embeddings API down or refusing (jul.backends.api.EmbeddingsError)
        if type(exc).__name__ != "EmbeddingsError":
            raise
        raise SystemExit(f"error: {exc}") from exc
    from jul import cross as cross_models
    found = a.cross or (None if a.no_cross or not a.repo else cross_models.find(a.repo))
    if found:
        from jul.presets import save_preset
        preset = _with_cross(preset, found)
        save_preset(preset)
    c = preset.calibration
    print(f"\n{preset.name} on {c['backend']}: layers "
          + ", ".join(f"{f.name}@{f.layer}" for f in preset.formulations)
          + f", center {preset.center}, tau {preset.tau}")
    print(f"  dev accuracy {c['dev_accuracy']:.3f} ± {c['dev_accuracy_stderr']:.3f} (n={c['n_dev']}), "
          f"{c['dev_accuracy_task_center']:.3f} with a task center; ECE {c['dev_ece']:.3f}; "
          f"{c['latency_ms_p50']:.0f} ms per decision")
    print("  by set: " + ", ".join(f"{k} {v:.2f}" for k, v in c["dev_accuracy_by_set"].items()))
    print("  by center: " + ", ".join(f"{k} {v:.3f}" for k, v in c["dev_accuracy_by_center"].items()))
    print(f"\nUse it: jul ask ... --model {preset.name} --backend {c['backend']}")


def cmd_bench(a):
    import os

    from jul_cli.bench import run
    if a.on_long:          # read by every local client the bench builds (TypeSafeClient(on_long=) default)
        os.environ["JUL_ON_LONG"] = a.on_long
    run(a)


def cmd_setup(a):
    from jul_cli.setup import run
    run(a.model, a.backend, install=not a.no_install, skip_check=a.no_check)


def cmd_lab(a):
    """The prototype's research commands (feature extraction, head training, benchmarks)."""
    from jul.lab.cli import main as lab_main
    lab_main(a.rest)


def cmd_serve(a):
    """Serve the client over the Jev (System One) HTTP protocol, for non-Python callers."""
    from jul_cli.serve import serve
    serve(model=a.model, backend=a.backend, host=a.host, port=a.port,
          warmup=not a.no_warmup, api_key=a.api_key, escalate_to=a.escalate_to,
          escalate_model=a.escalate_model, escalate_key_env=a.escalate_key_env,
          min_confidence=a.min_confidence)


# --- parser -------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jul", description="Just use Less - local typed decisions.")
    model_kw = dict(default=None, help=f"preset: {', '.join(PRESETS)} (aliases: {', '.join(ALIASES)})")
    backend_kw = dict(choices=list(BACKENDS), default=None,
                      help="default: $JUL_BACKEND, else mlx on Apple Silicon, else torch")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="install the backend, download the model, check a first answer")
    s.add_argument("--model", **model_kw)
    s.add_argument("--backend", **backend_kw)
    s.add_argument("--no-install", action="store_true", help="do not pip install a missing backend")
    s.add_argument("--no-check", action="store_true", help="skip the final test decision")
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("ask", help="one typed question, answered now")
    s.add_argument("kind", choices=list(TYPES))
    s.add_argument("instructions")
    s.add_argument("-o", "--option", action="append",
                   help="choice: key[:description]; noul: true:meaning / false:meaning; "
                        "score: a level description, lowest first")
    s.add_argument("--state", action="append", required=True, help="the text to judge (repeatable)")
    s.add_argument("--model", **model_kw)
    s.add_argument("--backend", **backend_kw)
    s.add_argument("--context", help="name of a saved context")
    s.add_argument("--method", choices=["vector", "letters", "cross"])
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("run", help="answer a question file over a JSONL input")
    s.add_argument("questions")
    s.add_argument("--input", required=True)
    s.add_argument("--output")
    s.add_argument("--model", **model_kw)
    s.add_argument("--backend", **backend_kw)
    s.add_argument("--context")
    s.add_argument("--method", choices=["vector", "letters", "cross"])
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("context", help="create, list, show or delete a saved context")
    s.add_argument("action", choices=["create", "list", "show", "delete"])
    s.add_argument("name", nargs="?")
    s.add_argument("--description")
    s.add_argument("--examples", help=".txt (one per line) or .jsonl with a `text` field")
    s.add_argument("--use-description", action="store_true",
                   help="also put the description in the prompts; measured harmful on average, "
                        "so it is off unless you ask and measure on your own data")
    s.add_argument("--lazy", action="store_true", help="do not compute the centers now")
    s.add_argument("--model", **model_kw)
    s.add_argument("--backend", **backend_kw)
    s.set_defaults(fn=cmd_context)

    s = sub.add_parser("autotune", help="train a per-task head from labeled examples")
    s.add_argument("context", help="name of the context the head is stored in")
    s.add_argument("--questions", required=True)
    s.add_argument("--labeled", required=True, help="JSONL: {state, answers: {question: answer}}")
    s.add_argument("--features", choices=["vector", "lexical", "hybrid"], default="vector",
                   help="what the head reads: the model's vectors (default), TF-IDF of the text, or both")
    s.add_argument("--model", **model_kw)
    s.add_argument("--backend", **backend_kw)
    s.set_defaults(fn=cmd_autotune)

    s = sub.add_parser("pack", help="pack questions (and a context's heads) into a bundle to deploy; trains nothing")
    s.add_argument("out", help="directory to write")
    s.add_argument("--questions", required=True)
    s.add_argument("--context", help="context whose heads and calibration the bundle carries")
    s.add_argument("--model", **model_kw)
    s.add_argument("--backend", **backend_kw)
    s.add_argument("--reading", choices=["auto", "vector"], default="auto",
                   help="auto: the preset's cross model answers the types it declares (default); vector: every "
                        "question as vectors, so the bundle needs the vector model only")
    s.set_defaults(fn=cmd_pack)

    s = sub.add_parser("synth", help="write a synthetic labeled dataset (for a later autotune)")
    s.add_argument("questions")
    s.add_argument("--seeds", help="JSONL of real examples: {state|text, answers?}; answers optional")
    s.add_argument("--output", required=True, help="JSONL in the format of autotune --labeled")
    s.add_argument("--per-option", type=int, default=30, help="texts per option (default 30)")
    s.add_argument("--writer", required=True, help="a generative mlx-lm repo, e.g. an instruct model of 7B or more")
    s.add_argument("--batch", type=int, default=8, help="texts asked per call (default 8)")
    s.add_argument("--temperature", type=float, default=0.9)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("models", help="list the presets, or fit one for a new model (add)")
    s.add_argument("action", nargs="?", choices=["list", "add"], default="list")
    s.add_argument("name", nargs="?", help="add: the preset name")
    s.add_argument("--repo", help="add: Hugging Face repo (default: the known repo for this name), or a local "
                                  "directory; one holding a decision.json is added as a decision model")
    s.add_argument("--backend", **backend_kw)
    s.add_argument("--data", type=Path, help="add: calibration data dir (default: fetched into "
                                               "~/.jul/calibration-data)")
    s.add_argument("--n-dev", type=int, default=50, help="add: examples per dev set (default 50)")
    s.add_argument("--n-generic", type=int, default=200, help="add: generic texts (default 200)")
    s.add_argument("--no-routing", action="store_true",
                   help="add, decision model: skip fitting the vector reading long questions fall back to")
    s.add_argument("--route-above", type=int,
                   help="add, decision model: option count above which it falls back to vectors; "
                        "0 disables the routing. Default: measured (or its decision.json, if it says)")
    s.add_argument("--cross", help="add: a cross model (directory or repo with a cross.json) that answers the "
                                   "question types it declares; without --repo, attached to the fitted preset. "
                                   "A repo carrying one in cross/ gets it without this flag")
    s.add_argument("--no-cross", action="store_true", help="add: ignore the cross model the repo carries")
    s.add_argument("--train-heads", type=Path, metavar="ROWS",
                   help="add: train contrastive heads on the --repo backbone (kept frozen) from labeled rows, "
                        "JSONL {state, type, question, options, answer} or parquet; with --questions, "
                        "autotune's format {state, answers}. Needs torch for the training")
    s.add_argument("--questions", type=Path, help="add --train-heads: the question file the rows answer")
    s.add_argument("--epochs", type=int, default=40, help="add --train-heads: maximum epochs (default 40)")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("serve", help="serve the Jev HTTP protocol (POST /v1/systemone) locally")
    s.add_argument("--model", **model_kw)
    s.add_argument("--backend", **backend_kw)
    s.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1, local only)")
    s.add_argument("--port", type=int, default=8577, help="port (default 8577)")
    s.add_argument("--api-key", default=None,
                   help="require this key, as Authorization: Bearer or x-api-key (else $JUL_API_KEY). "
                        "Recommended when binding beyond 127.0.0.1.")
    s.add_argument("--no-warmup", action="store_true", help="do not load the model before serving")
    s.add_argument("--escalate-to", default=None, metavar="TARGET",
                   help="send answers below --min-confidence elsewhere: typesafe (key in TYPESAFE_API_KEY), "
                        "ollama[:model], cloudflare[:clef|clef-flash] (CLOUDFLARE_ACCOUNT_ID, "
                        "CLOUDFLARE_API_TOKEN), or the URL of any /v1/systemone server. The state leaves the machine")
    s.add_argument("--escalate-model", default=None, help="model asked there (default: the provider's)")
    s.add_argument("--escalate-key-env", default=None, metavar="VAR",
                   help="the environment variable holding that server's key, for a URL target")
    s.add_argument("--min-confidence", type=float, default=0.8,
                   help="the bar below which an answer escalates (default 0.8; a Noul counts max(p, 1-p))")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("bench", help="which model for your questions: accuracy, latency, autotune, on your data")
    s.add_argument("test", help="JSONL or CSV, one row per answer: question, state, options, answer (type optional)")
    s.add_argument("--train", help="same format: rows to autotune on; without it, zero-shot only")
    s.add_argument("--models", help="presets or aliases, comma-separated, each optionally @backend "
                                     "(e.g. jul-decision-e5-small@onnx,minicpm5-2b@torch); default: the default model. "
                                     "Also remote System One servers: typesafe (Jev), ollama[:model], "
                                     "cloudflare:clef|clef-flash, or URL#model (Kev, a hosted Laya...): the test rows "
                                     "are sent there")
    s.add_argument("--remote-key-env", default=None, metavar="VAR",
                   help="the environment variable holding the key of the URL targets in --models "
                        "(providers use their own: TYPESAFE_API_KEY, CLOUDFLARE_API_TOKEN)")
    s.add_argument("--escalate-to", default=None, metavar="TARGET",
                   help="also measure each local model as a cascade: itself first, then this remote server "
                        "(same targets as --models) for the answers below --min-confidence; reports the accuracy "
                        "and the share of rows sent there")
    s.add_argument("--min-confidence", type=float, default=0.8,
                   help="the cascade's bar (default 0.8, as jul serve; a Noul counts max(p, 1-p))")
    s.add_argument("--backend", **backend_kw)
    s.add_argument("--method", metavar="M[,M]|auto",
                   help="zero-shot readings to compare: vector, letters, cross, or auto for every one the model "
                        "takes (default: the model's own reading)")
    s.add_argument("--features", metavar="F[,F]|auto",
                   help="with --train: what the autotune heads read, vector, lexical, hybrid, or auto "
                        "(default: vector)")
    s.add_argument("--output", "-O", help="also write the full results as JSON to this file")
    s.add_argument("--json", action="store_true", help="print the JSON results instead of the tables")
    s.add_argument("--near", type=float, default=0.8,
                   help="similarity (character 5-grams) from which a test row counts as a near duplicate of "
                        "a train row (default 0.8)")
    s.add_argument("--drop-overlap", action="store_true", help="drop test rows found in train (exact or near)")
    s.add_argument("--allow-overlap", action="store_true", help="run even with exact duplicates train/test")
    s.add_argument("--quiet", "-q", action="store_true", help="no progress on stderr")
    s.add_argument("--on-long", choices=["cut", "error"],
                   help="a state over a reading's limit: cut (default, answer on its beginning) or error (refuse)")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("lab", help="research commands")
    s.add_argument("rest", nargs=argparse.REMAINDER,
                   help="passed through: extract, evaluate, bench, summary, ask, decide")
    s.set_defaults(fn=cmd_lab)
    return p


def main(argv=None) -> None:
    from jul import telemetry
    telemetry.ENTRYPOINT = "cli"
    a = build_parser().parse_args(argv)
    if a.command == "context" and a.action != "list" and not a.name:
        raise SystemExit(f"context {a.action} needs a name")
    if a.command == "bench":
        from jul_cli.setup import require_setup
        from jul_cli.bench import is_remote
        for model in (a.models or "").split(",") if a.models else [None]:
            if model and is_remote(model.strip()):
                continue                       # a server: nothing to set up here
            name, _, backend = (model or "").strip().partition("@")
            require_setup(name or None, backend or a.backend)
    if a.command in {"ask", "run", "autotune", "pack"} or (
            a.command == "context" and a.action == "create" and a.examples and not a.lazy):
        from jul_cli.setup import require_setup
        require_setup(a.model, a.backend)
    a.fn(a)


if __name__ == "__main__":
    main()
