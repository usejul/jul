"""Parity of jul's letter-readout models with each model's own runtime, on the examples of its README.

    pip install "jevk5 @ git+https://github.com/allebee/jevk5@v0.3.3" quyet \
                "open-spark-jev @ git+https://github.com/abhishek085/open-spark-jev"
    python scripts/letters_parity.py [--models jevk5,plumb,quyet,spark] [--device cpu] [--out parity.json]

For every model, jul reads the README's requests (`jul models add` on the same weights, torch backend)
and the model's runtime reads them too; the script prints both probability sets and their largest gap,
and fails above --tolerance. Both run in float32 on the same device, so the gap is numerical noise
(a different but equivalent forward: jul runs a prompt's shared beginning once as a cached prefix).
A JevK5 request with 20 options checks the knockout (groups of up to 16, then a final).

On a machine where a 4B model does not fit in float32 (a 16 GB Mac), run the runtimes elsewhere and compare:

    python scripts/letters_parity.py --device cuda --save-reference ref.json          # GPU, runtimes only
    python scripts/letters_parity.py --backend mlx --reference ref.json               # the Mac, jul on MLX
    python scripts/letters_parity.py --dtype bfloat16 --reference ref.json            # same, torch CPU bf16

The weights' commit is stored with the reference and must match the local cache.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile

STATE_BILLING = "I was billed twice for order #4411. Please refund the duplicate charge today."
TEAM = {"type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payments and refunds", "tech": "Bugs", "sales": "New purchases"}}
INTENTS = {"type": "choice", "instructions": "What is the customer's intent?",
           "criteria": [f"intent_{i}" for i in range(18)] + ["refund", "cancel"]}

#: The README examples, per model: (state, {name: question}).
CASES = {
    "jevk5": ("alibiserikbay/JevK5", [
        (STATE_BILLING, {"team": TEAM}),
        ("I was billed twice, please refund the duplicate.",
         {"refund": {"type": "noul", "instructions": "Does the customer ask for money back?"}}),
        (STATE_BILLING, {"intent": INTENTS}),
    ]),
    "plumb": ("crh225/plumb-4b", [
        ("Refunds need a receipt and a purchase within 30 days. The customer bought 12 days ago and has no receipt.",
         {"permitted": {"type": "noul", "instructions": "Is a refund permitted under the policy?"}}),
        ("Order #7120 shows delivered to No. 17; the customer lives at No. 71.",
         {"parcel": {"type": "choice", "instructions": "What happened to the parcel?",
                     "criteria": ["delivered", "misdelivered", "unknown"]}}),
        ("I was billed twice, please refund",
         {"refund": {"type": "noul", "instructions": "Asks for money back?"}}),
    ]),
    "quyet": ("chinhnc/Quyet-1.0-Medium", [
        ({"message": "Please close my card, I lost it yesterday."},
         {"intent": {"type": "choice", "instructions": "What does the customer want?",
                     "criteria": {"cancel": "close the card", "limit": "change the limit", "other": None}},
          "urgent": {"type": "noul", "instructions": "The request is urgent."},
          "mood": {"type": "score", "instructions": "How upset is the customer?",
                   "criteria": ["calm", "annoyed", "angry"]}}),
    ]),
    "spark": ("abhishek085/spark-s1-4b-v6", [
        ({"tool": "bash", "command": "kubectl get pods -n production"},
         {"approve": {"type": "choice", "instructions": "Approve this tool call?",
                      "criteria": {"allow": "run it", "ask": "ask the user first", "deny": "refuse"}}}),
        ({"queue": "billing", "message": "I was billed twice, please refund."},
         {"refund": {"type": "noul", "instructions": "The customer asks for money back."},
          "anger": {"type": "score", "instructions": "How angry is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}}),
    ]),
}


def keys(q: dict) -> list[str]:
    if q["type"] == "noul":
        return ["true", "false"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    c = q["criteria"]
    return list(c) if isinstance(c, dict) else [str(k) for k in c]


# --- the models' own runtimes: {name: {key: probability}} per request --------------------------

def ref_jevk5(repo, cases, device):
    import torch
    from jevk5 import JevK5
    model = JevK5(repo, device=device, dtype=torch.float32, graphs=False)
    out = []
    for state, qs in cases:
        out.append({n: model.probabilities(state, q)[0] for n, q in qs.items()})
    return out


def ref_quyet(repo, cases, device):
    import torch
    import quyet
    model = quyet.load(repo, device=device, dtype=torch.float32)
    out = []
    for state, qs in cases:
        (probs,) = model.raw_probabilities([{"state": state, "questions": qs}])
        out.append({n: dict(zip(keys(q), map(float, probs[n]))) for n, q in qs.items()})
    return out


def ref_spark(repo, cases, device):
    import torch
    from huggingface_hub import snapshot_download
    from open_spark_jev.model import MenuScorer
    from open_spark_jev.serve.gateway import JevQuestion, _from_jev
    from open_spark_jev.schema import State
    model = MenuScorer(snapshot_download(repo), dtype=torch.float32, device=device)
    out = []
    for state, qs in cases:
        sqs = [_from_jev(n, JevQuestion(**q)) for n, q in qs.items()]
        answers = model.decide(State(content=state), sqs)
        got = {}
        for (n, q), a in zip(qs.items(), answers):
            p = a.probs
            got[n] = ({"true": p[0], "false": p[1]} if q["type"] == "noul" else dict(zip(keys(q), p)))
        out.append(got)
    return out


REFS = {"jevk5": ref_jevk5, "plumb": ref_jevk5, "quyet": ref_quyet, "spark": ref_spark}


# --- jul ---------------------------------------------------------------------------------------

def jul_question(q: dict):
    from jul.types import Choice, Noul, Score
    if q["type"] == "choice":
        c = q["criteria"]
        return Choice(q["instructions"], {k: v or "" for k, v in c.items()} if isinstance(c, dict) else c)
    if q["type"] == "noul":
        return Noul(q["instructions"], q.get("criteria"))
    return Score(q["instructions"], q["criteria"])


def commit(repo: str) -> str:
    """The commit of the weights read: the cached snapshot (downloaded once if missing)."""
    from pathlib import Path
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo, local_files_only=True)).name
    except Exception:  # noqa: BLE001 - not cached yet
        return Path(snapshot_download(repo)).name


def run_jul(name, repo, cases, backend="torch"):
    from jul import TypeSafeClient
    from jul import letter_models
    from jul.letter_models import spec_from_repo
    from jul.presets import letters_preset, save_preset
    spec = spec_from_repo(repo)
    if backend == "mlx":  # what this script measures: let the reading run before its format is listed
        letter_models.MLX_MEASURED.add(spec.format)
    save_preset(letters_preset(f"parity-{name}", repo, backend, spec))
    client = TypeSafeClient(model=f"parity-{name}", backend=backend)
    out = []
    for state, qs in cases:
        r = client.system_one(state, {n: jul_question(q) for n, q in qs.items()})
        got = {}
        for n, q in qs.items():
            a = r.answers[n]
            got[n] = ({"true": a.noul, "false": 1 - a.noul} if q["type"] == "noul" else dict(a.probabilities))
        out.append(got)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=",".join(CASES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tolerance", type=float, default=0.01, help="largest allowed probability gap")
    ap.add_argument("--out")
    ap.add_argument("--backend", default="torch", choices=("torch", "mlx"), help="jul's backend (default: torch)")
    ap.add_argument("--dtype", default="float32",
                    help="jul's torch dtype (default float32); the runtimes always run in float32, and mlx reads "
                         "the weights as stored")
    ap.add_argument("--save-reference", metavar="JSON",
                    help="run the runtimes only and save their probabilities (and the weights' commit) there")
    ap.add_argument("--reference", metavar="JSON",
                    help="compare jul with the runtimes' probabilities saved by --save-reference, without running them")
    a = ap.parse_args()
    os.environ.setdefault("JUL_HOME", tempfile.mkdtemp(prefix="jul-parity-"))
    os.environ.setdefault("JUL_DEVICE", a.device)
    os.environ.setdefault("JUL_DTYPE", a.dtype)
    saved = json.load(open(a.reference)) if a.reference else {}
    references, report, worst = {}, {}, 0.0
    for name in a.models.split(","):
        repo, cases = CASES[name]
        if a.reference:
            if commit(repo) != saved[name]["commit"]:
                raise SystemExit(f"{name}: the reference was run on {repo}@{saved[name]['commit']}, "
                                 f"the cache holds {commit(repo)}")
            ref = saved[name]["probabilities"]
        else:
            ref = REFS[name](repo, cases, a.device)
            gc.collect()
        if a.save_reference:
            references[name] = {"repo": repo, "commit": commit(repo), "device": a.device, "dtype": a.dtype,
                                "probabilities": [{n: {k: float(v) for k, v in p.items()} for n, p in r.items()}
                                                  for r in ref]}
            with open(a.save_reference, "w") as f:
                json.dump(references, f, indent=1)
            print(f"{name}: reference saved", flush=True)
            continue
        mine = run_jul(name, repo, cases, a.backend)
        gc.collect()
        rows = []
        for (state, qs), r, m in zip(cases, ref, mine):
            for n, q in qs.items():
                gap = max(abs(float(r[n][k]) - float(m[n][k])) for k in keys(q))
                worst = max(worst, gap)
                rows.append({"question": n, "type": q["type"], "options": len(keys(q)), "gap": gap,
                             "runtime": {k: round(float(r[n][k]), 4) for k in keys(q)},
                             "jul": {k: round(float(m[n][k]), 4) for k in keys(q)}})
                top = max(keys(q), key=lambda k: r[n][k])
                print(f"{name:6} {n:10} {q['type']:6} {len(keys(q)):2} opts  runtime {top}={r[n][top]:.4f}  "
                      f"jul {top}={m[n][top]:.4f}  max gap {gap:.2e}", flush=True)
        report[name] = {"repo": repo, "questions": rows}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
    print(f"largest gap {worst:.2e} (tolerance {a.tolerance})")
    return 0 if worst <= a.tolerance else 1


if __name__ == "__main__":
    sys.exit(main())
