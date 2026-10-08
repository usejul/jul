"""Parity of jul's reading of a Strands Decider checkpoint with Strands Decider's own runtime.

    pip install "strands-decider @ git+https://github.com/Vivek0712/strands-decider@9800d14"
    python scripts/convert_strands_decider.py strands-decider-4B-hobson-v22.tar converted/
    python scripts/strands_parity.py ref  <checkpoint dir> --out ref.json   [--items items.jsonl] [--device cuda]
    python scripts/strands_parity.py jul  converted/       --out jul.json   [--items items.jsonl] [--backend torch]
    python scripts/strands_parity.py compare ref.json jul.json [--tolerance 0.02]

`ref` runs the archive as `strands-decider serve` does (base model + LoRA, unmerged, bfloat16 on cuda);
`jul` reads the converted directory (LoRA merged). Each side runs in its own process, so only one model is
in memory at a time. Requests are Jev requests ({"state", "questions"}), one per line in --items; the README
examples always run. `compare` fails when an argmax differs or a probability moves by more than --tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

#: The examples of Strands Decider's README (the server request, then the three questions of one `ask`).
README = [
    {"state": "Help! My payouts have been failing for 3 days!",
     "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}},
    {"state": "Help! My payouts have been failing for 3 days! ",
     "questions": {"choice_0": {"type": "choice", "instructions": "Which team should handle this?",
                                "criteria": ["billing", "sales", "retail"]},
                   "noul_0": {"type": "noul", "instructions": "Does this convey urgency?"},
                   "score_0": {"type": "score", "instructions": "How frustrated is the writer?",
                               "criteria": ["calm", "frustrated", "depressed"]}}},
]


def requests(path: str | None) -> list[dict]:
    extra = [json.loads(line) for line in open(path)] if path else []
    return README + extra


def keys(q: dict) -> list[str]:
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    c = q["criteria"]
    return list(c) if isinstance(c, dict) else [str(k) for k in c]


def probabilities(answer: dict | object, q: dict) -> dict[str, float]:
    a = answer if isinstance(answer, dict) else answer.model_dump()
    if q["type"] == "noul":
        return {"false": 1 - float(a["noul"]), "true": float(a["noul"])}
    return {k: float(v) for k, v in a["probabilities"].items()}


def run_ref(checkpoint: str, reqs: list[dict], device: str) -> list[dict]:
    from strands_decider.infer import load_engine
    engine = load_engine(checkpoint, device=device)
    out = []
    for r in reqs:
        resp = engine.ask(r["state"], {n: _strands_question(q) for n, q in r["questions"].items()})
        out.append({n: probabilities(resp.answers[n], q) for n, q in r["questions"].items()})
    return out


def _strands_question(q: dict):
    """Its schema takes a choice's criteria as a mapping only: a list of names becomes names without descriptions."""
    from pydantic import TypeAdapter
    from strands_decider.schema import Question
    if q["type"] == "choice" and isinstance(q["criteria"], list):
        q = {**q, "criteria": dict.fromkeys(q["criteria"])}
    return TypeAdapter(Question).validate_python(q)


def run_jul(directory: str, reqs: list[dict], backend: str) -> list[dict]:
    os.environ.setdefault("JUL_HOME", tempfile.mkdtemp(prefix="jul-parity-"))
    from jul import TypeSafeClient
    from jul.presets import pointer_preset, save_preset
    from jul.types import Choice, Noul, NoulCriteria, Score

    def question(q):
        if q["type"] == "choice":
            c = q["criteria"]
            return Choice(q["instructions"], {k: v or "" for k, v in c.items()} if isinstance(c, dict) else c)
        if q["type"] == "noul":
            c = q.get("criteria") or {}
            criteria = NoulCriteria(true=c.get("true", ""), false=c.get("false", "")) if c else None
            return Noul(q["instructions"], criteria)
        return Score(q["instructions"], q["criteria"])

    save_preset(pointer_preset("parity-strands", directory, backend))
    client = TypeSafeClient(model="parity-strands", backend=backend)
    out = []
    for r in reqs:
        resp = client.system_one(state=r["state"], questions={n: question(q) for n, q in r["questions"].items()})
        got = {}
        for n, q in r["questions"].items():
            a = resp.answers[n]
            got[n] = {"false": 1 - a.noul, "true": a.noul} if q["type"] == "noul" else dict(a.probabilities)
        out.append(got)
    client.close()
    return out


def compare(ref: dict, mine: dict, tolerance: float) -> int:
    worst, flips, rows = 0.0, 0, 0
    for r, a, b in zip(ref["requests"], ref["answers"], mine["answers"]):
        for n, q in r["questions"].items():
            ks = keys(q)
            gap = max(abs(a[n][k] - b[n][k]) for k in ks)
            top_a, top_b = max(ks, key=lambda k: a[n][k]), max(ks, key=lambda k: b[n][k])
            worst, flips, rows = max(worst, gap), flips + (top_a != top_b), rows + 1
            flag = "  ARGMAX" if top_a != top_b else ""
            print(f"{q['type']:6} {len(ks):2} opts  ref {top_a}={a[n][top_a]:.4f}  jul {top_b}={b[n][top_b]:.4f}  "
                  f"max |dp| {gap:.2e}{flag}")
    print(f"{rows} questions, {flips} argmax differences, largest |dp| {worst:.2e} (tolerance {tolerance})")
    return 0 if flips == 0 and worst <= tolerance else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("side", choices=["ref", "jul", "compare"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--items")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda", help="ref: torch device of Strands Decider")
    ap.add_argument("--backend", default="torch", help="jul: backend")
    ap.add_argument("--tolerance", type=float, default=0.02)
    a = ap.parse_args()
    if a.side == "compare":
        return compare(*(json.load(open(p)) for p in a.paths[:2]), a.tolerance)
    reqs = requests(a.items)
    t = time.time()
    answers = run_ref(a.paths[0], reqs, a.device) if a.side == "ref" else run_jul(a.paths[0], reqs, a.backend)
    print(f"{a.side}: {len(reqs)} requests in {time.time() - t:.1f} s", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"side": a.side, "requests": reqs, "answers": answers}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
