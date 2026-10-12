"""Parity of jul's d3 reading (Decision 3.0, jul/letter_models.py format "d3") with the model's own runtime.

The runtime is the model repo's `d3_runtime.D3` (trust_remote_code-style code, run from the pinned snapshot),
here on CPU in float32; jul reads the same snapshot with its letters preset (no remote code). The script
prints both probability sets per question, their largest gap and whether the top option agrees, and the
prompt length of each request on both sides (the runtime's `usage.input_tokens`: the prompts are rebuilt
token for token, so the counts must be equal). It fails above --tolerance.

jul pins transformers to its own range and the runtime to its own (5.17.0): run them in two environments
and pass the probabilities through a file.

    # the runtime's environment: pip install "transformers==5.17.0" torch accelerate einops safetensors
    python scripts/d3_parity.py --model vllm-sr/d3-edge --revision c322a0dd... --save-reference ref.json
    # jul's environment (pip install -e '.[torch,mlx]')
    python scripts/d3_parity.py --reference ref.json --backend torch            # float32 on CPU
    python scripts/d3_parity.py --reference ref.json --backend mlx              # bf16 weights as stored
    python scripts/d3_parity.py --reference ref.json --backend mlx --mlx-dtype float32

Requests: the model README's text examples, one choice of 30 options (codes past Z), plus any
`--items requests.jsonl` (one `{"id", "state", "questions": {name: question}}` per line, System One
questions). Items are read from the file each time and only their ids and numbers are written out.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

ROUTE = {"type": "choice", "instructions": "Which team should handle this request?",
         "criteria": {"returns": "Refunds, replacements and damaged deliveries",
                      "billing": "Payments, invoices and charges", "technical": "Product setup and faults"}}
#: The README's text requests (the image and video ones keep their text questions, without the media).
README = [
    ("readme-text", "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement "
                    "today.",
     {"route": ROUTE, "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
      "urgency": {"type": "score", "instructions": "How urgent is this request?",
                  "criteria": ["Routine", "Soon", "Today"]}}),
    ("readme-image-text", "The customer says the blender arrived cracked and attached the receipt.",
     {"route": ROUTE, "on_receipt": {"type": "noul", "instructions": "Does the receipt list the blender?"},
      "payment": {"type": "choice", "instructions": "How was the order paid?",
                  "criteria": {"card": None, "cash": None, "gift card": None}}}),
    ("readme-video-text", "A short clip from a test camera.",
     {"direction": {"type": "choice", "instructions": "Which way does the square move?",
                    "criteria": {"right": "From left to right", "left": "From right to left",
                                 "still": "It does not move"}},
      "color_change": {"type": "noul", "instructions": "Does the square change color?"}}),
    ("readme-engine", "Write a Python function that merges two sorted lists.",
     {"kind": {"type": "choice", "instructions": "What kind of work is this?",
               "criteria": {"code": "Writing or fixing code", "math": "Mathematics", "chat": "Anything else"}},
      "reasoning": {"type": "noul", "instructions": "Does answering this need multi-step reasoning?"}}),
    ("thirty-options", {"ticket": 4411, "text": "My parcel went to my neighbour at No. 71, I live at No. 17."},
     {"topic": {"type": "choice", "instructions": "Which queue should take this ticket?",
                "criteria": {f"queue_{i:02d}": None for i in range(27)} | {
                    "misdelivery": "The parcel was delivered to the wrong address",
                    "refund": "The customer wants their money back", "other": None}},
      "neighbour": {"type": "noul", "instructions": "Was the parcel left with someone else?",
                    "criteria": {"true": "It was left with someone else", "false": ""}}}),
]


def load_items(path: str | None) -> list[tuple[str, object, dict]]:
    cases = list(README)
    if path:
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cases.append((r["id"], r["state"], r["questions"]))
    return cases


def keys(q: dict) -> list[str]:
    if q["type"] == "noul":
        return ["true", "false"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return list(q["criteria"])


# --- the model's runtime ------------------------------------------------------------------------

def run_reference(repo: str, revision: str | None, cases, device: str,
                  dtype: str = "float32") -> tuple[str, list[dict]]:
    import torch
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo, revision=revision)       # the whole package: its manifest is checked
    sys.path.insert(0, root)
    from d3_runtime import D3
    torch.set_grad_enabled(False)
    model = D3.from_pretrained(root, device=device, batch_size=1)    # one prompt per pass, as jul
    if dtype == "float32":
        model.backbone.float()                                       # loaded in bf16 (its default)
    out = []
    for cid, state, qs in cases:
        r = model.system_one(state=state, questions=qs)
        got = {}
        for n, q in qs.items():
            a = r["answers"][n]
            if "error" in a:
                got[n] = {"error": a["error"], "message": a.get("message")}
            elif q["type"] == "noul":
                got[n] = {"true": a["noul"], "false": 1 - a["noul"]}
            else:
                got[n] = {k: float(v) for k, v in a["probabilities"].items()}
        out.append({"id": cid, "input_tokens": r["usage"]["input_tokens"], "answers": got})
        print(f"reference {cid}: {r['usage']['input_tokens']} tokens", flush=True)
    del model
    gc.collect()
    return Path(root).name, out


# --- jul ----------------------------------------------------------------------------------------

def jul_question(q: dict):
    from jul.types import Choice, Noul, Score
    if q["type"] == "choice":
        # a None description is no description: jul's Choice says it with ""
        return Choice(q.get("instructions") or "", {k: v or "" for k, v in q["criteria"].items()})
    if q["type"] == "noul":
        return Noul(q.get("instructions") or "", q.get("criteria"))
    return Score(q.get("instructions") or "", q["criteria"])


def run_jul(root: str, cases, backend: str, mlx_dtype: str | None) -> list[dict]:
    import numpy as np
    from jul import TypeSafeClient, letter_models
    from jul.letter_models import spec_from_repo
    from jul.presets import letters_preset, save_preset
    spec = spec_from_repo(root)
    if backend == "mlx":    # what this script measures: let the reading run before its format is listed
        letter_models.MLX_MEASURED.add(spec.format)
    if backend == "mlx" and mlx_dtype == "float32":
        # float32 from the file on: mlx-lm adds 1 to Qwen3.5's norm weights while loading, in the stored dtype
        # (bf16), so a cast after loading would keep that rounding
        import mlx.core as mx
        load = mx.load

        def load_float32(*args, **kwargs):
            arrays = load(*args, **kwargs)
            return {k: v.astype(mx.float32) if mx.issubdtype(v.dtype, mx.floating) else v
                    for k, v in arrays.items()} if isinstance(arrays, dict) else arrays
        mx.load = load_float32
    save_preset(letters_preset("parity-d3", root, backend, spec))
    client = TypeSafeClient(model="parity-d3", backend=backend)
    reader = client._engine_for(None).reader
    # the client's answers round probabilities to 4 decimals: the reader's own (unrounded) are compared
    seen, read = [], reader.logits

    def recording(*args, **kwargs):
        logits, tokens = read(*args, **kwargs)
        seen.append(logits)
        return logits, tokens
    reader.logits = recording
    out = []
    for cid, state, qs in cases:
        got, tokens = {}, 0
        for n, q in qs.items():
            seen.clear()
            try:
                r = client.system_one(state, {n: jul_question(q)})
            except ValueError as e:
                got[n] = {"error": str(e)}
                continue
            tokens += r.usage.input_tokens
            (z,) = seen[0]
            p = np.exp(z - z.max())
            p = p / p.sum()
            ks = keys(q) if q["type"] != "noul" else ["true", "false"]     # jul's option order
            got[n] = dict(zip(ks, map(float, p)))
            a = r.answers[n]                                                # the client's answer agrees
            assert abs((a.noul if q["type"] == "noul" else a.probabilities[ks[0]]) - got[n][ks[0]]) < 1e-4
        out.append({"id": cid, "input_tokens": tokens, "answers": got})
    return out


def snapshot(repo: str, revision: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="vllm-sr/d3-edge")
    ap.add_argument("--revision", help="the model's commit (default: main)")
    ap.add_argument("--items", help="more requests, one JSON per line (kept out of the outputs)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default="torch", choices=("torch", "mlx"), help="jul's backend (default: torch)")
    ap.add_argument("--dtype", default="float32", help="jul's torch dtype (default float32)")
    ap.add_argument("--mlx-dtype", choices=("stored", "float32"), default="stored",
                    help="mlx: the weights as stored (bf16) or cast to float32")
    ap.add_argument("--tolerance", type=float, default=5e-3, help="largest allowed probability gap")
    ap.add_argument("--reference-dtype", choices=("float32", "bfloat16"), default="float32",
                    help="the runtime's dtype (default float32; bfloat16 is its own default)")
    ap.add_argument("--save-reference", metavar="JSON", help="run the runtime only and save its probabilities")
    ap.add_argument("--reference", metavar="JSON", help="compare jul with probabilities saved by --save-reference")
    ap.add_argument("--out", help="the comparison, per question (ids and numbers only)")
    a = ap.parse_args()
    cases = load_items(a.items)

    if a.reference:
        saved = json.load(open(a.reference))
        repo, revision, ref = saved["repo"], saved["commit"], saved["requests"]
    else:
        repo, revision = a.model, a.revision
        revision, ref = run_reference(repo, revision, cases, a.device, a.reference_dtype)
        if a.save_reference:
            with open(a.save_reference, "w") as f:
                json.dump({"repo": repo, "commit": revision, "device": a.device, "dtype": a.reference_dtype,
                           "requests": ref}, f, indent=1)
            print(f"reference saved: {repo}@{revision}, {len(ref)} requests")
            return 0
    if [r["id"] for r in ref] != [c[0] for c in cases]:
        raise SystemExit("the reference was run on other requests: pass the same --items")

    os.environ.setdefault("JUL_HOME", tempfile.mkdtemp(prefix="jul-parity-"))
    os.environ.setdefault("JUL_DEVICE", a.device)
    os.environ.setdefault("JUL_DTYPE", a.dtype)
    mine = run_jul(snapshot(repo, revision), cases, a.backend, a.mlx_dtype)

    worst, agree, total, token_mismatch, rows = 0.0, 0, 0, [], []
    for (cid, _, qs), r, m in zip(cases, ref, mine):
        if r["input_tokens"] != m["input_tokens"] and all("error" not in v for v in r["answers"].values()):
            token_mismatch.append((cid, r["input_tokens"], m["input_tokens"]))
        for n, q in qs.items():
            rr, mm = r["answers"][n], m["answers"][n]
            if "error" in rr or "error" in mm:
                same = ("error" in rr) == ("error" in mm)
                print(f"{cid:40} {n:12} both refuse: {same}  ({rr.get('error') or mm.get('error')})")
                rows.append({"id": cid, "question": n, "refused": {"runtime": "error" in rr, "jul": "error" in mm}})
                if not same:
                    worst = float("inf")
                continue
            ks = keys(q)
            gap = max(abs(float(rr[k]) - float(mm[k])) for k in ks)
            top_r, top_m = max(ks, key=lambda k: rr[k]), max(ks, key=lambda k: mm[k])
            worst, total, agree = max(worst, gap), total + 1, agree + (top_r == top_m)
            rows.append({"id": cid, "question": n, "type": q["type"], "options": len(ks), "gap": gap,
                         "argmax_agrees": top_r == top_m, "runtime_top": round(float(rr[top_r]), 4),
                         "jul_top": round(float(mm[top_r]), 4)})
            print(f"{cid:40} {n:12} {q['type']:6} {len(ks):3} opts  runtime {rr[top_r]:.4f}  jul {mm[top_r]:.4f}"
                  f"  gap {gap:.2e}{'' if top_r == top_m else '  ARGMAX DIFFERS'}", flush=True)
    summary = {"repo": repo, "commit": revision, "backend": a.backend,
               "dtype": a.dtype if a.backend == "torch" else a.mlx_dtype, "questions": total,
               "largest_gap": worst, "argmax_agreement": f"{agree}/{total}", "token_mismatches": token_mismatch}
    if a.out:
        with open(a.out, "w") as f:
            json.dump({**summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary))
    print(f"largest gap {worst:.2e} (tolerance {a.tolerance}); argmax {agree}/{total}; "
          f"prompt lengths differ on {len(token_mismatch)} requests")
    return 0 if worst <= a.tolerance and not token_mismatch else 1


if __name__ == "__main__":
    sys.exit(main())
