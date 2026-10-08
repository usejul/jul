"""Writes tests/fixtures/decision_strands_cases.json with Strands Decider's own code.

For each request: the prefix and branch token ids its engine feeds the model (`SystemOneEngine._fit`, which
tokenizes the state and the questions apart, with `render_state` and `render_question`), the question
position (the branch's last token, `<answer>`) and each option's position (`_option_token_index`). Plus a
tiny random `PointerHead` (LayerNorm, q, k) with its logits for given hidden states, divided by a
temperature per question type as `apply_temperature` does: jul's numpy head is checked against them.

Runs outside jul, in an environment holding torch, transformers 5 and pydantic, with the Strands Decider
source on the path (git checkout 9800d14) and the Qwen3.5-4B tokenizer in the Hugging Face cache:

    PYTHONPATH=<strands-decider>/src python scripts/strands_decider_fixture.py [--tokenizer <checkpoint dir>]
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from strands_decider.infer import EngineConfig, SystemOneEngine, _option_token_index
from strands_decider.modeling import PointerHead, apply_temperature
from strands_decider.prompting import render_question, render_state
from strands_decider.schema import SystemOneRequest
from transformers import AutoTokenizer

REPO, REVISION, MAX_LENGTH = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", 4096
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "decision_strands_cases.json"
TEMPERATURE = {"noul": 1.0143052922252125, "choice": 0.6246728003026183, "score": 0.5203771293158895}

REQUESTS = [
    {"state": "  Export declined by 6 percent to 16.4 million liters .\n",
     "questions": {"sent": {"type": "choice", "instructions": "Which single label best describes the input text?",
                            "criteria": {"negative": None, "neutral": None, "positive": None}}}},
    {"state": {"subject": "Charged twice", "body": "I see two charges for order #4411. Refund me NOW — été.",
               "items": [1, {"sku": "A-1"}]},
     "questions": {
         "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                  "criteria": {"billing": "Payments  and\nrefunds ", "shipping": "Delivery problems",
                               "access": None}},
         "angry": {"type": "noul", "instructions": "Is the customer angry?"},
         "prio": {"type": "score", "instructions": "How urgent is this ticket?",
                  "criteria": ["low", "normal, can wait a day", "high"]}}},
    {"state": "my package never arrived and nobody answers",
     "questions": {
         "late": {"type": "noul", "instructions": "Is this about a late delivery?",
                  "criteria": {"true": "the parcel is late or lost", "false": "anything else"}},
         "late_true_only": {"type": "noul", "instructions": "Is this about a late delivery?",
                            "criteria": {"true": "the parcel is late or lost"}}}},
    # longer than the window: the state is cut from the right to what the question leaves
    {"state": "The quick brown fox jumps over the lazy dog. " * 500,
     "questions": {"fox": {"type": "noul", "instructions": "Does a fox appear?"}}},
]


def encode(tok, request: dict) -> dict:
    req = SystemOneRequest.model_validate(request)
    rendered = [render_question(q) for q in req.questions.values()]
    engine = SimpleNamespace(tok=tok, cfg=EngineConfig(device="cpu"),
                             model=SimpleNamespace(config=SimpleNamespace(max_length=MAX_LENGTH)))
    s, q = SystemOneEngine._fit(engine, render_state(req.state), [rq.text for rq in rendered])
    branches = [{"ids": ids, "question": len(ids) - 1, "options": _option_token_index(offs, rq.option_spans, 0)}
                for ids, offs, rq in zip(q, engine._last_offsets, rendered)]
    return {"request": request, "prefix": s, "branches": branches}


def head_case() -> dict:
    torch.manual_seed(0)
    hidden, dim = 8, 4
    head = PointerHead(hidden, dim=dim).eval()
    with torch.no_grad():                    # a LayerNorm that is not the identity
        head.norm.weight.uniform_(0.5, 1.5)
        head.norm.bias.uniform_(-0.2, 0.2)
    decide, options = torch.randn(3, hidden) * 3 + 1, torch.randn(3, 5, hidden) * 2
    kinds = ["choice", "noul", "score"]
    with torch.no_grad():
        logits = apply_temperature(head(decide, options), torch.tensor([TEMPERATURE[k] for k in kinds]))
    return {"weights": {f"{k.replace('.', '_')}": v.tolist() for k, v in head.state_dict().items()},
            "dim": dim, "kinds": kinds, "decide": decide.tolist(), "options": options.tolist(),
            "logits": logits.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", help="also check that this tokenizer (the checkpoint's) gives the same")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, local_files_only=True)
    cases = [encode(tok, r) for r in REQUESTS]
    if a.tokenizer:
        other = AutoTokenizer.from_pretrained(a.tokenizer)
        assert [encode(other, r) for r in REQUESTS] == cases, "the checkpoint's tokenizer encodes differently"
    OUT.write_text(json.dumps({"tokenizer": {"repo": REPO, "revision": REVISION}, "max_length": MAX_LENGTH,
                               "cases": cases, "head": head_case()}, ensure_ascii=False) + "\n")
    print(f"{OUT}: {len(cases)} requests, {sum(len(c['branches']) for c in cases)} questions")


if __name__ == "__main__":
    main()
