"""Convert a Strands Decider checkpoint (https://github.com/strands-labs/strands-decider, Apache 2.0) for jul.

The archive holds a LoRA adapter, a pointer head (`slot_head.pt`) and the model's config; the base model is
pinned in that config. This writes a directory jul reads as a decision model (`decision.json`):

    base weights with the LoRA merged (W + B @ A * alpha / r, in float32, stored in the base's dtype),
    the base's config, the archive's tokenizer, the pointer head as `pointer_head.npz`, and decision.json.

The merge runs tensor by tensor: the model is never loaded, so it needs about one output shard of memory.

    python scripts/convert_strands_decider.py <archive.tar | extracted dir> <out dir> [--sha256 <hex>]

Needs torch, safetensors and huggingface_hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path

import numpy as np

SHARD_BYTES = 2 * 1024 ** 3
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")

TEXT_FORMAT = {
    "prefix": "<state>\n{state}\n</state>\n",
    "branch": "<question type=\"{kind}\">\n{header}\n{instructions}\n<options>\n{options}\n</options>\n</question>\n"
              "<answer>",
    "headers": {"choice": "Select exactly one option.",
                "noul": "Decide whether the statement is true of the state.",
                "score": "Rate the state against the ordered levels below (lowest first)."},
    "option": "{number}. {name}",
    "option_with_description": "{number}. {name} \u2014 {description}",
    "option_separator": "\n",
    "collapse_whitespace": True,
    "score_name": "{index}",
    "noul_defaults": {"false": "the statement does not hold for this state",
                      "true": "the statement holds for this state"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_dir(source: Path, work: Path, expected: str | None) -> Path:
    """The extracted checkpoint; an archive is checked against `expected` first, then extracted."""
    if source.is_dir():
        return source
    if expected and sha256(source) != expected:
        raise SystemExit(f"{source}: sha256 does not match {expected}")
    with tarfile.open(source) as tar:
        tar.extractall(work, filter="data")
    return next(p.parent for p in work.rglob("strands_decider_config.json"))


def lora_deltas(ckpt: Path) -> tuple[dict[str, tuple], float]:
    """{base weight name: (A, B)} and the scale alpha / r."""
    from safetensors.torch import load_file
    cfg = json.loads((ckpt / "lora" / "adapter_config.json").read_text())
    if cfg.get("use_dora") or cfg.get("use_rslora"):
        raise SystemExit("DoRA and rsLoRA adapters are not supported")
    w = load_file(ckpt / "lora" / "adapter_model.safetensors")
    pairs: dict[str, tuple] = {}
    for name in w:
        m = re.fullmatch(r"base_model\.model\.(.+)\.lora_A\.weight", name)
        if m:
            # the adapter wraps the torso (Qwen3_5Model); the checkpoint's names start one level up
            pairs[f"model.{m.group(1)}.weight"] = (w[name], w[name.replace("lora_A", "lora_B")])
    if 2 * len(pairs) != len(w):
        raise SystemExit(f"unexpected adapter tensors: {len(w)} for {len(pairs)} LoRA pairs")
    return pairs, float(cfg["lora_alpha"]) / float(cfg["r"])


def merge(base: Path, ckpt: Path, out: Path) -> int:
    """Writes the merged shards and their index; returns the number of merged weights."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    deltas, scale = lora_deltas(ckpt)
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    weight_map, shard, size, n, merged = {}, {}, 0, 0, set()

    def flush():
        nonlocal shard, size, n
        if shard:
            n += 1
            name = f"model-{n:05d}.safetensors"
            save_file(shard, out / name, metadata={"format": "pt"})
            weight_map.update({k: name for k in shard})
            shard, size = {}, 0

    for file in sorted(set(index.values())):
        with safe_open(base / file, "pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if key in deltas:
                    a, b = deltas[key]
                    t = (t.float() + (b.float() @ a.float()) * scale).to(t.dtype)
                    merged.add(key)
                shard[key] = t.contiguous()
                size += t.numel() * t.element_size()
                if size >= SHARD_BYTES:
                    flush()
    flush()
    missing = set(deltas) - merged
    if missing:
        raise SystemExit(f"{len(missing)} LoRA weights have no base weight, e.g. {sorted(missing)[0]}")
    total = sum((out / f).stat().st_size for f in set(weight_map.values()))
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))
    return len(merged)


def write_head(ckpt: Path, out: Path) -> int:
    import torch
    sd = torch.load(ckpt / "slot_head.pt", map_location="cpu", weights_only=True)
    np.savez(out / "pointer_head.npz", **{k.replace(".", "_"): v.float().numpy() for k, v in sd.items()})
    return int(sd["q.weight"].shape[0])


def decision_json(cfg: dict, dim: int, name: str) -> dict:
    max_length = int(cfg["max_length"])
    return {
        "method": "pointer",
        "about": f"{name}: {cfg['base_model']}@{cfg['base_revision'][:8]} with its LoRA merged and a pointer head "
                 "with a LayerNorm. Format of Strands Decider (prompting.py): plain-text tags, options read at "
                 "the last token of their line.",
        "text": TEXT_FORMAT,
        "noul_options": ["false", "true"],
        "state_render": "json-indent",
        "add_special_tokens": True,
        "readout": {"layer": "last_normed", "question_token": "last", "option_token": "option_line"},
        "head": {"file": "pointer_head.npz", "query": "q", "key": "k", "norm": "norm", "norm_eps": 1e-5,
                 "dim": dim, "temperature": float(cfg.get("temperature", 1.0)),
                 "temperature_by_kind": {k: float(v) for k, v in (cfg.get("temperature_by_kind") or {}).items()}},
        # the runtime keeps up to 3/4 of the window for the question and gives the state the rest
        "limits": {"max_length": max_length, "max_state_tokens": max_length,
                   "max_branch_tokens": int(max_length * 0.75)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", type=Path, help="the archive (.tar) or its extracted checkpoint directory")
    ap.add_argument("out", type=Path)
    ap.add_argument("--sha256", help="expected sha256 of the archive")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_dir(args.source, args.out / ".extract", args.sha256)
    cfg = json.loads((ckpt / "strands_decider_config.json").read_text())
    if cfg.get("head_type") != "pointer":
        raise SystemExit(f"head_type {cfg.get('head_type')!r}: only pointer heads are read")
    base = Path(snapshot_download(cfg["base_model"], revision=cfg["base_revision"]))
    n = merge(base, ckpt, args.out)
    for f in ("config.json", "generation_config.json", "preprocessor_config.json"):
        if (base / f).exists():
            shutil.copy(base / f, args.out / f)
    for f in TOKENIZER_FILES:
        if (ckpt / f).exists():
            shutil.copy(ckpt / f, args.out / f)
    dim = write_head(ckpt, args.out)
    (args.out / "decision.json").write_text(json.dumps(decision_json(cfg, dim, ckpt.name), indent=1) + "\n")
    shutil.rmtree(args.out / ".extract", ignore_errors=True)
    print(f"{args.out}: {n} LoRA weights merged into {cfg['base_model']}@{cfg['base_revision'][:8]}, head dim {dim}")


if __name__ == "__main__":
    main()
