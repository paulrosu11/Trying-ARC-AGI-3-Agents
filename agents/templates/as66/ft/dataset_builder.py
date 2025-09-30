#!/usr/bin/env python3
"""
Build SFT-ready JSONL from AS66 multi-turn conversations.

Input: transcripts/*/as66/sft_multi_turn.jsonl  (one episode per line)
Each line is:
{
  "id": "...",
  "messages": [ system, user, assistant(tool_calls), tool, user, assistant(tool_calls), ... ],
  "tools": [ACTION1..4 schemas],
  "meta": {...}
}

Output:
  /usr/xtmp/par55/huggingface_cache/as66_manual_traces/{train.jsonl,val.jsonl}

We keep conversations as-is; TRL SFTTrainer will apply the model's chat template
and learn to produce the tool call at each assistant turn.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Iterable

def _iter_jsonl(p: Path):
    if not p.exists(): return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def collect_episodes(transcripts_root: Path):
    for stamp_dir in sorted(transcripts_root.glob("*/as66")):
        multi = stamp_dir / "sft_multi_turn.jsonl"
        for row in _iter_jsonl(multi):
            # Basic sanity: must have system + at least one assistant tool_call
            msgs = row.get("messages") or []
            if not msgs or msgs[0].get("role") != "system":
                continue
            yield {
                "id": row.get("id"),
                "messages": msgs,
                "tools": row.get("tools") or [],
                "meta": row.get("meta") or {},
            }

def split(items, val_ratio: float):
    items = list(items)
    n = len(items)
    v = max(1, int(n * val_ratio)) if n > 0 else 0
    return items[v:], items[:v]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts-root", default="transcripts")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    args = ap.parse_args()

    tr = Path(args.transcripts_root).resolve()
    out = Path(args.out_root).resolve()
    out.mkdir(parents=True, exist_ok=True)

    episodes = list(collect_episodes(tr))
    train, val = split(episodes, args.val_ratio)

    def dump(path: Path, items):
        with path.open("w", encoding="utf-8") as f:
            for ex in items:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    dump(out / "train.jsonl", train)
    dump(out / "val.jsonl", val)
    print(f"[dataset_builder] episodes: train={len(train)} val={len(val)} → {out}")

if __name__ == "__main__":
    main()
