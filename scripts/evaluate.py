#!/usr/bin/env python3
"""Score decision examples with the engine and report metrics per family.

    uv run python scripts/evaluate.py --examples data/examples/*.jsonl --tag base \
        [--adapter artifacts/pilot/adapter | --ollama MODEL | --url http://host:8000] [--tta] [--limit 500]

* ``--limit N`` scores a random sample of N rows per family (seed 7), the same rows for every
  model, so runs with the same --examples and --limit are directly comparable.

* Every row of --split (default ``test``) is scored exactly as the server scores it
  (``DecisionEngine.logprobs``: fixed option order, frame from the question's language).
* ``--ollama`` scores a model served by Ollama instead (``bullfinch.ollama``), e.g.
  ``SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M``; the prompt and the scoring are the same.
* ``--url`` scores another decision server over ``POST /v1/systemone`` (``bullfinch.systemone``):
  state and typed question as they are, no Bullfinch template; ``--key-env`` names the env var
  holding its API key.
* Writes ``<out>/<tag>.jsonl`` (one scored row per example, so metrics can be recomputed
  without the GPU) and ``<out>/<tag>.json`` (``bullfinch.metrics.summarise`` per family).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bullfinch.metrics import summarise  # noqa: E402
from bullfinch.sources import canonical_label  # noqa: E402
from train_qlora import load_examples  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=Path, nargs="+", required=True)
    ap.add_argument("--tag", required=True, help="name of this run in --out")
    ap.add_argument("--out", type=Path, default=Path("artifacts/bench"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--adapter", help="LoRA adapter directory (default: untuned base)")
    ap.add_argument("--base", help="base model for the local engine (default: Qwen/Qwen3.5-4B, pinned)")
    ap.add_argument("--revision", help="pinned revision of --base")
    ap.add_argument("--ollama", help="score this Ollama model instead of the local engine")
    ap.add_argument("--url", help="score a /v1/systemone server instead of the local engine")
    ap.add_argument("--key-env", help="env var with the --url server's API key")
    ap.add_argument("--model", help="model name to send to the --url server")
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--limit", type=int, help="random sample of N rows per family (seed 7)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    per_family: dict[str, list] = defaultdict(list)
    for ex, split in load_examples(args.examples):
        if split == args.split:
            per_family[ex.family].append(ex)
    if args.limit is not None:
        rng = random.Random(7)
        per_family = {k: rng.sample(v, min(args.limit, len(v))) for k, v in sorted(per_family.items())}
    panel = [ex for exs in per_family.values() for ex in exs]
    print(f"{len(panel)} rows: " + ", ".join(f"{k} {len(v)}" for k, v in per_family.items()), flush=True)

    if args.url:
        import os

        from bullfinch.systemone import SystemOneEngine

        engine = SystemOneEngine(args.url, key=os.environ.get(args.key_env) if args.key_env else None,
                                 model=args.model)
    elif args.ollama:
        from bullfinch.ollama import OllamaEngine

        engine = OllamaEngine(args.ollama, tta=args.tta, num_ctx=args.max_tokens)
    else:
        from bullfinch.engine import DecisionEngine

        base = {"model": args.base, "revision": args.revision} if args.base else {}
        engine = DecisionEngine(**base, adapter=args.adapter, calibration=args.calibration,
                                max_length=args.max_tokens, tta=args.tta)
    args.out.mkdir(parents=True, exist_ok=True)
    scored: dict[str, list[dict]] = defaultdict(list)
    skipped, t0 = 0, time.perf_counter()
    with (args.out / f"{args.tag}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for i, ex in enumerate(panel, start=1):
            try:
                logp = engine.logprobs(ex.state, ex.question)
            except ValueError:  # longer than --max-tokens
                skipped += 1
                continue
            probs = {k: math.exp(v) for k, v in logp.items()}
            total = sum(probs.values())
            probs = {k: p / total for k, p in probs.items()}
            top = max(probs, key=probs.get)
            row = {"id": ex.id, "family": ex.family, "instructions": ex.question.instructions,
                   "gold": canonical_label(ex.question, ex.answer), "pred": canonical_label(ex.question, top),
                   "p_gold": probs[ex.answer], "p_top": probs[top], "probs": probs}
            scored[ex.family].append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(panel)} ({rate:.2f} ex/s)", flush=True)

    result = {"tag": args.tag, "adapter": args.adapter or "base", "tta": args.tta, "split": args.split,
              "model": engine.model_name, "revision": engine.revision, "skipped_long": skipped,
              "seconds": round(time.perf_counter() - t0), "families": {}}
    for family, rows in sorted(scored.items()):
        m = summarise(rows)
        result["families"][family] = m
        extra = f"  f1_true {m['f1_true']}" if "f1_true" in m else ""
        print(f"{family}: n {m['n']}  acc {m['accuracy']}  macro_f1 {m['macro_f1']}{extra}  "
              f"nll {m['nll']}  ece {m['ece']}  pred {m['pred']}")
    (args.out / f"{args.tag}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", args.out / f"{args.tag}.json", f"| skipped {skipped} long prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
