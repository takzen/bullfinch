#!/usr/bin/env python3
"""Fit per-question-type temperatures for a trained adapter (CALIBRATION.json for the server).

    uv run python scripts/calibrate.py --run artifacts/pilot [--tta] \
        --examples data/examples/polemo.jsonl data/examples/dyk.jsonl ...

* The panel is drawn with the run's --mix from the calibration split of its --examples: rows the
  converters set aside, never trained on and never used to pick a checkpoint.
* Rows are scored exactly as the server scores them (``DecisionEngine.logprobs``, with --tta
  when the adapter is served that way). A temperature divides every option logit, so it never
  changes which option wins; it only makes the confidences honest.
* Each type with at least --min-type rows gets the temperature with the lowest NLL on a
  log-spaced grid; rarer types keep 1.0.

Serve with ``--calibration <run>/CALIBRATION.json`` (``-tta`` suffix when fitted with --tta).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bullfinch.engine import softmax  # noqa: E402
from bullfinch.questions import TEMPLATE_VERSION  # noqa: E402
from train_qlora import load_examples, sample  # noqa: E402

GRID = [math.exp(math.log(0.25) + i * (math.log(8.0) - math.log(0.25)) / 200) for i in range(201)]


def nll(rows: list[tuple[dict[str, float], dict[str, float]]], temperature: float) -> float:
    total = 0.0
    for logp, target in rows:
        keys = list(logp)
        probs = softmax([logp[k] for k in keys], temperature)
        total -= sum(target.get(k, 0.0) * math.log(max(p, 1e-12)) for k, p in zip(keys, probs))
    return total / len(rows)


def ece(rows: list[tuple[dict[str, float], dict[str, float]]], temperature: float, bins: int = 10) -> float:
    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for logp, target in rows:
        keys = list(logp)
        probs = softmax([logp[k] for k in keys], temperature)
        top = max(range(len(keys)), key=probs.__getitem__)
        gold = max(target, key=target.get)
        buckets[min(bins - 1, int(probs[top] * bins))].append((probs[top], float(keys[top] == gold)))
    return sum(len(b) / len(rows) * abs(sum(c for c, _ in b) / len(b) - sum(a for _, a in b) / len(b))
               for b in buckets.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="training output dir (adapter/, run_config.json)")
    ap.add_argument("--examples", type=Path, nargs="+", required=True, help="the run's --examples")
    ap.add_argument("--tta", action="store_true", help="calibrate the --tta serving mode")
    ap.add_argument("--n", type=int, default=600, help="panel size")
    ap.add_argument("--min-type", type=int, default=50, help="fewest rows to fit a type")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    run_args = json.loads((args.run / "run_config.json").read_text(encoding="utf-8"))["args"]
    mix = {k: float(v) for k, v in (p.split("=") for p in run_args["mix"].split(","))}
    print("loading examples ...")
    pool = [ex for ex, split in load_examples(args.examples) if split == "calibration"]
    print("calibration panel:")
    panel = sample(pool, args.n, mix, random.Random(int(run_args["seed"]) + 2))

    from bullfinch.engine import DecisionEngine

    engine = DecisionEngine(adapter=str(args.run / "adapter"), max_length=args.max_tokens, tta=args.tta)
    by_type: dict[str, list[tuple[dict[str, float], dict[str, float]]]] = defaultdict(list)
    skipped, t0 = 0, time.perf_counter()
    for i, ex in enumerate(panel, start=1):
        try:
            logp = engine.logprobs(ex.state, ex.question)
        except ValueError:  # longer than --max-tokens
            skipped += 1
            continue
        by_type[ex.question.type].append((logp, ex.soft_targets or {ex.answer: 1.0}))
        if i % 50 == 0:
            print(f"  {i}/{len(panel)} scored ({time.perf_counter() - t0:.0f} s)", flush=True)

    result = {"temperature": {}, "template": TEMPLATE_VERSION, "adapter": str(args.run / "adapter"),
              "tta": args.tta, "panel": {"n": sum(map(len, by_type.values())), "skipped_long": skipped,
                                         "families": Counter(ex.family.split(":")[0] for ex in panel)},
              "by_type": {}}
    for qtype, rows in sorted(by_type.items()):
        best = min(GRID, key=lambda t: nll(rows, t)) if len(rows) >= args.min_type else 1.0
        result["temperature"][qtype] = round(best, 3)
        result["by_type"][qtype] = {
            "n": len(rows), "nll_before": round(nll(rows, 1.0), 4), "nll_after": round(nll(rows, best), 4),
            "ece_before": round(ece(rows, 1.0), 4), "ece_after": round(ece(rows, best), 4)}
        print(f"{qtype}: n {len(rows)}  T {best:.3f}  " + "  ".join(f"{k} {v}" for k, v in result["by_type"][qtype].items()))
    out = args.run / ("CALIBRATION-tta.json" if args.tta else "CALIBRATION.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
