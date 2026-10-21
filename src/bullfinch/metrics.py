"""Metrics over scored decisions: accuracy, macro-F1, F1 of "true", NLL and ECE.

A scored row carries the canonical gold and predicted labels (``sources.canonical_label``),
the probability of the gold option and of the top option, and the question's instructions,
so results pool every phrasing of a task and can still be broken down per phrasing.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict


def f1(rows: list[dict], label: str) -> float:
    tp = sum(r["gold"] == label and r["pred"] == label for r in rows)
    fp = sum(r["gold"] != label and r["pred"] == label for r in rows)
    fn = sum(r["gold"] == label and r["pred"] != label for r in rows)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def ece(rows: list[dict], bins: int = 10) -> float:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[min(bins - 1, int(r["p_top"] * bins))].append(r)
    return sum(len(b) / len(rows) * abs(sum(r["p_top"] for r in b) / len(b)
                                        - sum(r["gold"] == r["pred"] for r in b) / len(b))
               for b in buckets.values())


def summarise(rows: list[dict]) -> dict:
    labels = sorted({r["gold"] for r in rows})
    out = {
        "n": len(rows),
        "accuracy": round(sum(r["gold"] == r["pred"] for r in rows) / len(rows), 4),
        "macro_f1": round(sum(f1(rows, label) for label in labels) / len(labels), 4),
        "nll": round(-sum(math.log(max(r["p_gold"], 1e-12)) for r in rows) / len(rows), 4),
        "ece": round(ece(rows), 4),
        "gold": dict(Counter(r["gold"] for r in rows)),
        "pred": dict(Counter(r["pred"] for r in rows)),
    }
    if set(labels) <= {"true", "false"}:
        out["f1_true"] = round(f1(rows, "true"), 4)
    by_phrasing: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_phrasing[r["instructions"]].append(r)
    out["accuracy_by_phrasing"] = {
        k: round(sum(r["gold"] == r["pred"] for r in v) / len(v), 4) for k, v in sorted(by_phrasing.items())
    }
    return out
