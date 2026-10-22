#!/usr/bin/env python3
"""PolQA as 4-option reading comprehension, in the Belebele format: a relevant Wikipedia
passage, the question, its gold answer and three answers to other questions of the same entity
type, in random order.

    uv run python scripts/build_polqa_mc.py --out data/examples

* Only PolQA train rows (split by question group, the same groups as ``polqa.jsonl``) and
  validation rows (``dev``). A question or passage that matches any PolQA test text or any
  Belebele passage or question is dropped (``overlap_key``).
* A distractor never equals one of the question's gold answers and never occurs in the passage.
* Writes ``<out>/polqa_mc.jsonl`` (family ``polqa_mc``).
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from bullfinch.questions import DecisionExample, TypedQuestion
from bullfinch.sources import overlap_key
from bullfinch.splits import split_for

POLQA = "hf://datasets/ipipan/polqa@refs%2Fconvert%2Fparquet/pairs/{}/0000.parquet"


def norm(text: str) -> str:
    return " ".join(str(text).split()).lower()


def answer_list(raw) -> list[str]:
    """The parquet stores answers as the text of a Python list, e.g. "['alfa']"."""
    items = ast.literal_eval(raw) if isinstance(raw, str) and raw.startswith("[") else raw
    items = [items] if isinstance(items, str) else list(items)
    return [str(a).strip() for a in items if str(a).strip()]


def shape(answer: str) -> tuple:
    """Words (capped at 3), capitalised, has a digit: distractors look like the gold answer."""
    return min(len(answer.split()), 3), answer[:1].isupper(), any(c.isdigit() for c in answer)


SKIP_TYPES = {"YES/NO", "ENTITY CHOICE"}  # tak/nie, or the options are already in the question


def build(df, hf_split: str, banned: set[str], dropped: Counter) -> list[DecisionExample]:
    answers = {qid: answer_list(g.iloc[0]["answers"]) for qid, g in df.groupby("question_id")}
    pools: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for qid, g in df.groupby("question_id"):
        first = g.iloc[0]
        if first["question_type"] in SKIP_TYPES:
            continue
        for a in answers[qid]:
            pools[("sub", str(first["entity_subtype"]), shape(a))].append((qid, a))
            pools[("type", str(first["entity_type"]), shape(a))].append((qid, a))
            pools[("shape", shape(a))].append((qid, a))

    out = []
    for qid, g in df[df.relevant == True].groupby("question_id"):  # noqa: E712
        if not answers.get(qid) or g.iloc[0]["question_type"] in SKIP_TYPES:
            continue
        rng = random.Random(f"polqa_mc-{qid}")
        row = g.iloc[rng.randrange(len(g))]
        question, passage = str(row["question"]).strip(), str(row["passage_text"]).strip()
        if overlap_key(question) in banned or overlap_key(passage) in banned:
            dropped[hf_split] += 1
            continue
        gold = answers[qid][0]
        golds, low_passage = {norm(a) for a in answers[qid]}, norm(passage)
        seen, distractors = set(golds), []
        sub, s = str(row["entity_subtype"]), shape(gold)
        tiers = ([pools[("sub", sub, s)]] if sub != "-" else []) + [pools[("type", str(row["entity_type"]), s)],
                                                                    pools[("shape", s)]]
        candidates = []
        for tier in tiers:
            tier = tier[:]
            rng.shuffle(tier)
            candidates += tier
        for other, a in candidates:
            if other != qid and norm(a) not in seen and norm(a) not in low_passage:
                distractors.append(a)
                seen.add(norm(a))
                if len(distractors) == 3:
                    break
        if len(distractors) < 3:
            continue
        options = [gold, *distractors]
        rng.shuffle(options)
        key = str(options.index(gold) + 1)
        title = str(row.get("passage_title") or "").strip()
        group = f"polqa:{qid}"
        out.append(DecisionExample(
            id=f"polqa_mc-{hf_split}-{qid}", state=f"{title}\n\n{passage}" if title else passage,
            question=TypedQuestion("choice", question, {str(i + 1): o for i, o in enumerate(options)}),
            answer=key, family="polqa_mc", language="pl", source="ipipan/polqa", group=group,
            meta={"label": key, "hf_split": hf_split, "entity_type": str(row["entity_type"])},
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/examples"))
    args = ap.parse_args()

    import pandas as pd
    from datasets import load_dataset

    test = pd.read_parquet(POLQA.format("test"))
    bele = load_dataset("facebook/belebele", "pol_Latn", split="test")
    banned = {overlap_key(t) for t in [*test["question"], *test["passage_text"]]}
    banned |= {overlap_key(t) for r in bele for t in (r["flores_passage"], r["question"])}

    dropped: Counter = Counter()
    recs = []
    for hf_split, split in (("train", None), ("validation", "dev")):
        for ex in build(pd.read_parquet(POLQA.format(hf_split)), hf_split, banned, dropped):
            rec = dataclasses.asdict(ex)
            rec["split"] = split or split_for(ex.group)
            recs.append(rec)
    path = args.out / "polqa_mc.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"polqa_mc: {len(recs)} rows {dict(Counter(r['split'] for r in recs))}, "
          f"gold position {dict(sorted(Counter(r['answer'] for r in recs).items()))}, dropped for overlap {dict(dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
