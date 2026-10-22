#!/usr/bin/env python3
"""Build decision examples from the labelled Polish sets (see README, Data).

    uv run python scripts/build_dataset.py --out data/examples

Writes, under --out:

* ``<source>.jsonl``  one DecisionExample per row with its ``split``: ``test`` for official
  test rows, ``dev`` for official validation rows, and train rows split by group into
  train / dev / calibration (``bullfinch.splits``);
* ``manifest.json``   counts per source, split and answer, rows dropped for overlap, hashes.

Non-test rows that share a text with any test row of the same pool are dropped, so no test
text is ever trained on, even paired with something else.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from bullfinch.sources import SOURCES, text_key
from bullfinch.splits import split_for

MASSIVE = {"path": "parquet", "data_files": {
    s: f"hf://datasets/AmazonScience/massive@refs%2Fconvert%2Fparquet/pl-PL/{s}/0000.parquet"
    for s in ("train", "validation", "test")}}  # the repo itself is a loading script

# (source, hf id / (id, config) / load_dataset kwargs, hf split, our split or None to split
# train rows by group)
PLAN = [
    ("polemo_in", "allegro/klej-polemo2-in", "train", None),
    ("polemo_in", "allegro/klej-polemo2-in", "validation", "dev"),
    ("polemo_in", "allegro/klej-polemo2-in", "test", "test"),
    ("polemo_out", "allegro/klej-polemo2-out", "test", "test"),
    ("dyk", "allegro/klej-dyk", "train", None),
    ("dyk", "allegro/klej-dyk", "test", "test"),
    ("psc", "allegro/klej-psc", "train", None),
    ("psc", "allegro/klej-psc", "test", "test"),
    ("massive", MASSIVE, "train", None),
    ("massive", MASSIVE, "validation", "dev"),
    ("massive", MASSIVE, "test", "test"),
    # evaluation only
    ("belebele", ("facebook/belebele", "pol_Latn"), "test", "test"),
    ("include", ("CohereLabs/include-base-44", "Polish"), "test", "test"),
    ("opendecisions", "jgeuter/OpenDecisions", "test", "test"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/examples"))
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("needs the data extra: `uv sync --extra data`") from exc

    raw: list[tuple[str, str, str | None, list[dict]]] = []
    for name, spec, hf_split, split in PLAN:
        if isinstance(spec, dict):
            kwargs = spec
        else:
            hf_id, config = spec if isinstance(spec, tuple) else (spec, None)
            kwargs = {"path": hf_id, "name": config}
        print(f"loading {name} [{hf_split}] ...")
        ds = load_dataset(**kwargs, split=hf_split)
        labels = {c: f for c, f in ds.features.items() if hasattr(f, "int2str")}
        rows = [{**r, **{f"{c}_name": f.int2str(r[c]) for c, f in labels.items()}} for r in ds]
        raw.append((name, hf_split, split, rows))

    # Every text of every test row, per pool.
    test_texts: dict[str, set[str]] = defaultdict(set)
    for name, _, split, rows in raw:
        if split == "test":
            src = SOURCES[name]
            test_texts[src.pool].update(text_key(r[f]) for r in rows for f in src.texts if r.get(f))

    args.out.mkdir(parents=True, exist_ok=True)
    out_rows: dict[str, list[dict]] = defaultdict(list)
    dropped: Counter = Counter()
    for name, hf_split, split, rows in raw:
        src = SOURCES[name]
        for idx, row in enumerate(rows):
            if split != "test" and any(text_key(row[f]) in test_texts[src.pool] for f in src.texts if row.get(f)):
                dropped[f"{name}:{hf_split}"] += 1
                continue
            for ex in src.convert(row, idx, hf_split):
                rec = dataclasses.asdict(ex)
                rec["split"] = split or split_for(ex.group)
                rec["meta"]["hf_split"] = hf_split
                out_rows[name].append(rec)

    manifest: dict = {"sources": {}, "dropped_test_overlap": dict(dropped)}
    for name, recs in out_rows.items():
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        by_split = Counter(r["split"] for r in recs)
        labels = defaultdict(Counter)
        for r in recs:
            labels[r["split"]][r["meta"]["label"]] += 1
        manifest["sources"][name] = {
            "rows": len(recs), "splits": dict(by_split),
            "labels": {s: dict(c) for s, c in labels.items()}, "sha256": sha256(path),
        }
        print(f"{name}: {len(recs)} rows {dict(by_split)}")
    for key, n in sorted(dropped.items()):
        print(f"dropped {n} rows of {key}: they share a text with a test row")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
