#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Chance AI
"""Build attrs.csv (target_id, query, answer_label) for the ETI benchmark.

For LaSOT, target_id is the sequence folder name. Queries default to a template;
answer labels come from an optional label CSV, otherwise ``unknown``. ``--mock``
emits a synthetic attribute table.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ATTRS = ["round_neck", "v_neck", "denim", "leather", "striped", "plain",
         "red", "blue", "black", "white"]


def read_labels(path: Path | None) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if path and path.exists():
        with path.open() as fh:
            for row in csv.DictReader(fh):
                out[row["target_id"]] = row
    return out


def lasot_targets(lasot_root: Path) -> list[str]:
    return sorted(gt.parent.name for gt in lasot_root.rglob("groundtruth.txt"))


def write_rows(rows: list[dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target_id", "query", "answer_label"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build attrs.csv for ETI episodes.")
    ap.add_argument("--lasot-root", type=Path)
    ap.add_argument("--labels", type=Path, help="CSV: target_id,query,answer_label (partial OK)")
    ap.add_argument("--default-query", default="what attribute does the target have?")
    ap.add_argument("--out", type=Path, default=Path("data/attrs.csv"))
    ap.add_argument("--mock", type=int, default=0, help="emit N synthetic rows")
    args = ap.parse_args()

    if args.mock:
        rng = random.Random(42)
        rows = [{"target_id": f"mock{i:05d}", "query": args.default_query,
                 "answer_label": rng.choice(ATTRS)} for i in range(args.mock)]
        write_rows(rows, args.out)
        print(f"wrote {len(rows)} mock attrs -> {args.out}")
        return 0

    if not args.lasot_root:
        raise SystemExit("--lasot-root required (or use --mock)")
    labels = read_labels(args.labels)
    rows = []
    for tid in lasot_targets(args.lasot_root):
        meta = labels.get(tid, {})
        rows.append({"target_id": tid,
                     "query": meta.get("query", args.default_query),
                     "answer_label": meta.get("answer_label", "unknown")})
    write_rows(rows, args.out)
    print(f"wrote {len(rows)} attrs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
