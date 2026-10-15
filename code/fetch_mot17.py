#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Chance AI
"""Fetch MOT17 ground-truth (gt.txt) for enter/exit ephemeral episodes.

Only gt.txt is downloaded (small); the ETI benchmark needs track lifetimes,
not pixels. Public via a HuggingFace mirror.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/Lekim89/MOT17/resolve/main"
DEFAULT_SEQS = [
    "MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
    "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN",
]


def fetch(seq: str, split: str, dest: Path) -> Path:
    url = f"{BASE}/{split}/{seq}/gt/gt.txt"
    out = dest / seq / "gt" / "gt.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url}")
    urllib.request.urlretrieve(url, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch MOT17 gt.txt for ETI episodes.")
    ap.add_argument("--dest", type=Path, default=Path("data/mot17"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--seqs", nargs="*", default=DEFAULT_SEQS)
    args = ap.parse_args()
    n = 0
    for seq in args.seqs:
        try:
            fetch(seq, args.split, args.dest)
            n += 1
        except Exception as exc:
            print(f"  skip {seq}: {exc}")
    print(f"staged {n} sequences under {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
