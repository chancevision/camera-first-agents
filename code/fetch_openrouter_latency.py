#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Chance AI
"""Read OpenRouter's aggregated per-endpoint latency/throughput stats.

OpenRouter's public API returns null latency, but each model page embeds the
stats payload. This fetches https://openrouter.ai/<author>/<slug>, extracts each
endpoint's p50 latency (ms) / throughput (tok/s) over the last 30 min, and reports
the request-weighted p50 across providers.

Usage: python3 scripts/fetch_openrouter_latency.py [model ...]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

DEFAULT_MODELS = [
    "openai/gpt-6-sol", "openai/gpt-6-luna", "openai/gpt-5.6-terra", "openai/gpt-5.6-sol",
    "anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5",
    "google/gemini-3.8-flash", "google/gemini-3.7-flash", "google/gemini-3.5-flash-lite",
    "qwen/qwen3-vl-235b-a22b-instruct", "qwen/qwen3-vl-8b-instruct",
]
OBJ = re.compile(
    r'"p50_latency":([0-9.]+).*?"p50_throughput":([0-9.]+).*?"request_count":(\d+)'
)


def fetch_stats(slug: str, out_tokens: int = 64) -> dict:
    req = urllib.request.Request(f"https://openrouter.ai/{slug}",
                                 headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "ignore")
    text = html.replace('\\"', '"')
    rows = OBJ.findall(text)
    endpoints = [{"p50_ms": float(a), "thr": float(b), "req": int(c)} for a, b, c in rows]
    total = sum(e["req"] for e in endpoints) or 1
    w = sum(e["p50_ms"] * e["req"] for e in endpoints) / total
    thr = sum(e["thr"] * e["req"] for e in endpoints) / total
    est_complete_ms = w + (out_tokens / thr) * 1000 if thr else None
    return {"slug": slug, "endpoints": endpoints,
            "p50_ms_min": min((e["p50_ms"] for e in endpoints), default=None),
            "p50_ms_weighted": round(w, 1) if endpoints else None,
            "throughput_weighted": round(thr, 1) if endpoints else None,
            "est_complete_ms": round(est_complete_ms, 1) if est_complete_ms else None,
            "est_out_tokens": out_tokens,
            "n_endpoints": len(endpoints)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Read OpenRouter aggregated latency stats.")
    ap.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--out", type=Path, default=Path("out/openrouter_latency.json"))
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--out-tokens", type=int, default=64, help="assumed answer length for time-to-answer")
    args = ap.parse_args()

    results = []
    for slug in args.models:
        try:
            r = fetch_stats(slug, args.out_tokens)
        except Exception as exc:
            r = {"slug": slug, "error": str(exc)[:100]}
        results.append(r)
        time.sleep(args.sleep)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"{'model':38s} {'ttfb p50(s)':>11s} {'thr tok/s':>9s} {'time-to-answer(s)':>17s}")
    for r in results:
        if "error" in r:
            print(f"{r['slug']:38s}  -- {r['error']}")
        else:
            t = f"{r['p50_ms_weighted']/1000:.2f}" if r["p50_ms_weighted"] else "?"
            th = str(r.get("throughput_weighted", "?"))
            e = f"{r['est_complete_ms']/1000:.2f}" if r.get("est_complete_ms") else "?"
            print(f"{r['slug']:38s} {t:>11s} {th:>9s} {e:>17s}")
    print(f"\nwrote {args.out}  (OpenRouter, last-30-min, request-weighted; time-to-answer for {args.out_tokens} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
