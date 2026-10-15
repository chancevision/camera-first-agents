#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Chance AI
"""Measure VLM time-to-first-token (TTFB) and total latency via OpenRouter.

Sends one small probe image + a fixed prompt to each model with streaming and
times the first SSE delta. Outputs a table and out/vlm_latency.json.

Requires OPENROUTER_API_KEY. Latency includes network RTT to OpenRouter, i.e. the
end-to-end API latency a camera-first agent would actually see.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

API = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = [
    "openai/gpt-6-sol", "openai/gpt-6-luna", "openai/gpt-5.6-terra", "openai/gpt-5.6-sol",
    "anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5",
    "google/gemini-3.8-flash", "google/gemini-3.7-flash", "google/gemini-3.5-flash-lite",
    "qwen/qwen3-vl-235b-a22b-instruct", "qwen/qwen3-vl-8b-instruct",
]
PROMPT = "Name the color of the circle in this image in one word."


def probe_image_b64() -> str:
    fig, ax = plt.subplots(figsize=(1.2, 1.2), dpi=64)
    ax.add_patch(plt.Circle((0.5, 0.5), 0.3, color="red"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    path = Path("/tmp/eti_probe.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def one_call(model: str, img_b64: str, api_key: str, max_tokens: int) -> dict:
    body = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]}],
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://chance.vision", "X-Title": "paper4-latency"},
    )
    t0 = time.perf_counter()
    ttfb = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=90) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0].get("delta", {})
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
            if delta.get("content"):
                if ttfb is None:
                    ttfb = time.perf_counter() - t0
                ntok += 1
    total = time.perf_counter() - t0
    return {"model": model, "ttfb_s": ttfb, "total_s": round(total, 3), "chunks": ntok}


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure frontier VLM TTFB via OpenRouter.")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between calls (rate limits)")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("out/vlm_latency.json"))
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    img = probe_image_b64()

    results = []
    for model in args.models:
        runs = []
        for i in range(args.runs):
            time.sleep(args.sleep)
            try:
                r = one_call(model, img, key, args.max_tokens)
            except urllib.error.HTTPError as exc:
                r = {"model": model, "error": f"HTTP {exc.code}", "detail": exc.read().decode()[:120]}
            except Exception as exc:
                r = {"model": model, "error": str(exc)[:120]}
            runs.append(r)
            print(f"  {model:40s} run{i} ttfb={r.get('ttfb_s')} total={r.get('total_s')} err={r.get('error','')}")
        ok = [r["ttfb_s"] for r in runs if isinstance(r.get("ttfb_s"), (int, float))]
        results.append({"model": model, "runs": runs,
                        "ttfb_median_s": round(sorted(ok)[len(ok) // 2], 3) if ok else None})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n{'model':40s} {'TTFB median (s)':>16s}")
    for r in sorted(results, key=lambda x: (x["ttfb_median_s"] is None, x["ttfb_median_s"] or 0)):
        print(f"{r['model']:40s} {str(r['ttfb_median_s']):>16s}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
