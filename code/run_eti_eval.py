#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Chance AI
"""Evaluate ETI capture policies on benchmark episodes (scheduler simulation).

Deterministic, event-driven policies over REAL track timelines (oracle
perception: per-track windows are given; no pixels). This tests the capture
SCHEDULING, not a detector. Model-dependent quantities are explicit knobs.

Policies:
  monolithic      - one end-to-end pass with latency L; misses tracks with dwell < L
  fixed_sampling  - decide every 1/fps s; misses tracks no sample lands in
  saliency_gate   - goal-agnostic: capture iff bbox area >= median (real geometry)
  remember        - event-driven remembering on entry, under a per-sequence budget B
  oracle          - capture everything (upper bound)

Metrics are budget-conditioned; capture-recall is reported with the deadline
curve (recall vs dwell). No MOTA/HOTA/SUC/AUC.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from build_eti_benchmark import Episode

TAU_BUCKETS = [0.5, 1.0, 2.0, 4.0]


@dataclass
class Result:
    captured: bool
    ttc: float | None


def _seq(tid: str) -> str:
    return tid.rsplit("#", 1)[0]


def _bucket(dwell: float) -> float:
    return min(TAU_BUCKETS, key=lambda b: abs(b - dwell))


def p_monolithic(ep: Episode, cfg, ctx) -> Result:
    latency = cfg.monolithic_latency
    captured = latency <= ep.dwell
    return Result(captured, latency if captured else None)


def p_fixed_sampling(ep: Episode, cfg, ctx) -> Result:
    period = 1.0 / cfg.fps
    n = math.floor(ep.t_out / period) - math.ceil(ep.t_in / period) + 1
    if n >= 1:
        return Result(True, math.ceil(ep.t_in / period) * period - ep.t_in)
    return Result(False, None)


def p_saliency(ep: Episode, cfg, ctx) -> Result:
    thr = ctx["salience_thr"]
    captured = ep.salience is not None and ep.salience >= thr
    return Result(captured, cfg.gate_ttc if captured else None)


def p_oracle(ep: Episode, cfg, ctx) -> Result:
    return Result(True, 0.0)


POLICIES = {
    "monolithic": p_monolithic,
    "fixed_sampling": p_fixed_sampling,
    "saliency_gate": p_saliency,
    "oracle": p_oracle,
}


def remember_under_budget(episodes: list[Episode], cfg) -> dict[float, dict[str, Result]]:
    """Event-driven FIFO capture, capped per sequence at ceil(B * n_entries)."""
    by_seq: dict[str, list[Episode]] = defaultdict(list)
    for e in episodes:
        by_seq[_seq(e.target_id)].append(e)
    out: dict[float, dict[str, Result]] = {}
    for B in cfg.budgets:
        res: dict[str, Result] = {}
        for seq, eps in by_seq.items():
            ordered = sorted(eps, key=lambda e: e.t_in)
            k = math.ceil(B * len(ordered))
            for i, e in enumerate(ordered):
                captured = i < k
                res[e.target_id] = Result(captured, cfg.gate_ttc if captured else None)
        out[B] = res
    return out


def p_lazy_gate(ep: Episode, delta: float, horizon: float) -> bool:
    """Periodic gate (period delta) that may only act within `horizon` s of entry.

    Models a bounded passive store: evidence is overwritten after `horizon`, so a
    target is captured only if a decision instant falls in
    [t_in, min(t_out, t_in + horizon)]. The event-driven remembering is horizon-robust.
    """
    window = min(ep.dwell, horizon)
    k0 = math.ceil(ep.t_in / delta)
    k1 = math.floor((ep.t_in + window) / delta)
    return k1 >= k0


def horizon_sweep(episodes: list[Episode], delta: float, grid: list[float]) -> list[dict]:
    rows = []
    n = len(episodes) or 1
    for h in grid:
        recall = sum(p_lazy_gate(e, delta, h) for e in episodes) / n
        rows.append({"decision_period_s": delta, "buffer_horizon_s": h, "capture_recall": round(recall, 4)})
    return rows


def aggregate(episodes: list[Episode], results: dict[str, Result]) -> dict:
    n = len(episodes) or 1
    captured = sum(1 for e in episodes if results[e.target_id].captured)
    ttc = [results[e.target_id].ttc for e in episodes
           if results[e.target_id].captured and results[e.target_id].ttc is not None]
    by_bucket: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for e in episodes:
        by_bucket[_bucket(e.dwell)][0] += 1
        by_bucket[_bucket(e.dwell)][1] += int(results[e.target_id].captured)
    return {
        "n": len(episodes),
        "capture_recall": round(captured / n, 4),
        "window_miss_rate": round(1 - captured / n, 4),
        "ttc_mean": round(sum(ttc) / len(ttc), 4) if ttc else None,
        "deadline_curve": {str(b): round(v[1] / v[0], 4) for b, v in sorted(by_bucket.items())},
    }


def load_episodes(path: Path) -> list[Episode]:
    eps = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                eps.append(Episode(**json.loads(line)))
    return eps


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ETI capture policies (scheduler simulation).")
    ap.add_argument("--episodes", type=Path, default=Path("out/eti_mot/episodes.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("out/eti_mot_eval"))
    ap.add_argument("--monolithic-latency", type=float, default=1.5)
    ap.add_argument("--latency-grid", type=float, nargs="*", default=[5.0, 12.0],
                    help="extra monolithic latency values (s); the base L is --monolithic-latency")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--gate-ttc", type=float, default=0.05)
    ap.add_argument("--budgets", type=float, nargs="*", default=[0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--lazy-delta", type=float, default=0.5, help="decision period (s) for the bounded-horizon gate")
    ap.add_argument("--horizons", type=float, nargs="*", default=[0.25, 0.5, 1.0, 2.0, 1000.0])
    args = ap.parse_args()

    episodes = load_episodes(args.episodes)
    saliences = [e.salience for e in episodes if e.salience is not None]
    args.salience_thr = statistics.median(saliences) if saliences else 0.0

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"mode": "scheduler-simulation (oracle perception)",
               "n": len(episodes),
               "monolithic_latency_s": args.monolithic_latency,
               "sampling_fps": args.fps,
               "salience_median": args.salience_thr,
               "policies": {}}

    for name, fn in POLICIES.items():
        results = {e.target_id: fn(e, args, {"salience_thr": args.salience_thr}) for e in episodes}
        summary["policies"][name] = aggregate(episodes, results)

    for L in args.latency_grid:
        cfg = argparse.Namespace(**vars(args))
        cfg.monolithic_latency = L
        results = {e.target_id: p_monolithic(e, cfg, {}) for e in episodes}
        summary["policies"][f"monolithic_L{L}"] = aggregate(episodes, results)

    pareto = []
    for B, results in remember_under_budget(episodes, args).items():
        agg = aggregate(episodes, results)
        summary["policies"][f"remember_b{B}"] = agg
        pareto.append({"policy": "remember", "budget": B, "capture_recall": agg["capture_recall"]})

    horizon = horizon_sweep(episodes, args.lazy_delta, args.horizons)
    summary["horizon_sweep"] = horizon

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out / "deadline_curve.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "tau_bucket", "capture_recall"])
        for name, res in summary["policies"].items():
            for tau, rec in res["deadline_curve"].items():
                w.writerow([name, tau, rec])
    with (args.out / "pareto.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "budget", "capture_recall"])
        for row in pareto:
            w.writerow([row["policy"], row["budget"], row["capture_recall"]])
    with (args.out / "horizon_sweep.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["decision_period_s", "buffer_horizon_s", "capture_recall"])
        for row in horizon:
            w.writerow([row["decision_period_s"], row["buffer_horizon_s"], row["capture_recall"]])

    print(f"[scheduler-simulation] {len(episodes)} episodes -> {args.out}")
    for name, res in summary["policies"].items():
        print(f"  {name:16s} recall={res['capture_recall']:.3f}  wmr={res['window_miss_rate']:.3f}  "
              f"ttc={res['ttc_mean']}  curve={res['deadline_curve']}")
    print(f"  horizon_sweep (lazy gate, delta={args.lazy_delta}s):")
    for row in horizon:
        print(f"    H={row['buffer_horizon_s']:>7.2f}s recall={row['capture_recall']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
