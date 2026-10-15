#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Chance AI
"""Render the figures for this paper from the ETI evaluation artifacts.

Inputs (CSV/JSON under --eval-dir):
  deadline_curve.csv, pareto.csv, horizon_sweep.csv, summary.json
Outputs (PDF under --out-dir):
  fig1_architecture.pdf, fig2_deadline_curve.pdf, fig3_budget.pdf, fig4_horizon.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

MAIN_POLICIES = ["monolithic", "monolithic_L5.0", "monolithic_L12.0", "fixed_sampling",
                 "saliency_gate", "remember_b1.0", "oracle"]
LABELS = {"monolithic": "monolithic (L=1.5s)", "monolithic_L5.0": "monolithic (L=5s)",
          "monolithic_L12.0": "monolithic (L=12s)", "fixed_sampling": "fixed sampling (2 fps)",
          "saliency_gate": "saliency gate (goal-agnostic)", "remember_b1.0": "ours (event-driven)",
          "oracle": "oracle"}
STYLES = {"monolithic": "--o", "monolithic_L5.0": "--v", "monolithic_L12.0": "--^",
          "fixed_sampling": "-s", "saliency_gate": "-x", "remember_b1.0": "-D", "oracle": ":x"}


def read_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def fig_architecture(out: Path) -> None:
    stages = [
        ("Always-on perception\n(passive, bounded)", "#e8f0fe"),
        ("Bounded working\nmemory (horizon H)", "#e8f0fe"),
        ("ACTIVE CAPTURE\n(event-driven, hard deadline)", "#fde8e8"),
        ("Durable memory\n/ workspace", "#e8f7e8"),
        ("Deferred deliberation\n(VLM + tools, soft deadline)", "#e8f7e8"),
    ]
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.axis("off")
    x = 0.0
    w, h, gap = 1.75, 1.0, 0.35
    centers = []
    for label, color in stages:
        box = FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.06",
                             linewidth=1, edgecolor="#444", facecolor=color)
        ax.add_patch(box)
        ax.text(x + w / 2, h / 2, label, ha="center", va="center", fontsize=8)
        centers.append(x + w / 2)
        x += w + gap
    for a, b in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch((a + w / 2, h / 2), (b - w / 2, h / 2),
                                     arrowstyle="-|>", mutation_scale=12, color="#333"))
    ax.text(centers[1], h + 0.45, "overwrite -> lost", ha="center", fontsize=7, color="#b00")
    ax.add_patch(FancyArrowPatch((centers[1], h + 0.3), (centers[1], h + 0.05),
                                 arrowstyle="-|>", mutation_scale=10, color="#b00"))
    ax.set_xlim(-0.2, x)
    ax.set_ylim(-0.3, h + 0.7)
    fig.tight_layout()
    fig.savefig(out / "fig1_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_deadline_curve(rows: list[dict], out: Path) -> None:
    by = {}
    for r in rows:
        by.setdefault(r["policy"], {})[float(r["tau_bucket"])] = float(r["capture_recall"])
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for pol in MAIN_POLICIES:
        if pol not in by:
            continue
        xs = sorted(by[pol])
        ys = [by[pol][x] for x in xs]
        ax.plot(xs, ys, STYLES.get(pol, "-"), label=LABELS.get(pol, pol), markersize=5,
                color="#d62728" if pol == "remember_b1.0" else None,
                linewidth=2 if pol == "remember_b1.0" else 1.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.set_xlabel("target dwell window $\\tau$ (s)")
    ax.set_ylabel("memory recall")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig2_deadline_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_budget(rows: list[dict], out: Path) -> None:
    xs = [float(r["budget"]) for r in rows]
    ys = [float(r["capture_recall"]) for r in rows]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(xs, ys, "-D", color="#d62728", label="ours (event-driven, FIFO)")
    ax.plot([0, 1], [0, 1], ":", color="#888", label="recall = budget (ref)")
    ax.set_xlabel("memory budget $B$ (fraction of entries)")
    ax.set_ylabel("memory recall")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig3_budget.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_horizon(rows: list[dict], out: Path) -> None:
    rows = sorted(rows, key=lambda r: float(r["buffer_horizon_s"]))
    xs = [float(r["buffer_horizon_s"]) for r in rows]
    ys = [float(r["capture_recall"]) for r in rows]
    delta = rows[0].get("decision_period_s", "?")
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(xs, ys, "-o", color="#1f77b4", label=f"lazy gate (period {delta}s)")
    ax.axhline(1.0, ls="--", color="#d62728", linewidth=1.4, label="ours (event-driven)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("passive buffer horizon $H$ (s)")
    ax.set_ylabel("memory recall")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig4_horizon.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render paper4 figures.")
    ap.add_argument("--eval-dir", type=Path, default=Path("out/eti_mot_eval"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    deadline = read_csv(args.eval_dir / "deadline_curve.csv")
    budget = read_csv(args.eval_dir / "pareto.csv")
    horizon = read_csv(args.eval_dir / "horizon_sweep.csv")

    fig_architecture(args.out_dir)
    fig_deadline_curve(deadline, args.out_dir)
    fig_budget(budget, args.out_dir)
    fig_horizon(horizon, args.out_dir)
    print(f"wrote 4 figures -> {args.out_dir}")
    n = json.loads((args.eval_dir / "summary.json").read_text()).get("n")
    print(f"(from {n} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
