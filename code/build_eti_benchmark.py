#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Chance AI
"""Build the Ephemeral-Target Interaction (ETI) benchmark.

Emits episodes carrying an explicit query per episode:
    {t_in, t_out, target_id, query, answer_label, source_dataset, synthetic, event_class}

Public-data only. ``--mock`` generates a synthetic set with no downloads.
All capture metrics are budget-conditioned: metrics are computed as functions
of a perceptual budget B (working-memory horizon x compute), never unbounded.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

TAU_CHOICES = [0.5, 1.0, 2.0, 4.0]
EVENT_CLASSES = ["ephemeral-target", "hazard", "known-entity", "goal-relevant-target"]
ATTR_VOCAB = [
    "round_neck", "v_neck", "denim", "leather", "striped", "plain",
    "red", "blue", "black", "white",
]


@dataclass
class Episode:
    t_in: float
    t_out: float
    target_id: str
    query: str
    answer_label: str
    source_dataset: str
    synthetic: bool
    event_class: str = "ephemeral-target"
    salience: float | None = None

    @property
    def dwell(self) -> float:
        return self.t_out - self.t_in


def intervals_from_visibility(frames: list[tuple[int, bool]], fps: float) -> list[tuple[float, float]]:
    """Return [(t_in, t_out)] windows where the target is visible."""
    out: list[tuple[float, float]] = []
    start = None
    last_idx = 0
    for idx, visible in frames:
        last_idx = idx
        t = idx / fps
        if visible and start is None:
            start = t
        elif not visible and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, last_idx / fps))
    return out


def _read_frame_flags(path: Path | None) -> set[int]:
    """Read a LaSOT-style '*_full_occ_*.txt' / '*_out_of_view_*.txt' range file."""
    if path is None or not path.exists():
        return set()
    frames: set[int] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        try:
            a, b = int(parts[0]), int(parts[1])
        except (IndexError, ValueError):
            continue
        frames.update(range(a, b + 1))
    return frames


def load_lasot_sequences(lasot_root: Path, fps: float = 30.0):
    """Yield (seq_id, query, intervals) from a LaSOT root directory.

    Expects per-sequence folders containing ``groundtruth.txt`` (x,y,w,h per
    frame; non-positive coordinates treated as not visible) and, when present,
    ``*_full_occ_*.txt`` / ``*_out_of_view_*.txt`` frame-index ranges.
    """
    for gt in sorted(lasot_root.rglob("groundtruth.txt")):
        seq_dir = gt.parent
        occ = _read_frame_flags(next(iter(seq_dir.glob("*full_occ*.txt")), None))
        oov = _read_frame_flags(next(iter(seq_dir.glob("*out_of_view*.txt")), None))
        frames: list[tuple[int, bool]] = []
        for i, line in enumerate(gt.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.replace(",", " ").split()]
            x, y, w, h = vals[0], vals[1], vals[2], vals[3]
            visible = (w > 0 and h > 0 and x >= 0 and y >= 0) and i not in occ and i not in oov
            frames.append((i, visible))
        query = f"describe the target in {seq_dir.name}"
        yield seq_dir.name, query, intervals_from_visibility(frames, fps)


def load_attributes(path: Path) -> dict[str, dict[str, str]]:
    """CSV with columns: target_id, query, answer_label."""
    out: dict[str, dict[str, str]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            out[row["target_id"]] = {"query": row["query"], "answer_label": row["answer_label"]}
    return out


def build_real(lasot_root: Path, attrs_path: Path, fps: float) -> list[Episode]:
    attrs = load_attributes(attrs_path)
    episodes: list[Episode] = []
    for seq_id, query, intervals in load_lasot_sequences(lasot_root, fps):
        meta = attrs.get(seq_id, {"query": query, "answer_label": "unknown"})
        for k, (t_in, t_out) in enumerate(intervals):
            episodes.append(Episode(
                t_in=round(t_in, 3), t_out=round(t_out, 3),
                target_id=f"{seq_id}#{k}", query=meta["query"],
                answer_label=meta["answer_label"],
                source_dataset="lasot", synthetic=False,
            ))
    return episodes


def load_mot_sequences(mot_root: Path, fps: float = 30.0, min_vis: float = 0.0):
    """Yield (seq_id, class_name, [(t_in, t_out, track_id)]) from MOT-format gt.txt.

    Expects <root>/<seq>/gt/gt.txt with lines frame,id,x,y,w,h,conf,class,vis.
    A track's interval is [first_present_frame, last_present_frame + 1]; enter and
    exit are the hard boundaries that make the target ephemeral.
    """
    for gt in sorted(mot_root.rglob("gt.txt")):
        seq_id = gt.parent.parent.name
        frames_by_id: dict[int, list[int]] = defaultdict(list)
        area_by_id: dict[int, float] = defaultdict(float)
        count_by_id: dict[int, int] = defaultdict(int)
        for line in gt.read_text().splitlines():
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                frame = int(float(parts[0]))
                tid = int(float(parts[1]))
                w = float(parts[4])
                h = float(parts[5])
                vis = float(parts[8]) if len(parts) > 8 else 1.0
            except ValueError:
                continue
            if w <= 0 or h <= 0 or vis < min_vis:
                continue
            frames_by_id[tid].append(frame)
            area_by_id[tid] += w * h
            count_by_id[tid] += 1
        intervals = []
        for tid, frames in frames_by_id.items():
            frames.sort()
            intervals.append((frames[0] / fps, (frames[-1] + 1) / fps, tid, area_by_id[tid] / count_by_id[tid]))
        yield seq_id, "person", intervals


def build_mot(mot_root: Path, attrs_path: Path | None, fps: float, query_template: str) -> list[Episode]:
    attrs = load_attributes(attrs_path) if attrs_path else {}
    episodes: list[Episode] = []
    for seq_id, cls, intervals in load_mot_sequences(mot_root, fps):
        for t_in, t_out, tid, area in intervals:
            key = f"{seq_id}#{tid}"
            meta = attrs.get(key, {})
            episodes.append(Episode(
                t_in=round(t_in, 3), t_out=round(t_out, 3), target_id=key,
                query=meta.get("query", f"{query_template} ({cls})"),
                answer_label=meta.get("answer_label", "unknown"),
                source_dataset="mot", synthetic=False,
                salience=round(area, 2),
            ))
    return episodes


def build_mock(n: int, seed: int = 42) -> list[Episode]:
    rng = random.Random(seed)
    episodes: list[Episode] = []
    for i in range(n):
        tau = rng.choice(TAU_CHOICES)
        t_in = round(rng.uniform(0, 60), 3)
        episodes.append(Episode(
            t_in=t_in, t_out=round(t_in + tau, 3), target_id=f"mock{i:05d}",
            query="what attribute?", answer_label=rng.choice(ATTR_VOCAB),
            source_dataset="mock", synthetic=True,
            event_class=rng.choice(EVENT_CLASSES),
        ))
    return episodes


def compute_metrics(
    episodes: list[Episode],
    consolidated_within: dict[str, bool],
    ttc: dict[str, float | None],
    answer_correct: dict[str, bool],
    n_false_capture: int = 0,
    n_capture_events: int = 0,
) -> dict:
    """Budget-conditioned ETI metrics for one (policy, budget) setting."""
    n = len(episodes) or 1
    captured = [consolidated_within.get(e.target_id, False) for e in episodes]
    wmr = 1.0 - sum(captured) / n
    captured_eps = [e for e in episodes if consolidated_within.get(e.target_id)]
    ttc_vals = [v for v in (ttc.get(e.target_id) for e in captured_eps) if v is not None]
    daa_cond = None
    if captured_eps:
        daa_cond = sum(bool(answer_correct.get(e.target_id)) for e in captured_eps) / len(captured_eps)
    joint = sum(bool(answer_correct.get(e.target_id)) for e in captured_eps) / n
    return {
        "n": len(episodes),
        "window_miss_rate": round(wmr, 4),
        "consolidation_rate": round(sum(captured) / n, 4),
        "time_to_consolidate_mean": round(sum(ttc_vals) / len(ttc_vals), 4) if ttc_vals else None,
        "deferred_answer_acc_given_consolidated": round(daa_cond, 4) if daa_cond is not None else None,
        "joint_success": round(joint, 4),
        "false_capture_rate": round(n_false_capture / n_capture_events, 4) if n_capture_events else None,
    }


def demo_policies(episodes: list[Episode]) -> dict[str, dict]:
    """Two illustrative policies: monolithic (never captures in time) and oracle."""
    monolithic = {e.target_id: False for e in episodes}
    oracle = {e.target_id: True for e in episodes}
    ttc_oracle = {e.target_id: e.dwell * 0.5 for e in episodes}
    ans = {e.target_id: True for e in episodes}
    return {
        "monolithic": compute_metrics(episodes, monolithic, {}, {}),
        "oracle": compute_metrics(episodes, oracle, ttc_oracle, ans),
    }


def write_episodes(episodes: list[Episode], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "episodes.jsonl"
    with path.open("w") as fh:
        for e in episodes:
            fh.write(json.dumps(asdict(e)) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the ETI benchmark.")
    ap.add_argument("--mock", action="store_true", help="generate synthetic episodes, no download")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lasot-root", type=Path)
    ap.add_argument("--mot-root", type=Path, help="dir with <seq>/gt/gt.txt (MOT enter/exit)")
    ap.add_argument("--attrs", type=Path, help="CSV: target_id,query,answer_label")
    ap.add_argument("--query-template", default="identify the target", help="injected query for enter/exit sources")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-dwell", type=float, help="keep only episodes with dwell <= this many seconds")
    ap.add_argument("--out", type=Path, default=Path("out/eti"))
    ap.add_argument("--config", type=Path, help="JSON config (keys mirror the flags)")
    args = ap.parse_args()

    if args.config:
        cfg = json.loads(args.config.read_text())
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    if args.mot_root:
        episodes = build_mot(args.mot_root, args.attrs, args.fps, args.query_template)
    elif args.mock or not args.lasot_root:
        episodes = build_mock(args.n, args.seed)
    else:
        episodes = build_real(args.lasot_root, args.attrs, args.fps)

    total = len(episodes)
    if args.max_dwell:
        episodes = [e for e in episodes if e.dwell <= args.max_dwell]
    dwells = sorted(e.dwell for e in episodes)
    if dwells:
        print(f"dwell s: min={dwells[0]:.2f} median={statistics.median(dwells):.2f} max={dwells[-1]:.2f}")
    if args.max_dwell:
        print(f"kept {len(episodes)}/{total} episodes with dwell <= {args.max_dwell}s")

    path = write_episodes(episodes, args.out)
    print(f"wrote {len(episodes)} episodes -> {path}")
    print(json.dumps(demo_policies(episodes), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
