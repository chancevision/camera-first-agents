# camera-first-agents — code

Reference code for **How Should a Camera-First AI Agent Work?**
([arXiv:2610.14772](https://arxiv.org/abs/2610.14772)). MIT licensed — see [`LICENSE`](LICENSE).

The results in the paper are a **labeled scheduler simulation**: policies are evaluated over real
track timelines with oracle perception (per-track windows are given, no pixels, no detector, no VLM).

| Script | What it does |
|---|---|
| [`build_eti_benchmark.py`](build_eti_benchmark.py) | builds the Ephemeral-Target Interaction (ETI) benchmark: episodes carrying `{t_in, t_out, target_id, query, answer_label, source_dataset, synthetic, event_class}`. `--mock` generates a synthetic set with no downloads. |
| [`run_eti_eval.py`](run_eti_eval.py) | runs the capture policies (`monolithic`, `fixed_sampling`, `saliency_gate`, `remember` under a per-sequence budget, `oracle`) and emits `summary.json`, `deadline_curve.csv`, `pareto.csv`, `horizon_sweep.csv`. |
| [`make_figures.py`](make_figures.py) | regenerates the four figures from the eval artifacts. |
| [`fetch_mot17.py`](fetch_mot17.py) | downloads **only** MOT17 `gt.txt` (track lifetimes, not pixels) from a public mirror. |
| [`make_attrs.py`](make_attrs.py) | builds the attribute/query table for the LaSOT (secondary) source; `--mock` emits a synthetic one. |
| [`fetch_openrouter_latency.py`](fetch_openrouter_latency.py) | reads OpenRouter's published per-endpoint latency/throughput stats. |
| [`measure_vlm_latency.py`](measure_vlm_latency.py) | measures time-to-first-token and total latency through OpenRouter. Requires `OPENROUTER_API_KEY` in the environment. |
| [`configs/eti_mot.json`](configs/eti_mot.json) | the MOT17 benchmark config (`fps`, `max_dwell`, output path). |

End-to-end, no downloads, no API key:

```bash
python build_eti_benchmark.py --mock --out out/eti_mock
python run_eti_eval.py --episodes out/eti_mock/episodes.jsonl --out out/eti_mock_eval
```

With real track lifetimes:

```bash
python fetch_mot17.py --dest data/mot17
python build_eti_benchmark.py --config configs/eti_mot.json --mot-root data/mot17 --out out/eti_mot
python run_eti_eval.py --episodes out/eti_mot/episodes.jsonl --out out/eti_mot_eval
python -m pip install -r requirements.txt   # matplotlib, numpy — figures only
python make_figures.py --eval-dir out/eti_mot_eval --out-dir figures
```

Metrics are budget-conditioned: capture recall is reported against the deadline curve (recall vs
dwell) under a perceptual budget *B*, never unbounded. There is no MOTA/HOTA/SUC/AUC here — this
tests capture **scheduling**, not detection.

## Data

MOT17 and LaSOT (the paper's secondary source) are third-party datasets: they are **not** mirrored in
this repository, only fetched from their public sources (`fetch_mot17.py`). No pixels are needed or
used — the benchmark consumes track windows.
