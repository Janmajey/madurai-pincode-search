import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from pincode_search import Catalog, IVFIndex
from pincode_search.bench import calibrate, evaluate, planner_fn, strategy
from pincode_search.strategies import BASELINES
from pincode_search.workload import BUCKETS, build_workload, sweep_workload, unfiltered_workload, zone_workload

LABELS = {
    "exact": "exact scan",
    "adaptive_post": "adaptive post-filter",
    "restricted": "partition-restricted",
}


def methods_for(index, t):
    params = dict(base_nprobe=t.base_nprobe, min_matches=t.min_matches, coverage=t.coverage)
    return {
        "post-filter (baseline)": lambda v, f: BASELINES["baseline_post"](index, v, f, nprobe=t.base_nprobe),
        "pre-filter (baseline)": lambda v, f: BASELINES["baseline_pre"](index, v, f),
        "exact scan": strategy(index, "exact", **params),
        "adaptive post-filter": strategy(index, "adaptive_post", **params),
        "partition-restricted": strategy(index, "restricted", **params),
        "planner": planner_fn(index, t),
    }


def rule_lines(t):
    lines, lo = [], 0.0
    for hi, name in t.ranges:
        lines.append(f"  - {lo:.2%} to {min(hi, 1):.2%}: {LABELS[name]}")
        lo = hi
    return lines


def bucket_table(results_by_bucket):
    lines = ["| Bucket | Method | Recall@10 | p50 ms | p95 ms |", "| --- | --- | --- | --- | --- |"]
    for bucket, results in results_by_bucket.items():
        for name, r in results.items():
            lines.append(f"| {bucket} | {name} | {r.recall:.3f} | {r.p(50):.2f} | {r.p(95):.2f} |")
    return lines


def choices_table(results_by_bucket):
    lines = ["| Bucket | exact scan | adaptive post-filter | partition-restricted |", "| --- | --- | --- | --- |"]
    for bucket, results in results_by_bucket.items():
        c = Counter(results["planner"].choices)
        lines.append(f"| {bucket} | {c['exact']} | {c['adaptive_post']} | {c['restricted']} |")
    return lines


def crossover_table(bins):
    head = "| Selectivity | Queries | " + " | ".join(f"{LABELS[n]} p50 ms (recall)" for n in LABELS) + " | Pick (fastest with recall 0.97 or more, a tie within 15% goes to better recall) |"
    lines = [head, "| --- " * (3 + len(LABELS)) + "|"]
    for b in bins:
        cells = " | ".join(f"{b['stats'][n][1]:.2f} ({b['stats'][n][0]:.2f})" for n in LABELS)
        lines.append(f"| {b['lo']:.2%} to {min(b['hi'], 1):.2%} | {b['n']} | {cells} | {LABELS[b['winner']]} |")
    return lines


def zone_table(catalog, index, t, queries):
    methods = methods_for(index, t)
    picked = {"post-filter (baseline)": methods["post-filter (baseline)"], "planner": methods["planner"]}
    per_zone = {}
    for zone in catalog.zone_names:
        qs = [q for q in queries if q.filter.zone == zone]
        if qs:
            per_zone[zone] = (qs[0].count, evaluate(index, qs, picked))

    sizes = np.array([c for c, _ in per_zone.values()], dtype=float)
    lines = ["| Method | Traffic-weighted recall | Plain average | Worst zone (items) | Worst recall |", "| --- | --- | --- | --- | --- |"]
    for name in picked:
        recalls = np.array([res[name].recall for _, res in per_zone.values()])
        worst = int(np.argmin(recalls))
        zone = list(per_zone)[worst]
        lines.append(
            f"| {name} | {np.average(recalls, weights=sizes):.3f} | {recalls.mean():.3f} | "
            f"{zone} ({per_zone[zone][0]}) | {recalls[worst]:.3f} |"
        )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--per-bucket", type=int, default=50)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    catalog = Catalog.load(args.data)
    print(f"catalog: {len(catalog)} items, {catalog.vectors.shape[1]} dims")
    index = IVFIndex(catalog)

    print("calibrating planner...")
    t, calib = calibrate(index, sweep_workload(catalog, n=1500, seed=1), unfiltered_workload(catalog, n=100, seed=3))
    t.save(out / "thresholds.json")

    print("running benchmark...")
    queries = build_workload(catalog, per_bucket=args.per_bucket, seed=2)
    methods = methods_for(index, t)
    by_bucket = {b: evaluate(index, [q for q in queries if q.bucket == b], methods) for b in BUCKETS}

    report = [
        "# Benchmark results",
        "",
        f"Catalog: {len(catalog)} items, {catalog.vectors.shape[1]} dims, {index.n_lists} partitions.",
        "",
        "## Planner settings (measured)",
        "",
        f"- base nprobe: {t.base_nprobe} (smallest nprobe with unfiltered recall ≥ 0.97)",
        f"- partition-restricted coverage: {t.coverage:.0%} of the expected matches (smallest value with recall ≥ 0.97 in every band from 2% to 50%)",
        "- decision rule (estimated selectivity → strategy), fastest strategy with recall ≥ 0.97 in each band:",
        *rule_lines(t),
        "",
        "## nprobe sweep (no filter)",
        "",
        "| nprobe | Recall@10 | p50 ms |",
        "| --- | --- | --- |",
        *[f"| {n} | {r:.3f} | {ms:.2f} |" for n, r, ms in calib["nprobe"]],
        "",
        "## Coverage sweep for partition-restricted search (2% to 50% selectivity)",
        "",
        "| Coverage | Worst band recall | Mean p50 ms |",
        "| --- | --- | --- |",
        *[f"| {c:.0%} | {r:.3f} | {ms:.2f} |" for c, r, ms in calib["coverage"]],
        "",
        "## Crossover (calibration set, bucketed by true selectivity)",
        "",
        *crossover_table(calib["bins"]),
        "",
        "## Results by selectivity bucket",
        "",
        *bucket_table(by_bucket),
        "",
        "## What the planner picked",
        "",
        *choices_table(by_bucket),
        "",
        "## Per-zone recall (zone-only filters, 20 queries per zone)",
        "",
        *zone_table(catalog, index, t, zone_workload(catalog)),
    ]
    text = "\n".join(report) + "\n"
    (out / "benchmark.md").write_text(text, encoding="utf-8")
    print()
    print(text)


if __name__ == "__main__":
    main()
