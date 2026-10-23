import time
from dataclasses import dataclass, field

import numpy as np

from .metrics import percentile_ms, recall_at_k
from .planner import Planner, Thresholds
from .strategies import STRATEGIES

SELECTIVITY_EDGES = [0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.01]
RECALL_TARGET = 0.97


@dataclass
class Run:
    recalls: list = field(default_factory=list)
    seconds: list = field(default_factory=list)
    choices: list = field(default_factory=list)

    @property
    def recall(self):
        return float(np.mean(self.recalls)) if self.recalls else float("nan")

    def p(self, q):
        return percentile_ms(self.seconds, q) if self.seconds else float("nan")


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def run(queries, fn, **kwargs):
    for q in queries[:5]:
        fn(q.vector, q.filter, **kwargs)
    result = Run()
    for q in queries:
        out, secs = timed(fn, q.vector, q.filter, **kwargs)
        result.recalls.append(recall_at_k(q.true_scores, out[1]))
        result.seconds.append(secs)
        if len(out) > 2:
            result.choices.append(out[2])
    return result


def run_side_by_side(queries, fns, repeats=3):
    """Times every strategy on the same query back to back, so machine noise hits them equally."""
    for q in queries[:5]:
        for fn in fns.values():
            fn(q.vector, q.filter)
    results = {name: Run() for name in fns}
    for q in queries:
        best = {name: float("inf") for name in fns}
        for _ in range(repeats):
            for name, fn in fns.items():
                out, secs = timed(fn, q.vector, q.filter)
                best[name] = min(best[name], secs)
        for name, fn in fns.items():
            results[name].recalls.append(recall_at_k(q.true_scores, fn(q.vector, q.filter)[1]))
            results[name].seconds.append(best[name])
    return results


def strategy(index, name, **params):
    return lambda v, f, **kw: STRATEGIES[name](index, v, f, **params, **kw)


def pick_nprobe(index, queries, target=0.97):
    rows = []
    best = None
    for nprobe in [1, 2, 4, 8, 16, 32, 64, 128]:
        r = run(queries, lambda v, f: index.search(v, nprobe, 10))
        rows.append((nprobe, r.recall, r.p(50)))
        if best is None and r.recall >= target:
            best = nprobe
    return best or 128, rows


def pick_coverage(index, queries, base_nprobe, target=RECALL_TARGET):
    total = len(index.catalog)
    bands = [(0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5)]
    rows = []
    best = None
    for cov in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
        fn = strategy(index, "restricted", base_nprobe=base_nprobe, coverage=cov)
        worst, ms = 1.0, []
        for lo, hi in bands:
            qs = [q for q in queries if lo <= q.count / total < hi]
            if qs:
                r = run(qs, fn)
                worst = min(worst, r.recall)
                ms.append(r.p(50))
        rows.append((cov, worst, float(np.mean(ms))))
        if best is None and worst >= target:
            best = cov
    return best or 0.4, rows


def selectivity_bins(index, queries, base_nprobe, coverage, min_queries=20):
    total = len(index.catalog)
    bins = []
    for lo, hi in zip(SELECTIVITY_EDGES[:-1], SELECTIVITY_EDGES[1:]):
        qs = [q for q in queries if lo <= q.count / total < hi]
        if len(qs) < min_queries:
            continue
        fns = {name: strategy(index, name, base_nprobe=base_nprobe, coverage=coverage) for name in STRATEGIES}
        runs = run_side_by_side(qs, fns, repeats=3)
        stats = {name: (r.recall, r.p(50)) for name, r in runs.items()}
        bins.append({"lo": lo, "hi": hi, "n": len(qs), "stats": stats, "winner": pick_winner(stats)})
    return bins


def pick_winner(stats, tie=0.15):
    ok = {n: s for n, s in stats.items() if s[0] >= RECALL_TARGET}
    if not ok:
        return max(stats, key=lambda n: stats[n][0])
    fastest = min(s[1] for s in ok.values())
    close = {n: s for n, s in ok.items() if s[1] <= fastest * (1 + tie)}
    return max(close, key=lambda n: (close[n][0], -close[n][1]))


def ranges_from_bins(bins):
    winners = [b["winner"] for b in bins]
    for i in range(1, len(winners) - 1):
        if winners[i - 1] == winners[i + 1] != winners[i]:
            winners[i] = winners[i - 1]

    ranges = []
    for b, winner in zip(bins, winners):
        if ranges and ranges[-1][1] == winner:
            ranges[-1][0] = b["hi"]
        else:
            ranges.append([b["hi"], winner])
    ranges[-1][0] = 1.01
    return ranges


def calibrate(index, sweep, unfiltered):
    base_nprobe, nprobe_rows = pick_nprobe(index, unfiltered)
    coverage, coverage_rows = pick_coverage(index, sweep, base_nprobe)
    bins = selectivity_bins(index, sweep, base_nprobe, coverage)
    t = Thresholds(ranges=ranges_from_bins(bins), base_nprobe=base_nprobe, coverage=coverage)
    return t, {"nprobe": nprobe_rows, "coverage": coverage_rows, "bins": bins}


def evaluate(index, queries, methods):
    return {name: run(queries, fn) for name, fn in methods.items()}


def planner_fn(index, thresholds):
    planner = Planner(index, thresholds)
    return lambda v, f, k=10: planner.search(v, f, k)
