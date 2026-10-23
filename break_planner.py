import argparse

import numpy as np

from pincode_search import Catalog, Filter, IVFIndex, Planner, Thresholds
from pincode_search.bench import planner_fn, run, strategy
from pincode_search.workload import fixed_workload, price_percentiles, random_filter

CASE = Filter(category="Snacks & Branded Foods", max_price=200)
MADURAI = Filter(category="Ethnic Wear", zone="TN-SOUTH", max_price=1500)
PRICE_BANDS = [(None, 200), (None, 500), (None, 1000), (200, 1000), (500, 2000), (1000, None), (2000, None)]


def band_for(t, selectivity):
    for upper, name in t.ranges:
        if selectivity < upper:
            return name
    return t.ranges[-1][1]


def explain(catalog, index, t, f, label):
    planner = Planner(index, t)
    name, est = planner.choose(f)
    true = catalog.mask(f).sum() / len(catalog)
    best = band_for(t, true)
    queries = fixed_workload(catalog, [f], 30, seed=21)
    got = run(queries, planner_fn(index, t))
    ideal = run(queries, strategy(index, best, base_nprobe=t.base_nprobe, coverage=t.coverage))
    print(f"  {label:<22} estimate {est:7.3%}  true {true:7.3%}  picks {name:<13} "
          f"recall {got.recall:.3f}  p50 {got.p(50):.2f} ms   (right pick: {best}, p50 {ideal.p(50):.2f} ms)")


def scan(catalog, index, t):
    planner = Planner(index, t)
    total = len(catalog)
    wrong = []
    for cat in [None, *catalog.category_names]:
        for zone in [None, *catalog.zone_names]:
            for lo, hi in PRICE_BANDS:
                f = Filter(cat, zone, lo, hi)
                n = int(catalog.mask(f).sum())
                if n < 15:
                    continue
                name, est = planner.choose(f)
                if name != band_for(t, n / total):
                    wrong.append((f, est, n / total, name))
    return wrong


def random_sample(catalog, index, n=20, seed=5):
    rng = np.random.default_rng(seed)
    pct = price_percentiles(catalog)
    print(f"\n{n} random filters, estimate vs truth (joint counters)")
    shown = 0
    while shown < n:
        f = random_filter(catalog, rng, pct)
        true = catalog.mask(f).sum() / len(catalog)
        if true == 0:
            continue
        est = index.stats.selectivity(f)
        print(f"  {str(f):<62} estimate {est:7.3%}  true {true:7.3%}  error x{est / true:5.2f}")
        shown += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--thresholds", default="results/thresholds.json")
    args = parser.parse_args()

    catalog = Catalog.load(args.data)
    t = Thresholds.load(args.thresholds)
    naive = IVFIndex(catalog, joint_stats=False)
    fixed = IVFIndex(catalog, joint_stats=True)

    random_sample(catalog, fixed)

    for f in [CASE, MADURAI]:
        print(f"\n{f}")
        explain(catalog, naive, t, f, "independent counters")
        explain(catalog, fixed, t, f, "joint counters")

    for label, index in [("independent counters", naive), ("joint counters", fixed)]:
        wrong = scan(catalog, index, t)
        print(f"\n{label}: {len(wrong)} filters land in the wrong band")
        for f, est, true, name in wrong[:6]:
            print(f"  {str(f):<62} estimate {est:6.2%}  true {true:6.2%}  picks {name}")


if __name__ == "__main__":
    main()
