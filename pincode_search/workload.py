from dataclasses import dataclass

import numpy as np

from .catalog import Filter
from .exact import exact_search

BUCKETS = ["~100%", "~10%", "~1%", "~0.1%", "handful"]


def bucket_of(count, total):
    s = count / total
    if count == 0:
        return None
    if count <= 15:
        return "handful"
    if 0.0005 <= s < 0.002:
        return "~0.1%"
    if 0.005 <= s < 0.02:
        return "~1%"
    if 0.05 <= s < 0.2:
        return "~10%"
    if s >= 0.5:
        return "~100%"
    return None


@dataclass
class Query:
    vector: np.ndarray
    filter: Filter
    count: int
    bucket: str | None
    true_rows: np.ndarray
    true_scores: np.ndarray


def price_percentiles(catalog):
    return np.percentile(catalog.prices[catalog.alive], np.arange(101)).astype(int)


def random_filter(catalog, rng, percentiles):
    use = rng.random(3) < [0.6, 0.6, 0.5]
    category = str(rng.choice(catalog.category_names)) if use[0] else None
    zone = str(rng.choice(catalog.zone_names)) if use[1] else None
    lo = hi = None
    if use[2]:
        a, b = np.sort(rng.integers(0, 101, 2))
        lo = int(percentiles[a]) if a > 15 else None
        hi = int(percentiles[b]) if b < 85 else None
    return Filter(category, zone, lo, hi)


def make_query_vector(catalog, f, rng, noise=0.6):
    pool = catalog.alive
    if f.category is not None:
        pool = pool & (catalog.categories == catalog.category_codes.get(f.category))
    pool = np.flatnonzero(pool)
    v = catalog.vectors[rng.choice(pool)]
    v = v + rng.normal(0, noise / np.sqrt(len(v)), len(v)).astype(np.float32)
    return v / np.linalg.norm(v)


def build_workload(catalog, per_bucket=50, seed=0, k=10, candidates=6000):
    rng = np.random.default_rng(seed)
    total = len(catalog)
    picked = {b: [] for b in BUCKETS}
    seen = set()

    pct = price_percentiles(catalog)
    filters = [Filter()] + [random_filter(catalog, rng, pct) for _ in range(candidates)]
    for f in filters:
        if f in seen:
            continue
        seen.add(f)
        count = int(catalog.mask(f).sum())
        b = bucket_of(count, total)
        if b is not None and len(picked[b]) < per_bucket * 3:
            picked[b].append((f, count))

    queries = []
    for b in BUCKETS:
        pool = picked[b]
        reps = int(np.ceil(per_bucket / max(len(pool), 1)))
        chosen = [pool[i] for i in rng.permutation(len(pool))] * reps
        for f, count in chosen[:per_bucket]:
            q = make_query_vector(catalog, f, rng)
            rows, scores = exact_search(catalog, q, f, k)
            queries.append(Query(q, f, count, b, rows, scores))
    return queries


def sweep_workload(catalog, n=400, seed=1, k=10):
    """Filters spread over the whole selectivity range, used to find planner thresholds."""
    rng = np.random.default_rng(seed)
    pct = price_percentiles(catalog)
    queries = []
    while len(queries) < n:
        f = random_filter(catalog, rng, pct)
        count = int(catalog.mask(f).sum())
        if count == 0:
            continue
        q = make_query_vector(catalog, f, rng)
        rows, scores = exact_search(catalog, q, f, k)
        queries.append(Query(q, f, count, bucket_of(count, len(catalog)), rows, scores))
    return queries


def fixed_workload(catalog, filters, per_filter, seed, k=10):
    rng = np.random.default_rng(seed)
    total = len(catalog)
    queries = []
    for f in filters:
        count = int(catalog.mask(f).sum())
        if count == 0:
            continue
        for _ in range(per_filter):
            q = make_query_vector(catalog, f, rng)
            rows, scores = exact_search(catalog, q, f, k)
            queries.append(Query(q, f, count, bucket_of(count, total), rows, scores))
    return queries


def unfiltered_workload(catalog, n=100, seed=3):
    return fixed_workload(catalog, [Filter()], n, seed)


def zone_workload(catalog, per_zone=20, seed=4):
    return fixed_workload(catalog, [Filter(zone=z) for z in catalog.zone_names], per_zone, seed)
