import numpy as np

from .exact import exact_search, score_rows


def prefilter(index, query, f, k=10, **_):
    return exact_search(index.catalog, query, f, k)


def postfilter(index, query, f, k=10, nprobe=16, fetch=100, **_):
    rows, scores = index.search(query, nprobe, fetch)
    keep = index.catalog.mask(f, rows)
    return rows[keep][:k], scores[keep][:k]


def adaptive_postfilter(index, query, f, k=10, selectivity=None, base_nprobe=16, **_):
    if selectivity is None:
        selectivity = index.stats.selectivity(f)
    nprobe = int(min(index.n_lists, np.ceil(base_nprobe / max(selectivity, 1e-9))))
    rows = index.gather(index.nearest_lists(query, nprobe))
    rows = rows[index.catalog.mask(f, rows)]
    return score_rows(index.catalog, rows, query, k)


def restricted(index, query, f, k=10, base_nprobe=16, min_matches=40, coverage=0.2, **_):
    expected = index.stats.partition_estimates(f)
    eligible = np.flatnonzero(expected > 0)
    if len(eligible) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    order = eligible[np.argsort(-(index.centroids[eligible] @ query))]
    need = max(min_matches, coverage * expected.sum())
    reached = np.searchsorted(np.cumsum(expected[order]), need) + 1
    lists = order[: max(base_nprobe, reached)]

    rows = index.gather(lists)
    rows = rows[index.catalog.mask(f, rows)]
    return score_rows(index.catalog, rows, query, k)


STRATEGIES = {
    "exact": prefilter,
    "adaptive_post": adaptive_postfilter,
    "restricted": restricted,
}

BASELINES = {
    "baseline_post": postfilter,
    "baseline_pre": prefilter,
}
