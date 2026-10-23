import numpy as np

PRICE_EDGES = np.concatenate([[0], np.geomspace(20, 1_000_000, 48)])


def price_bins(prices):
    return np.clip(np.searchsorted(PRICE_EDGES, prices, side="right") - 1, 0, len(PRICE_EDGES) - 2)


def price_weights(lo, hi):
    lo = 0 if lo is None else lo
    hi = np.inf if hi is None else hi + 1
    left, right = PRICE_EDGES[:-1], PRICE_EDGES[1:]
    overlap = np.clip(np.minimum(right, hi) - np.maximum(left, lo), 0, None)
    return overlap / (right - left)


def _grow(counts, cols):
    if counts.shape[-1] >= cols:
        return counts
    pad = [(0, 0)] * (counts.ndim - 1) + [(0, cols - counts.shape[-1])]
    return np.pad(counts, pad)


class PartitionStats:
    """Counters per partition, kept in sync with the index on every add and delete."""

    def __init__(self, n_lists, catalog, joint=False):
        self.n_lists = n_lists
        self.catalog = catalog
        self.joint = joint
        self.sizes = np.zeros(n_lists, dtype=np.int64)
        self.category_counts = np.zeros((n_lists, 0), dtype=np.int64)
        self.zone_counts = np.zeros((n_lists, 0), dtype=np.int64)
        self.price_counts = np.zeros((n_lists, len(PRICE_EDGES) - 1), dtype=np.int64)
        self.pair_counts = np.zeros((0, 0), dtype=np.int64)
        self.category_price_counts = np.zeros((0, len(PRICE_EDGES) - 1), dtype=np.int64)

    def update(self, lists, categories, zones, prices, sign=1):
        n_cat = len(self.catalog.category_names)
        n_zone = len(self.catalog.zone_names)
        self.category_counts = _grow(self.category_counts, n_cat)
        self.zone_counts = _grow(self.zone_counts, n_zone)
        self.pair_counts = _grow(_grow(self.pair_counts, n_zone).T, n_cat).T
        self.category_price_counts = _grow(self.category_price_counts.T, n_cat).T

        np.add.at(self.sizes, lists, sign)
        np.add.at(self.category_counts, (lists, categories), sign)
        np.add.at(self.zone_counts, (lists, zones), sign)
        bins = price_bins(prices)
        np.add.at(self.price_counts, (lists, bins), sign)
        np.add.at(self.pair_counts, (categories, zones), sign)
        np.add.at(self.category_price_counts, (categories, bins), sign)

    @property
    def total(self):
        return int(self.sizes.sum())

    def _codes(self, f):
        cat = self.catalog.category_codes.get(f.category) if f.category is not None else None
        zone = self.catalog.zone_codes.get(f.zone) if f.zone is not None else None
        return cat, zone

    def selectivity(self, f):
        total = max(self.total, 1)
        cat, zone = self._codes(f)
        if cat == -1 or zone == -1:
            return 0.0

        frac = 1.0
        if self.joint and cat is not None and zone is not None:
            frac *= self.pair_counts[cat, zone] / total
        else:
            if cat is not None:
                frac *= self.category_counts[:, cat].sum() / total
            if zone is not None:
                frac *= self.zone_counts[:, zone].sum() / total
        if f.min_price is not None or f.max_price is not None:
            w = price_weights(f.min_price, f.max_price)
            if self.joint and cat is not None:
                in_cat = max(self.category_price_counts[cat].sum(), 1)
                frac *= self.category_price_counts[cat] @ w / in_cat
            else:
                frac *= self.price_counts.sum(axis=0) @ w / total
        return float(frac)

    def partition_estimates(self, f):
        cat, zone = self._codes(f)
        if cat == -1 or zone == -1:
            return np.zeros(self.n_lists)

        sizes = np.maximum(self.sizes, 1)
        est = self.sizes.astype(np.float64)
        if cat is not None:
            est = est * self.category_counts[:, cat] / sizes
        if zone is not None:
            est = est * self.zone_counts[:, zone] / sizes
        if f.min_price is not None or f.max_price is not None:
            est = est * (self.price_counts @ price_weights(f.min_price, f.max_price)) / sizes
        return est
