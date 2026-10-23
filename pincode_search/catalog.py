from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Filter:
    category: str | None = None
    zone: str | None = None
    min_price: int | None = None
    max_price: int | None = None

    def __str__(self):
        parts = []
        if self.category:
            parts.append(f"category={self.category}")
        if self.zone:
            parts.append(f"zone={self.zone}")
        if self.min_price is not None or self.max_price is not None:
            lo = self.min_price or 0
            hi = "inf" if self.max_price is None else self.max_price
            parts.append(f"price={lo}-{hi}")
        return " AND ".join(parts) or "no filter"


class Codes:
    def __init__(self, values):
        self.names = list(dict.fromkeys(values))
        self.index = {n: i for i, n in enumerate(self.names)}

    def encode(self, values):
        for v in values:
            if v not in self.index:
                self.index[v] = len(self.names)
                self.names.append(v)
        return np.array([self.index[v] for v in values], dtype=np.int32)

    def get(self, name):
        return self.index.get(name, -1)


class Catalog:
    def __init__(self, vectors, items):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.titles = items["title"].to_numpy(dtype=object)
        self.category_codes = Codes(sorted(items["category"].unique()))
        self.zone_codes = Codes(sorted(items["zone"].unique()))
        self.categories = self.category_codes.encode(items["category"])
        self.zones = self.zone_codes.encode(items["zone"])
        self.prices = items["price"].to_numpy(dtype=np.int64)
        self.alive = np.ones(len(self.prices), dtype=bool)

    @classmethod
    def load(cls, data_dir="data"):
        data_dir = Path(data_dir)
        return cls(np.load(data_dir / "vectors.npy"), pd.read_csv(data_dir / "items.csv"))

    def __len__(self):
        return int(self.alive.sum())

    @property
    def category_names(self):
        return self.category_codes.names

    @property
    def zone_names(self):
        return self.zone_codes.names

    def mask(self, f, rows=None):
        if rows is None:
            rows = slice(None)
        m = self.alive[rows].copy()
        if f.category is not None:
            m &= self.categories[rows] == self.category_codes.get(f.category)
        if f.zone is not None:
            m &= self.zones[rows] == self.zone_codes.get(f.zone)
        if f.min_price is not None:
            m &= self.prices[rows] >= f.min_price
        if f.max_price is not None:
            m &= self.prices[rows] <= f.max_price
        return m

    def matching_rows(self, f):
        return np.flatnonzero(self.mask(f))

    def append(self, vectors, titles, categories, zones, prices):
        start = len(self.prices)
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self.titles = np.concatenate([self.titles, np.asarray(titles, dtype=object)])
        self.categories = np.concatenate([self.categories, self.category_codes.encode(categories)])
        self.zones = np.concatenate([self.zones, self.zone_codes.encode(zones)])
        self.prices = np.concatenate([self.prices, np.asarray(prices, dtype=np.int64)])
        self.alive = np.concatenate([self.alive, np.ones(len(prices), dtype=bool)])
        return np.arange(start, len(self.prices))

    def delete(self, rows):
        self.alive[rows] = False
