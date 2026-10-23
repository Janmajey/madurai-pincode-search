import numpy as np
from sklearn.cluster import KMeans

from .exact import score_rows
from .stats import PartitionStats


class IVFIndex:
    def __init__(self, catalog, n_lists=256, seed=42, joint_stats=True):
        self.catalog = catalog
        self.n_lists = n_lists
        rows = np.flatnonzero(catalog.alive)

        km = KMeans(n_clusters=n_lists, n_init=1, max_iter=60, random_state=seed)
        km.fit(catalog.vectors[rows])
        centroids = km.cluster_centers_
        self.centroids = (centroids / np.linalg.norm(centroids, axis=1, keepdims=True)).astype(np.float32)

        labels = self.assign(catalog.vectors[rows])
        self.list_of = np.full(len(catalog.prices), -1, dtype=np.int64)
        self.list_of[rows] = labels
        self.lists = [rows[labels == p] for p in range(n_lists)]

        self.stats = PartitionStats(n_lists, catalog, joint=joint_stats)
        self.stats.update(labels, catalog.categories[rows], catalog.zones[rows], catalog.prices[rows])

    def assign(self, vectors):
        return np.argmax(vectors @ self.centroids.T, axis=1)

    def nearest_lists(self, query, n=None):
        scores = self.centroids @ query
        if n is None or n >= self.n_lists:
            return np.argsort(-scores)
        part = np.argpartition(-scores, n)[:n]
        return part[np.argsort(-scores[part])]

    def gather(self, lists):
        if len(lists) == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate([self.lists[p] for p in lists])

    def search(self, query, nprobe, k):
        rows = self.gather(self.nearest_lists(query, nprobe))
        return score_rows(self.catalog, rows, query, k)

    def add(self, rows):
        c = self.catalog
        if len(self.list_of) < len(c.prices):
            grow = np.full(len(c.prices) - len(self.list_of), -1, dtype=np.int64)
            self.list_of = np.concatenate([self.list_of, grow])
        labels = self.assign(c.vectors[rows])
        for p in np.unique(labels):
            self.lists[p] = np.concatenate([self.lists[p], rows[labels == p]])
        self.list_of[rows] = labels
        self.stats.update(labels, c.categories[rows], c.zones[rows], c.prices[rows], sign=1)

    def remove(self, rows):
        c = self.catalog
        labels = self.list_of[rows]
        for p in np.unique(labels):
            self.lists[p] = np.setdiff1d(self.lists[p], rows[labels == p], assume_unique=True)
        self.list_of[rows] = -1
        self.stats.update(labels, c.categories[rows], c.zones[rows], c.prices[rows], sign=-1)

    def list_sizes(self):
        return np.array([len(x) for x in self.lists])
