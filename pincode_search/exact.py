import numpy as np


def top_k(rows, scores, k):
    if len(rows) > k:
        part = np.argpartition(-scores, k)[:k]
        rows, scores = rows[part], scores[part]
    order = np.argsort(-scores, kind="stable")
    return rows[order], scores[order]


def score_rows(catalog, rows, query, k):
    return top_k(rows, catalog.vectors[rows] @ query, k)


def exact_search(catalog, query, f, k=10):
    mask = catalog.mask(f)
    rows = np.flatnonzero(mask)
    if len(rows) > len(mask) // 4:
        # gathering most of the matrix costs more than scoring all of it
        return top_k(rows, (catalog.vectors @ query)[rows], k)
    return score_rows(catalog, rows, query, k)
