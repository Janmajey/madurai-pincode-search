import numpy as np
import pandas as pd
import pytest

from pincode_search import Catalog, IVFIndex


def make_catalog(n=3000, d=32, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(12, d))
    labels = rng.integers(0, 12, n)
    vectors = centers[labels] + 0.35 * rng.normal(size=(n, d))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    items = pd.DataFrame({
        "title": [f"item {i}" for i in range(n)],
        "category": [f"cat{label % 6}" for label in labels],
        "zone": rng.choice(["big", "mid", "tiny"], n, p=[0.8, 0.19, 0.01]),
        "price": rng.integers(50, 5000, n),
    })
    return Catalog(vectors, items)


def make_query(catalog, rng, noise=0.3):
    v = catalog.vectors[rng.integers(len(catalog.prices))]
    v = v + noise / np.sqrt(len(v)) * rng.normal(size=len(v))
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture(scope="module")
def catalog():
    return make_catalog()


@pytest.fixture(scope="module")
def index(catalog):
    return IVFIndex(catalog, n_lists=32, seed=0)
