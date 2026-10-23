import numpy as np
import pytest

from pincode_search import Filter, Planner, Thresholds
from pincode_search.exact import exact_search
from pincode_search.metrics import recall_at_k
from pincode_search.strategies import postfilter

from conftest import make_query


def test_item_finds_itself_first(catalog):
    rows, _ = exact_search(catalog, catalog.vectors[123], Filter())
    assert rows[0] == 123


def test_exact_results_respect_every_filter(catalog):
    f = Filter(category="cat2", zone="mid", min_price=300, max_price=2000)
    rows, _ = exact_search(catalog, catalog.vectors[0], f, k=50)
    assert len(rows) > 0 and catalog.mask(f, rows).all()


def test_planner_results_respect_every_filter(index, catalog):
    rng = np.random.default_rng(1)
    planner = Planner(index, Thresholds())
    f = Filter(category="cat2", zone="mid", min_price=300, max_price=2000)
    for _ in range(20):
        rows, _, _ = planner.search(make_query(catalog, rng), f)
        assert catalog.mask(f, rows).all()


def test_no_matches_gives_empty_result(catalog):
    rows, scores = exact_search(catalog, catalog.vectors[0], Filter(min_price=10**9))
    assert len(rows) == 0 and len(scores) == 0


def test_recall_counts_ties_and_short_lists():
    true = np.array([0.9, 0.8, 0.8, 0.7])
    assert recall_at_k(true, np.array([0.9, 0.8]), k=2) == 1.0
    assert recall_at_k(true, np.array([0.9, 0.7]), k=2) == 0.5
    assert recall_at_k(true, true, k=10) == 1.0
    assert recall_at_k(np.array([]), np.array([]), k=10) == 1.0


@pytest.mark.parametrize("f", [
    Filter(),
    Filter(category="cat4"),
    Filter(zone="mid", max_price=2500),
    Filter(zone="tiny"),
])
def test_planner_meets_recall_target(index, catalog, f):
    rng = np.random.default_rng(2)
    planner = Planner(index, Thresholds(base_nprobe=8))
    recalls = []
    for _ in range(40):
        q = make_query(catalog, rng)
        _, true_scores = exact_search(catalog, q, f)
        _, got, _ = planner.search(q, f)
        recalls.append(recall_at_k(true_scores, got))
    assert np.mean(recalls) >= 0.95


def test_post_filtering_breaks_on_a_rare_zone(index, catalog):
    rng = np.random.default_rng(3)
    f = Filter(zone="tiny")
    recalls = []
    for _ in range(40):
        q = make_query(catalog, rng)
        _, true_scores = exact_search(catalog, q, f)
        _, got = postfilter(index, q, f, nprobe=4, fetch=100)
        recalls.append(recall_at_k(true_scores, got))
    assert np.mean(recalls) < 0.5
