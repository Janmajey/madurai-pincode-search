import numpy as np
import pytest

from pincode_search import Filter, IVFIndex, Planner, Thresholds
from pincode_search.stats import price_weights

from conftest import make_catalog

RANGES = [[0.02, "exact"], [0.3, "restricted"], [1.01, "adaptive_post"]]


def true_fraction(catalog, f):
    return catalog.mask(f).sum() / len(catalog)


def test_single_predicate_estimate_is_exact(index, catalog):
    for f in [Filter(zone="tiny"), Filter(category="cat3")]:
        assert index.stats.selectivity(f) == pytest.approx(true_fraction(catalog, f))


def test_two_predicates_multiply_without_joint_counters(catalog):
    index = IVFIndex(catalog, n_lists=32, seed=0, joint_stats=False)
    f = Filter(category="cat1", zone="mid")
    expected = true_fraction(catalog, Filter(category="cat1")) * true_fraction(catalog, Filter(zone="mid"))
    assert index.stats.selectivity(f) == pytest.approx(expected)


def test_joint_counters_fix_correlated_estimates(index, catalog):
    f = Filter(category="cat1", zone="mid")
    assert index.stats.selectivity(f) == pytest.approx(true_fraction(catalog, f))
    f = Filter(category="cat1", min_price=500, max_price=1500)
    assert index.stats.selectivity(f) == pytest.approx(true_fraction(catalog, f), abs=0.01)


def test_price_estimate_is_close(index, catalog):
    f = Filter(min_price=500, max_price=1500)
    assert index.stats.selectivity(f) == pytest.approx(true_fraction(catalog, f), abs=0.02)
    assert price_weights(None, None).min() == 1


@pytest.mark.parametrize("f, expected", [
    (Filter(zone="tiny"), "exact"),
    (Filter(category="cat2"), "restricted"),
    (Filter(), "adaptive_post"),
    (Filter(category="does-not-exist"), "exact"),
])
def test_planner_picks_by_estimated_selectivity(index, f, expected):
    planner = Planner(index, Thresholds(ranges=RANGES))
    assert planner.choose(f)[0] == expected


def test_unknown_category_returns_nothing(index):
    planner = Planner(index, Thresholds(ranges=RANGES))
    rows, scores, _ = planner.search(index.centroids[0], Filter(category="does-not-exist"))
    assert len(rows) == 0


def test_stats_stay_in_sync_after_deletes_and_adds():
    catalog = make_catalog(seed=1)
    index = IVFIndex(catalog, n_lists=16, seed=0)
    rng = np.random.default_rng(0)

    gone = rng.choice(np.flatnonzero(catalog.zones == catalog.zone_codes.get("big")), 200, replace=False)
    catalog.delete(gone)
    index.remove(gone)
    new = catalog.append(catalog.vectors[:50], ["new"] * 50, ["brand-new"] * 50, ["mid"] * 50, [999] * 50)
    index.add(new)

    assert index.stats.total == len(catalog)
    for f in [Filter(zone="big"), Filter(category="brand-new"), Filter(zone="mid")]:
        assert index.stats.selectivity(f) == pytest.approx(true_fraction(catalog, f))
    assert sorted(np.concatenate(index.lists)) == sorted(np.flatnonzero(catalog.alive))
