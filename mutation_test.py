import argparse
from pathlib import Path

import numpy as np
import spacy

from pincode_search import Catalog, Filter, IVFIndex, Thresholds
from pincode_search.bench import evaluate, planner_fn
from pincode_search.workload import BUCKETS, build_workload, fixed_workload

SEED = 7
BUSIEST = "MUM-METRO"
NEW_CATEGORY = "Festive Gifting"

GIFTS = ["Dry Fruit Gift Hamper", "Kaju Katli Gift Box", "Brass Diya Set", "Rangoli Colour Powder",
         "Scented Candle Gift Set", "Silver Coin Lakshmi Ganesh", "Decorative Toran", "LED String Lights",
         "Puja Thali Set", "Chocolate Gift Pack", "Soan Papdi Box", "Clay Lantern"]
STYLES = ["Premium", "Handcrafted", "Festive", "Traditional", "Luxury", "Classic", "Designer"]
VARIANTS = ["Pack of 2", "Set of 6", "500 g", "1 kg", "Small", "Large", "Gold", "Red", "Combo"]
GIFT_ZONES = ["DEL-NCR", "MUM-METRO", "JAI", "AHM", "LKO"]


def new_items(nlp, n, dim, rng):
    titles = [f"{rng.choice(STYLES)} {rng.choice(GIFTS)} - {rng.choice(VARIANTS)}" for _ in range(n)]
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i, doc in enumerate(nlp.tokenizer.pipe(titles)):
        vectors[i] = np.mean([t.vector for t in doc if t.has_vector and not t.is_punct], axis=0)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors += rng.normal(0, 0.3 / np.sqrt(dim), vectors.shape).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    zones = rng.choice(GIFT_ZONES, n, p=[0.35, 0.3, 0.15, 0.1, 0.1])
    prices = np.exp(rng.uniform(np.log(199), np.log(4999), n)).astype(int)
    return vectors, titles, [NEW_CATEGORY] * n, zones, prices


def partition_report(index):
    sizes = index.list_sizes()
    rows = np.flatnonzero(index.catalog.alive)
    fit = np.einsum("ij,ij->i", index.catalog.vectors[rows], index.centroids[index.list_of[rows]])
    return sizes, float(fit.mean())


def measure(catalog, index, t):
    planner = {"planner": planner_fn(index, t)}
    queries = build_workload(catalog, per_bucket=40, seed=2)
    by_bucket = {b: evaluate(index, [q for q in queries if q.bucket == b], planner)["planner"] for b in BUCKETS}
    gifting = fixed_workload(catalog, [Filter(category=NEW_CATEGORY), Filter(category=NEW_CATEGORY, zone="JAI")], 30, seed=3)
    by_bucket["new category"] = evaluate(index, gifting, planner)["planner"] if gifting else None
    return by_bucket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--thresholds", default="results/thresholds.json")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    t = Thresholds.load(args.thresholds)
    catalog = Catalog.load(args.data)
    index = IVFIndex(catalog)
    stages = {}
    sizes, fit = partition_report(index)
    stages["before"] = (measure(catalog, index, t), sizes, fit)

    busiest = np.flatnonzero(catalog.alive & (catalog.zones == catalog.zone_codes.get(BUSIEST)))
    gone = rng.choice(busiest, size=len(busiest) // 10, replace=False)
    catalog.delete(gone)
    index.remove(gone)

    nlp = spacy.load("en_core_web_md", exclude=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler", "senter"])
    n_new = int(0.05 * len(catalog))
    added = catalog.append(*new_items(nlp, n_new, catalog.vectors.shape[1], rng))
    index.add(added)
    sizes, fit = partition_report(index)
    stages["after (incremental)"] = (measure(catalog, index, t), sizes, fit)

    new_lists = np.unique(index.list_of[added], return_counts=True)
    top = np.sort(new_lists[1])[::-1][:3]

    rebuilt = IVFIndex(catalog)
    sizes, fit = partition_report(rebuilt)
    stages["after (rebuilt)"] = (measure(catalog, rebuilt, t), sizes, fit)

    lines = [
        "# Mutation test",
        "",
        f"Deleted {len(gone)} items from {BUSIEST} (10%), added {n_new} '{NEW_CATEGORY}' items (5%) "
        "with a new category, new titles and their own price/zone mix.",
        f"The new items landed in {len(new_lists[0])} of {index.n_lists} partitions; the three biggest took {', '.join(map(str, top))}.",
        "",
        "| Stage | " + " | ".join(stages["before"][0]) + " | Largest partition | Size std | Mean item-centroid similarity |",
        "| --- " * (len(stages["before"][0]) + 4) + "|",
    ]
    for name, (results, sizes, fit) in stages.items():
        cells = " | ".join(f"{r.recall:.3f} / {r.p(50):.2f} ms" if r else "n/a" for r in results.values())
        lines.append(f"| {name} | {cells} | {sizes.max()} | {sizes.std():.1f} | {fit:.3f} |")
    lines += ["", "Cells are planner recall@10 / p50 latency."]

    text = "\n".join(lines) + "\n"
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "mutation.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
