import argparse
from pathlib import Path

import numpy as np
import spacy

from pincode_search import Catalog, Filter, IVFIndex, Planner, Thresholds

QUERIES = [
    ("cotton saree", Filter(category="Ethnic Wear", zone="TN-SOUTH", max_price=1500)),
    ("wireless bluetooth headphones", Filter(zone="MUM-METRO")),
    ("copper water bottle", Filter(max_price=2000)),
]


def embed(nlp, text):
    v = np.mean([t.vector for t in nlp.tokenizer(text) if t.has_vector], axis=0)
    return (v / np.linalg.norm(v)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--thresholds", default="results/thresholds.json")
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_md")
    catalog = Catalog.load(args.data)
    index = IVFIndex(catalog)
    planner = Planner(index, Thresholds.load(args.thresholds))
    if not Path(args.thresholds).exists():
        print("no thresholds file found, using defaults (run benchmark.py first)")

    for text, f in QUERIES:
        name, est = planner.choose(f)
        true = catalog.mask(f).sum() / len(catalog)
        rows, scores, _ = planner.search(embed(nlp, text), f)
        print(f'\n"{text}"  |  {f}')
        print(f"  estimated selectivity {est:.3%}, true {true:.3%}  ->  {name}")
        for r, s in zip(rows[:5], scores[:5]):
            print(f"  {s:.3f}  Rs {catalog.prices[r]:>6}  {catalog.titles[r][:70]}")


if __name__ == "__main__":
    main()
