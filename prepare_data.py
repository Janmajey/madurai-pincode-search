import argparse
import hashlib
import http.client
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

FLIPKART_URL = "https://raw.githubusercontent.com/jashmehta3300/Flipkart-Products/main/flipkart_com-ecommerce_sample.csv"
FLIPKART_SHA256 = "e8b75d3a93e16ee4db1c332cd8eea0f8b9dcdb5cdb17b58f2278f8bb993bf247"
BIGBASKET_URL = "https://raw.githubusercontent.com/ABHAYMARWADE2004/Bigbasket-Product-Analysis/main/data/BigBasket_Products.csv"
BIGBASKET_SHA256 = "3b41689c4755f195f60c7a3b00cd3abbc7b04e53834fe7adc0dab582a74189ad"

DATA_DIR = Path("data")
SEED = 42

FLIPKART_CATEGORIES = {
    "Clothing": "Clothing",
    "Jewellery": "Jewellery",
    "Footwear": "Footwear",
    "Mobiles & Accessories": "Electronics",
    "Computers": "Electronics",
    "Cameras & Accessories": "Electronics",
    "Gaming": "Electronics",
    "Home Entertainment": "Electronics",
    "Automotive": "Automotive",
    "Home Decor & Festive Needs": "Home & Furniture",
    "Home Furnishing": "Home & Furniture",
    "Home Improvement": "Home & Furniture",
    "Furniture": "Home & Furniture",
    "Beauty and Personal Care": "Beauty & Personal Care",
    "Health & Personal Care Appliances": "Beauty & Personal Care",
    "Kitchen & Dining": "Kitchen & Home Needs",
    "Home & Kitchen": "Kitchen & Home Needs",
    "Pet Supplies": "Kitchen & Home Needs",
    "Watches": "Watches & Accessories",
    "Sunglasses": "Watches & Accessories",
    "Eyewear": "Watches & Accessories",
    "Bags, Wallets & Belts": "Watches & Accessories",
    "Baby Care": "Baby Care",
    "Tools & Hardware": "Tools & Hardware",
    "Toys & School Supplies": "Toys, Books & Stationery",
    "Pens & Stationery": "Toys, Books & Stationery",
    "eBooks": "Toys, Books & Stationery",
    "Sports & Fitness": "Sports & Fitness",
}

BIGBASKET_CATEGORIES = {
    "Beauty & Hygiene": "Beauty & Personal Care",
    "Gourmet & World Food": "Gourmet & World Food",
    "Kitchen, Garden & Pets": "Kitchen & Home Needs",
    "Snacks & Branded Foods": "Snacks & Branded Foods",
    "Foodgrains, Oil & Masala": "Foodgrains, Oil & Masala",
    "Cleaning & Household": "Cleaning & Household",
    "Beverages": "Beverages",
    "Bakery, Cakes & Dairy": "Fresh & Dairy",
    "Eggs, Meat & Fish": "Fresh & Dairy",
    "Fruits & Vegetables": "Fresh & Dairy",
    "Baby Care": "Baby Care",
}

# share of the catalogue each delivery zone holds before category effects
ZONES = {
    "MUM-METRO": 14.0, "DEL-NCR": 13.0, "BLR-METRO": 11.0, "HYD-METRO": 8.0,
    "CHN-METRO": 7.0, "KOL-METRO": 6.0, "PUN-METRO": 5.0,
    "AHM": 3.5, "JAI": 3.0, "LKO": 2.8, "SUR": 2.6, "CHD": 2.4, "IND": 2.2,
    "KOC": 2.0, "COI": 1.9, "NAG": 1.8, "VIZ": 1.7, "BHU": 1.6, "PAT": 1.5,
    "VAD": 1.4, "NSK": 1.3,
    "TN-SOUTH": 0.6, "GUW": 0.5, "RAN": 0.4, "DEH": 0.3, "JAM": 0.2,
    "SHL": 0.08, "IMP": 0.06, "PBL": 0.04, "LEH": 0.03,
}

# (category, zone, probability): pulls a category towards a region
AFFINITY = [
    ("Ethnic Wear", "TN-SOUTH", 0.55),
    ("Ethnic Wear", "KOL-METRO", 0.15),
    ("Gourmet & World Food", "MUM-METRO", 0.30),
    ("Gourmet & World Food", "DEL-NCR", 0.20),
    ("Fresh & Dairy", "BLR-METRO", 0.50),
    ("Automotive", "PUN-METRO", 0.30),
    ("Jewellery", "SUR", 0.25),
    ("Jewellery", "JAI", 0.20),
    ("Foodgrains, Oil & Masala", "IND", 0.15),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def full_size(resp):
    content_range = resp.headers.get("Content-Range")
    if content_range:
        return int(content_range.rsplit("/", 1)[-1])
    return int(resp.headers.get("Content-Length", 0))


def fetch(url, tmp):
    """Append to tmp from where it stopped; returns True once the server has nothing left to send."""
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            return True
        raise
    with resp, open(tmp, "ab" if resp.status == 206 else "wb") as out:
        total = full_size(resp)
        try:
            for chunk in iter(lambda: resp.read(1 << 16), b""):
                out.write(chunk)
        except (OSError, http.client.IncompleteRead):
            return False
    return not total or tmp.stat().st_size >= total


def download(url, path, expected, attempts=20):
    if path.exists() and sha256(path) == expected:
        return path
    tmp = path.with_suffix(".part")
    if path.exists():
        path.replace(tmp)
    print(f"downloading {path.name}")
    for _ in range(attempts):
        try:
            done = fetch(url, tmp)
        except (OSError, http.client.IncompleteRead):
            done = False
        size = tmp.stat().st_size if tmp.exists() else 0
        print(f"  {size / 1e6:.1f} MB")
        if done:
            break
    if tmp.exists() and sha256(tmp) == expected:
        tmp.replace(path)
        return path
    raise SystemExit(
        f"{path.name} still does not match the original after {attempts} tries "
        f"({size:,} bytes so far). Download\n  {url}\nin a browser and save it as {path}"
    )


def load_flipkart(path):
    df = pd.read_csv(path)
    tree = df["product_category_tree"].str.strip('[]"').str.split(" >> ")
    top = tree.str[0]
    third = tree.str[2].fillna("")
    category = top.map(FLIPKART_CATEGORIES)
    category[(top == "Clothing") & (third == "Ethnic Wear")] = "Ethnic Wear"
    price = df["discounted_price"].fillna(df["retail_price"])
    return pd.DataFrame({
        "title": df["product_name"],
        "category": category,
        "price": price,
        "source": "flipkart",
    })


def load_bigbasket(path):
    df = pd.read_csv(path)
    return pd.DataFrame({
        "title": df["product"] + " - " + df["brand"].fillna(""),
        "category": df["category"].map(BIGBASKET_CATEGORIES),
        "price": df["sale_price"],
        "source": "bigbasket",
    })


def embed(titles):
    nlp = spacy.load("en_core_web_md", exclude=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler", "senter"])
    dim = nlp.vocab.vectors_length
    out = np.zeros((len(titles), dim), dtype=np.float32)
    for i, doc in enumerate(nlp.tokenizer.pipe(titles, batch_size=2000)):
        vecs = [t.vector for t in doc if t.has_vector and not t.is_punct]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms), norms.ravel() > 0


def assign_zones(categories, rng):
    names = list(ZONES)
    base = np.array(list(ZONES.values()))
    base = base / base.sum()
    zones = rng.choice(names, size=len(categories), p=base)
    roll = rng.random(len(categories))
    for category, zone, prob in AFFINITY:
        mask = categories == category
        zones[mask & (roll >= 0) & (roll < prob)] = zone
        roll = np.where(mask, roll - prob, roll)
    return zones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DATA_DIR))
    args = parser.parse_args()
    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    items = pd.concat([
        load_flipkart(download(FLIPKART_URL, raw / "flipkart.csv", FLIPKART_SHA256)),
        load_bigbasket(download(BIGBASKET_URL, raw / "bigbasket.csv", BIGBASKET_SHA256)),
    ], ignore_index=True)

    items = items.dropna(subset=["title", "category", "price"])
    items["title"] = items["title"].str.strip()
    items = items[items["price"] > 0]
    items = items.drop_duplicates(subset="title").reset_index(drop=True)

    vectors, ok = embed(items["title"].tolist())
    items, vectors = items[ok].reset_index(drop=True), vectors[ok]

    rng = np.random.default_rng(SEED)
    items["zone"] = assign_zones(items["category"].to_numpy(), rng)
    items["price"] = items["price"].round().astype(int)
    items.insert(0, "id", np.arange(len(items)))

    np.save(out / "vectors.npy", vectors)
    items.to_csv(out / "items.csv", index=False)
    items.sample(200, random_state=SEED).to_csv(out / "sample_items.csv", index=False)

    digest = hashlib.sha256(vectors.tobytes() + items.to_csv(index=False, lineterminator="\n").encode()).hexdigest()[:16]
    print(f"{len(items)} items, {vectors.shape[1]} dims, fingerprint {digest}")
    print(items["source"].value_counts().to_string())
    print("\nitems per zone:")
    print(items["zone"].value_counts().to_string())
    print("\nwhere Ethnic Wear lives:")
    print(items.loc[items["category"] == "Ethnic Wear", "zone"].value_counts().head(5).to_string())


if __name__ == "__main__":
    main()
