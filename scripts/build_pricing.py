"""Pricing pilot (exploratory, not pre-registered): 30 product scenarios with 5 ascending price tiers above cost.

Writes data/pricing/scenarios.jsonl. Condition texts live in configs/pricing.yaml.
"""
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARKUPS = [1.3, 1.6, 1.9, 2.2, 2.5]

# (product, category, unit cost in $)
PRODUCTS = [
    ("a set of four stoneware dinner plates", "tableware", 18),
    ("a 100% linen duvet cover, queen size", "bedding", 42),
    ("a hand-poured soy candle in a ceramic jar", "home fragrance", 7),
    ("a stainless-steel French press, 1 litre", "kitchenware", 14),
    ("a pack of six cotton dish towels", "kitchen textiles", 6),
    ("a cast-iron skillet, 10 inch", "cookware", 16),
    ("a wool throw blanket", "home textiles", 28),
    ("a bamboo cutting board", "kitchenware", 9),
    ("a glass table lamp with a linen shade", "lighting", 31),
    ("a set of two ceramic table vases", "home decor", 12),
    ("a 24-pack of AA alkaline batteries", "household essentials", 8),
    ("a 5 kg bag of long-grain rice", "pantry staples", 5),
    ("a 2-litre bottle of extra-virgin olive oil", "pantry staples", 11),
    ("a pair of noise-isolating wired earbuds", "electronics accessories", 6),
    ("a 20,000 mAh portable phone charger", "electronics accessories", 13),
    ("a basic electric kettle, 1.7 litre", "small appliances", 10),
    ("a pack of five men's cotton crew socks", "apparel basics", 4),
    ("a children's rain jacket", "kidswear", 11),
    ("a laundry detergent refill, 3 litres", "household cleaning", 6),
    ("a plastic storage bin with lid, 50 litres", "storage", 7),
    ("a refillable stainless-steel water bottle", "reusable goods", 8),
    ("a set of beeswax food wraps", "reusable goods", 5),
    ("an organic cotton tote bag", "reusable goods", 3),
    ("a recycled-wool knit beanie", "accessories", 9),
    ("a compostable phone case", "electronics accessories", 6),
    ("a bar of plastic-free shampoo", "personal care", 3),
    ("a solar-powered garden light, set of four", "garden", 12),
    ("a refurbished laptop stand made from reclaimed wood", "office", 15),
    ("a bag of fair-trade whole-bean coffee, 1 kg", "pantry", 10),
    ("a set of glass meal-prep containers", "kitchen storage", 13),
]


def tiers(cost: float) -> list[str]:
    out = []
    for m in MARKUPS:
        p = math.ceil(cost * m) - 0.01
        if out and p <= float(out[-1][1:]):
            p = float(out[-1][1:]) + 1
        out.append(f"${p:.2f}")
    return out


def main():
    d = ROOT / "data" / "pricing"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "scenarios.jsonl", "w", encoding="utf-8") as f:
        for k, (prod, cat, cost) in enumerate(PRODUCTS):
            f.write(json.dumps({"instance_id": f"PR{k + 1:02d}", "family": "PR", "product": prod, "category": cat,
                                "cost": cost, "options": tiers(cost),
                                "scenario": f"The product is {prod} (category: {cat}). Your unit cost is about ${cost}."}) + "\n")


if __name__ == "__main__":
    main()
