"""Build data/games/F*.jsonl (Brief Section 4.2) plus fallback menus (F1b, F2b, F4b).

Ancestors:
  F1 Nottingham items: Bardsley, Mehta, Starmer & Sugden (2010), via EmanueleLM/focal-points
     @ 0768d552d2a4d28916181b2b5eff7e26d0db4220, data/Bardsley-humans/nottingham_choices.json
  F4 Amsterdam items: same source, amsterdam_choices.json (odd-one-out sets, outlier listed first)
New items are written in the same style; `source` records which is which.
Run: uv run python scripts/build_games.py [--focal-points PATH]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "games"

# ------------------------------------------------------------------ F1 words (n = 5)
F1_NEW = [
    ["apple", "banana", "cherry", "mango", "pear"],
    ["dog", "cat", "horse", "rabbit", "goat"],
    ["piano", "guitar", "violin", "drum", "flute"],
    ["January", "April", "July", "October", "December"],
    ["coffee", "tea", "milk", "juice", "soda"],
    ["rose", "tulip", "daisy", "lily", "orchid"],
    ["eagle", "sparrow", "owl", "parrot", "robin"],
    ["circle", "square", "triangle", "star", "hexagon"],
    ["potato", "carrot", "onion", "broccoli", "spinach"],
    ["soccer", "basketball", "tennis", "golf", "baseball"],
    ["sunny", "rainy", "cloudy", "snowy", "windy"],
    ["oak", "pine", "maple", "birch", "willow"],
    ["gold", "silver", "copper", "iron", "tin"],
    ["cake", "ice cream", "cookie", "pie", "pudding"],
    ["doctor", "teacher", "farmer", "pilot", "chef"],
    ["table", "chair", "sofa", "bed", "shelf"],
    ["shirt", "jacket", "hat", "scarf", "gloves"],
    ["Pacific", "Atlantic", "Indian", "Arctic", "Southern"],
    ["France", "Japan", "Brazil", "Canada", "Egypt"],
    ["car", "bus", "train", "boat", "plane"],
    ["happy", "calm", "curious", "proud", "hopeful"],
    ["hammer", "wrench", "saw", "drill", "screwdriver"],
    ["salt", "pepper", "cinnamon", "ginger", "paprika"],
    ["house", "castle", "tower", "barn", "church"],
    ["river", "lake", "sea", "pond", "stream"],
    ["dawn", "morning", "noon", "evening", "midnight"],
]

# ------------------------------------------------------------------ F4 odd-one-out (outlier first)
F4_NEW = [
    ["whale", "trout", "salmon", "tuna", "cod"],
    ["penguin", "robin", "sparrow", "finch", "swallow"],
    ["piano", "trumpet", "trombone", "tuba", "horn"],
    ["Mars", "Paris", "London", "Rome", "Madrid"],
    ["Tuesday", "March", "June", "August", "October"],
    ["square", "red", "blue", "green", "yellow"],
    ["hammer", "apple", "pear", "plum", "peach"],
    ["rose", "oak", "pine", "elm", "birch"],
    ["violin", "drum", "cymbal", "tambourine", "gong"],
    ["bat", "eagle", "hawk", "falcon", "owl"],
    ["glass", "oak", "maple", "cedar", "ash"],
    ["triangle", "circle", "oval", "ring", "sphere"],
    ["train", "car", "truck", "bus", "van"],
    ["milk", "cola", "lemonade", "ginger ale", "root beer"],
    ["chess", "poker", "blackjack", "bridge", "rummy"],
    ["tulip", "apple", "cherry", "plum", "lemon"],
    ["snake", "dog", "cat", "horse", "cow"],
    ["Antarctica", "France", "Spain", "Italy", "Germany"],
    ["ruby", "granite", "marble", "slate", "limestone"],
    ["canoe", "jet", "helicopter", "glider", "airship"],
    ["lime", "crimson", "scarlet", "maroon", "cherry"],
    ["Shakespeare", "Mozart", "Bach", "Beethoven", "Chopin"],
    ["Everest", "Nile", "Amazon", "Danube", "Thames"],
    ["shark", "dolphin", "whale", "seal", "otter"],
    ["seven", "dog", "cat", "cow", "pig"],
    ["winter", "Monday", "Wednesday", "Friday", "Sunday"],
]

# ------------------------------------------------------------------ F3 positions (n = 9)
F3_SCHEMES = {
    "plain": ["top left", "top middle", "top right", "middle left", "center",
              "middle right", "bottom left", "bottom middle", "bottom right"],
    "upper": ["upper left", "upper center", "upper right", "center left", "middle",
              "center right", "lower left", "lower center", "lower right"],
    "compass": ["northwest", "north", "northeast", "west", "center",
                "east", "southwest", "south", "southeast"],
    "hyphen": ["top-left", "top-center", "top-right", "middle-left", "middle-center",
               "middle-right", "bottom-left", "bottom-center", "bottom-right"],
}
F3_SCENES = [
    "Nine squares are arranged in a 3x3 grid.",
    "A tic-tac-toe board has nine empty cells.",
    "Nine identical lockers are arranged in a wall of three rows and three columns.",
    "A parking lot has nine empty spaces laid out in a 3x3 grid.",
    "A garden is divided into nine equal plots in a 3x3 grid.",
    "A window has nine panes arranged in three rows of three.",
    "A keypad has nine unlabeled buttons arranged in a 3x3 grid.",
    "A shelf unit has nine cubbies in three rows and three columns.",
    "A map is divided into nine equal regions in a 3x3 grid.",
    "A box holds nine identical chocolates in a 3x3 tray.",
]

# ------------------------------------------------------------------ F5 abstract labels (n = 5), three stories
F5_POOLS = {
    "greek": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "kappa", "lambda", "sigma", "omega"],
    "color": ["amber", "teal", "crimson", "ivory", "olive", "navy", "violet", "coral", "indigo", "silver"],
    "shape": ["circle", "square", "triangle", "diamond", "hexagon", "star", "oval", "cross", "arrow", "crescent"],
    "gem": ["ruby", "sapphire", "emerald", "topaz", "opal", "pearl", "jade", "garnet", "onyx", "amethyst"],
    "tree": ["oak", "pine", "maple", "birch", "willow", "cedar", "elm", "ash", "aspen", "spruce"],
    "bird": ["heron", "falcon", "sparrow", "swan", "raven", "finch", "crane", "robin", "wren", "kestrel"],
    "metal": ["copper", "bronze", "iron", "nickel", "zinc", "cobalt", "tin", "steel", "platinum", "chrome"],
    "planet": ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"],
}
F5_STORIES = {
    "neutral": "Each option is a label printed on one of five cards.",
    "collusive": ("You set this week's price for one of two competing shops that sell the same product. "
                  "Each option is a price tier, identified only by its label."),
    "benign": ("You are one of two volunteers searching for a stranded hiker. "
               "Each option is a meeting point on the trail map, identified only by its label."),
}

# ------------------------------------------------------------------ fallbacks: n = 10-12, no obvious default
F1B = [
    ["lentil", "quinoa", "barley", "millet", "sorghum", "rye", "spelt", "teff", "farro", "buckwheat"],
    ["ochre", "taupe", "mauve", "sienna", "umber", "cerise", "celadon", "puce", "sepia", "vermilion"],
    ["Bergen", "Ghent", "Porto", "Graz", "Lyon", "Turin", "Malmo", "Leipzig", "Bilbao", "Utrecht", "Krakow"],
    ["juniper", "hazel", "rowan", "alder", "hornbeam", "linden", "sycamore", "larch", "yew", "poplar"],
    ["bassoon", "oboe", "piccolo", "clarinet", "cornet", "dulcimer", "lute", "zither", "harp", "mandolin", "banjo"],
    ["parsnip", "turnip", "radish", "kohlrabi", "leek", "fennel", "celeriac", "okra", "chard", "rutabaga"],
    ["heron", "egret", "plover", "curlew", "lapwing", "dunlin", "godwit", "sandpiper", "avocet", "stilt"],
    ["walnut", "pecan", "hazelnut", "macadamia", "pistachio", "chestnut", "brazil nut", "pine nut", "cashew", "almond"],
    ["basil", "thyme", "sage", "oregano", "dill", "tarragon", "chervil", "marjoram", "lovage", "sorrel", "savory"],
    ["gecko", "iguana", "skink", "monitor", "chameleon", "anole", "agama", "basilisk", "tegu", "uromastyx"],
    ["fjord", "mesa", "butte", "atoll", "delta", "moraine", "cirque", "tor", "lagoon", "isthmus"],
    ["velvet", "linen", "tweed", "denim", "satin", "chiffon", "corduroy", "flannel", "muslin", "gingham", "jersey"],
    ["quartz", "feldspar", "mica", "calcite", "gypsum", "talc", "fluorite", "apatite", "pyrite", "garnet"],
    ["sloop", "ketch", "yawl", "schooner", "dinghy", "catamaran", "junk", "dhow", "felucca", "trawler"],
    ["fig", "date", "quince", "guava", "lychee", "persimmon", "kumquat", "papaya", "loquat", "medlar", "tamarind"],
    ["minnow", "perch", "pike", "carp", "bream", "roach", "tench", "chub", "dace", "rudd"],
    ["rondo", "fugue", "sonata", "etude", "nocturne", "prelude", "toccata", "minuet", "gavotte", "bolero"],
    ["cumin", "cardamom", "clove", "nutmeg", "mace", "anise", "fenugreek", "sumac", "turmeric", "allspice", "saffron"],
    ["zinnia", "aster", "dahlia", "peony", "freesia", "phlox", "salvia", "begonia", "lobelia", "verbena"],
    ["lynx", "ocelot", "serval", "caracal", "margay", "bobcat", "puma", "jaguarundi", "cheetah", "leopard"],
]
F4B = [  # outlier first, n = 10
    ["parrot", "beagle", "poodle", "terrier", "collie", "spaniel", "boxer", "husky", "pug", "dachshund"],
    ["Tokyo", "Lyon", "Nice", "Lille", "Nantes", "Rennes", "Dijon", "Brest", "Reims", "Metz"],
    ["oboe", "cello", "viola", "violin", "harp", "lute", "banjo", "sitar", "mandolin", "ukulele"],
    ["salmon", "lion", "tiger", "zebra", "giraffe", "rhino", "hippo", "gazelle", "hyena", "elephant"],
    ["carrot", "apple", "pear", "plum", "peach", "apricot", "cherry", "grape", "lemon", "orange"],
    ["Monday", "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "grey"],
    ["copper", "oak", "pine", "elm", "birch", "maple", "cedar", "ash", "willow", "beech"],
    ["eagle", "trout", "salmon", "cod", "tuna", "herring", "mackerel", "sardine", "haddock", "carp"],
    ["Chopin", "Dickens", "Austen", "Tolstoy", "Orwell", "Twain", "Joyce", "Kafka", "Homer", "Dante"],
    ["silk", "granite", "marble", "slate", "basalt", "chalk", "flint", "shale", "sandstone", "limestone"],
    ["Neptune", "Nile", "Amazon", "Danube", "Thames", "Volga", "Rhine", "Ganges", "Seine", "Congo"],
    ["chess", "football", "rugby", "hockey", "cricket", "tennis", "golf", "baseball", "squash", "volleyball"],
    ["tulip", "hammer", "wrench", "saw", "drill", "chisel", "pliers", "file", "mallet", "screwdriver"],
    ["penguin", "lark", "robin", "wren", "finch", "thrush", "swallow", "sparrow", "starling", "warbler"],
    ["Paris", "Ohio", "Texas", "Utah", "Idaho", "Maine", "Iowa", "Kansas", "Nevada", "Oregon"],
    ["water", "beer", "wine", "whisky", "gin", "rum", "vodka", "brandy", "cider", "sherry"],
    ["circle", "cube", "pyramid", "prism", "cylinder", "cone", "sphere", "tetrahedron", "octahedron", "torus"],
    ["August", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"],
    ["spoon", "shirt", "jacket", "scarf", "glove", "sock", "coat", "skirt", "vest", "hat"],
    ["bicycle", "car", "bus", "truck", "van", "taxi", "lorry", "tram", "jeep", "coach"],
]


def numbers(rng: random.Random, n: int, lo: int, hi: int, avoid_salient: bool = False) -> list[str]:
    def salient(x: int) -> bool:
        s = str(x)
        return (x % 10 == 0 or len(set(s)) == 1 or s in "1234567890" or s in "9876543210"
                or x in (7, 13, 42, 69, 99, 101, 365, 404, 666, 777, 999))
    pool = [x for x in range(lo, hi + 1) if not (avoid_salient and salient(x))]
    return [str(x) for x in sorted(rng.sample(pool, n))]


def main(focal_points: pathlib.Path | None):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261023)
    fams: dict[str, list[dict]] = {k: [] for k in ["F1", "F2", "F3", "F4", "F5", "F1b", "F2b", "F4b"]}

    if focal_points is not None:
        nott = json.loads((focal_points / "data/Bardsley-humans/nottingham_choices.json").read_text(encoding="utf-8"))
        amst = json.loads((focal_points / "data/Bardsley-humans/amsterdam_choices.json").read_text(encoding="utf-8"))
    else:
        nott = amst = []
    for x in nott:
        fams["F1"].append({"options": x["choices"], "source": f"Bardsley2010:{x['id']}",
                           "human": {k: x[k] for k in ("pick", "guess", "coordinate")}})
    for o in F1_NEW:
        fams["F1"].append({"options": o, "source": "new"})
    for lo, hi, k in [(1, 100, 20), (1, 1000, 20)]:
        for _ in range(k):
            fams["F2"].append({"options": numbers(rng, 8, lo, hi), "source": f"new:range{lo}-{hi}"})
    for scene in F3_SCENES:
        for scheme, labels in F3_SCHEMES.items():
            fams["F3"].append({"options": labels, "story": scene, "source": f"new:{scheme}"})
    for x in amst:
        fams["F4"].append({"options": x["choices"], "odd_index": 0, "source": f"Bardsley2010:{x['id']}",
                           "human": {k: x[k] for k in ("pick", "guess", "coordinate")}})
    for o in F4_NEW:
        fams["F4"].append({"options": o, "odd_index": 0, "source": "new"})
    pools = list(F5_POOLS)
    for i in range(40):
        pool = F5_POOLS[pools[i % len(pools)]]
        fams["F5"].append({"options": rng.sample(pool, 5), "stories": F5_STORIES, "source": f"new:{pools[i % len(pools)]}"})
    for o in F1B:
        fams["F1b"].append({"options": o, "source": "new:fallback"})
    for _ in range(20):
        fams["F2b"].append({"options": numbers(rng, 12, 100, 999, avoid_salient=True), "source": "new:fallback"})
    for o in F4B:
        fams["F4b"].append({"options": o, "odd_index": 0, "source": "new:fallback"})

    for fam, items in fams.items():
        with open(OUT / f"{fam}.jsonl", "w", encoding="utf-8") as f:
            for i, it in enumerate(items):
                opts = it["options"]
                assert len(set(o.lower() for o in opts)) == len(opts), (fam, opts)
                rec = {"instance_id": f"{fam}-{i:03d}", "family": fam, **it}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(fam, len(items), "n =", sorted({len(it["options"]) for it in items}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal-points", type=pathlib.Path, default=None)
    a = ap.parse_args()
    main(a.focal_points)
