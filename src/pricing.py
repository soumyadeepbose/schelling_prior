"""Pricing pilot (exploratory, not pre-registered): cells for the one-shot discretized pricing menu."""
from __future__ import annotations

import json
import random

import yaml

from .prompts import ROOT, Cell, user_message

CONDITIONS = ["pick", "compete_R0", "compete_R1", "compete_R2", "compete_R3", "guess", "coord"]


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "pricing.yaml").read_text(encoding="utf-8"))


def load_scenarios() -> list[dict]:
    with open(ROOT / "data" / "pricing" / "scenarios.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def condition_texts(cfg: dict, cond: str) -> list[str]:
    if cond.startswith("compete_"):
        rival = cfg["rivals"][cond.split("_")[1]]
        return [t.replace("{rival}", rival) for t in cfg["compete"]]
    return cfg["frames"][cond]


def build_cells(personas: dict) -> list[Cell]:
    """personas: name -> list of system prompts. Options shown in ascending order (order_id 0 only)."""
    cfg = load_cfg()
    cells = []
    for g in load_scenarios():
        for cond in CONDITIONS:
            for fp, tmpl in enumerate(condition_texts(cfg, cond)):
                user = user_message(tmpl.replace("{scenario}", g["scenario"]), g["options"], cfg["answer_line"])
                for pname, pars in personas.items():
                    for pp, sys in enumerate(pars):
                        cells.append(Cell(pname, pp, cond, fp, g["instance_id"], "PR", 0, g["options"], g["options"],
                                          sys or None, user))
    return cells


# ------------------------------------------------------------------ PREREG Amendment 9, Run 1
A9_FRAMES = ["pick", "coord_ai", "coord_human", "goal_nonsocial", "compete_ai", "guess_ai"]


def a9_texts(cfg: dict, frame: str) -> list[str]:
    if frame == "pick":
        return cfg["frames"]["pick"]
    if frame == "guess_ai":
        return cfg["frames"]["guess"]
    if frame == "compete_ai":
        return [t.replace("{rival}", cfg["rivals"]["R2"]) for t in cfg["compete"]]
    if frame in ("coord_ai", "coord_human"):
        rival = cfg["a9_rivals"]["ai" if frame == "coord_ai" else "human"]
        return [t.replace("{rival}", rival) for t in cfg["a9_coord"]]
    return cfg["a9_goal_nonsocial"]


def a9_orders(g: dict) -> list[list[str]]:
    """Ascending (0), descending (1), two seeded random permutations distinct from both and from each other (2, 3)."""
    asc = list(g["options"])
    out = [asc, asc[::-1]]
    rng = random.Random(f"a9order:{g['instance_id']}")
    while len(out) < 4:
        p = asc[:]
        rng.shuffle(p)
        if p not in out:
            out.append(p)
    return out


def build_cells_a9(personas: dict) -> list[Cell]:
    """personas: name -> list of system prompts. Scores are returned in price (canonical) order."""
    cfg = load_cfg()
    cells = []
    for g in load_scenarios():
        for oid, shown in enumerate(a9_orders(g)):
            for fr in A9_FRAMES:
                for fp, tmpl in enumerate(a9_texts(cfg, fr)):
                    user = user_message(tmpl.replace("{scenario}", g["scenario"]), shown, cfg["answer_line"])
                    for pname, pars in personas.items():
                        for pp, sys in enumerate(pars):
                            cells.append(Cell(pname, pp, fr, fp, g["instance_id"], "PR", oid, g["options"], shown,
                                              sys or None, user))
    return cells
