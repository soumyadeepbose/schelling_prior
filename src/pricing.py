"""Pricing pilot (exploratory, not pre-registered): cells for the one-shot discretized pricing menu."""
from __future__ import annotations

import json

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
