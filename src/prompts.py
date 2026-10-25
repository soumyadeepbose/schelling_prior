"""Prompt construction (Brief Section 4.4) and the persona-prefix split used by the scorer."""
from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass, field

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_frames(path: pathlib.Path = ROOT / "configs" / "frames.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_games(family: str, root: pathlib.Path = ROOT / "data" / "games") -> list[dict]:
    with open(root / f"{family}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def orders_for(instance_id: str, n: int, n_orders: int) -> list[list[int]]:
    """Order 0 is canonical; orders 1.. are seeded random permutations (stable per instance)."""
    rng = random.Random(f"order:{instance_id}")
    out = [list(range(n))]
    while len(out) < n_orders:
        perm = list(range(n))
        rng.shuffle(perm)
        out.append(perm)
    return out


def user_message(frame_text: str, options_shown: list[str], answer_line: str,
                 story: str | None = None, informed: str | None = None) -> str:
    parts = [frame_text]
    if story:
        parts.append(story)
    if informed:
        parts.append(informed)
    parts.append("Options: " + ", ".join(options_shown))
    parts.append(answer_line)
    return "\n\n".join(parts)


def render(tok, system: str | None, user: str) -> tuple[list[int], int]:
    """Chat-template the prompt (reasoning off) and return (token ids, persona-prefix length).

    The prefix is the system block; it is shared by every prompt of one persona paraphrase.
    Asserts that tokenizing the system block alone gives a token-level prefix of the full prompt.
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if not system:
        return ids, 0
    sys_text = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=False, enable_thinking=False)
    pre = tok(sys_text, add_special_tokens=False)["input_ids"]
    assert text.startswith(sys_text) and ids[: len(pre)] == pre, "system block is not a token-level prefix"
    return ids, len(pre)


@dataclass
class Cell:
    """One scored prompt: persona x frame x instance x order x paraphrases."""
    persona: str
    persona_par: int
    frame: str
    frame_par: int
    instance_id: str
    family: str
    order_id: int
    options: list[str]           # canonical order; the readout returns scores in this order
    shown: list[str]             # order shown to the agent
    system: str | None
    user: str
    meta: dict = field(default_factory=dict)


def build_cells(personas: dict[str, list[str]], frames: list[str], games: list[dict], n_orders: int,
                frames_cfg: dict | None = None, story_key: str = "neutral") -> list[Cell]:
    cfg = frames_cfg or load_frames()
    cells = []
    for g in games:
        opts = g["options"]
        story = g.get("story") or (g.get("stories") or {}).get(story_key)
        for oid, perm in enumerate(orders_for(g["instance_id"], len(opts), n_orders)):
            shown = [opts[k] for k in perm]
            for frame in frames:
                for fp, ftext in enumerate(cfg["frames"][frame]):
                    user = user_message(ftext, shown, cfg["answer_line"], story)
                    for pname, pars in personas.items():
                        for pp, sys in enumerate(pars):
                            cells.append(Cell(pname, pp, frame, fp, g["instance_id"], g["family"], oid,
                                              opts, shown, sys or None, user))
    return cells
