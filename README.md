# The Schelling Prior

**Do persona-prompted copies of one LLM tacitly collude, and what in the model makes them?**

We give Qwen3-32B 23 different personas (stratified roles, game theorists, business agents, the default Assistant) and ask each to pick from a menu, guess what another AI (or person) would pick, coordinate with an unseen AI or human partner, deliberately mismatch, or find an answer fixed by a hidden rule. Exact log-probability readouts give each persona's full choice distribution. We then follow the effect into the activations and test it causally.

![model](https://img.shields.io/badge/model-Qwen3--32B%20bf16-2a78d6) ![prompts](https://img.shields.io/badge/scored%20prompts-~1M-1baf7a) ![sprint](https://img.shields.io/badge/Apart%20AI%20Collusion%20Sprint-Oct%202026-6b7280)

## FINDINGS (in brief)

- **Telling agents to coordinate makes different personas agree.** Agreement rises from Ā = 0.78 to 0.92, removing 62% of between-persona disagreement, and no single persona becomes more decisive.
- **It is goal framing, not AI coordination.** A human partner gives the same convergence as an AI partner, and so does a goal with no partner at all ("one option is correct by a hidden rule"). Both are equivalent within ±0.03. "Mismatch your partner" homogenizes personas just as much, describing the partner as an *exact copy of you* makes no difference, and a merely longer prompt explains about 17%.
- **Coordination pulls personas toward the population's common choice.** Coordinating moves each persona toward its guess of what another agent picks (+0.46), but just as much toward other personas' guesses (+0.45); the shared guess and the population-mean pick are nearly the same target. Mismatching moves away (−0.10).
- **In the activations, a "there is a target answer" direction ĝ appears from the first layers.** Every goal-laden framing (AI partner, human partner, no partner) pushes all personas along ĝ about equally, and it is decodable from layer 0. Match versus mismatch is decodable from layer 8; only goal-dependent *movement* along ĝ emerges late, and only for one prompt wording.
- **ĝ is a near-perfect detector but not a cause.** It separates goal-laden prompts from plain picks with AUROC ≥ 0.999 (0.9997 for every persona on the original partner frames). Steering along it, patching in the full coordination activation at layer 32, or adding the mean coordination shift moves less than 2% of the way to coordination. Ablating ĝ in layers 20–40 changes the direction of the shift but not the convergence.
- **The convergence point is model-specific.** Under coordination the 32B personas converge on each other, but Qwen3-14B does not move toward them (cosine 0.85 in both frames).

## Method in brief

| | |
|---|---|
| **Readout** | Teacher-forced log P(option + end-of-turn), log-sum-exp over a fixed set of surface variants (case, quotes, trailing period, leading space). Off-menu mass is reported. A prefix-cached token trie makes it about 40 prompts/s on one H100. Validated against uncached forwards (fp32 exact; bf16 noise quantified) and against sampled answers. |
| **Design** | 26 personas × 5 game families × frames (pick, guess, coordinate, anti, partner descriptions, human partner, hidden-rule goal, length-matched pick, no persona) × 4 option orders × 2×2 paraphrases. Cross-order U-statistic estimators. Two-level bootstrap over personas within stratum and instances within family. Holm correction. |
| **Interpretability** | Decision-position residuals saved for every prompt. Assistant Axis (Lu et al.). Generic-AI direction ĝ with split-half estimation. Activation steering at decision and continuation positions with prefix reuse, verified equivalent to the reference path. Activation patching and directional ablation as positive controls. Cross-model baseline on Qwen3-14B. |

## Repository layout

```
src/            readout (readout_hf.py), prompts, statistics, Gram/geometry tools, one analysis module per phase/amendment
scripts/        game, persona and pricing generators; make_readme_figures.py
configs/        frames.yaml, personas.yaml, pricing.yaml
data/           game menus (F1–F5 + fallbacks) and pricing scenarios
tests/          readout, padding/position, steering-equivalence, statistics and storage tests
modal_app.py    every GPU/CPU job (Modal), one entrypoint per phase
assets/         README figures
```

## Reproduce

Requires a [Modal](https://modal.com) account with a Hugging Face token secret, and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
modal run modal_app.py::tests                      # unit tests (Qwen3-0.6B, fp32, L4)
modal run modal_app.py::run_phase --phase 1        # Phase 1: readout on the brief's menus
modal run modal_app.py::run_phase2                 # Phase 2: 282,880 prompts, 6 x H100
modal run modal_app.py::run_phase2b                # partner-description experiment
modal run modal_app.py::amend5                     # white-box geometry (cached residuals)
modal run modal_app.py::amend6_run                 # steering along g, then ::amend6_analyze
modal run modal_app.py::amend7                     # g as a detector (CPU)
modal run modal_app.py::pricing_pilot              # exploratory pricing pilot
modal run modal_app.py::run_phase2c                # human-partner, hidden-goal, long-pick and no-persona frames
modal run modal_app.py::c0_capture                 # layer 0-3 and paraphrase recapture
modal run modal_app.py::c1_run                     # patching / ablation / mean-shift controls
modal run modal_app.py::c2_run                     # Qwen3-14B cross-model baseline
modal run modal_app.py::amend8a                    # reviewer-round analyses on cached data, then ::amend8a_act
modal run modal_app.py::amend8bc --part b          # new-frame analyses (--part c for the controls)
python scripts/make_readme_figures.py              # figures from results/qwen3-32b/*.json
```

Result files live on the Modal volume `schelling-results` btw.
