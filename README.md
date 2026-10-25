# The Schelling Prior

**Do persona-prompted copies of one LLM tacitly collude, and what in the model makes them?**

We give Qwen3-32B 23 different personas (stratified roles, game theorists, business agents, the default Assistant) and ask each to pick from a menu, guess what another AI would pick, coordinate with an unseen partner, or deliberately mismatch. Exact log-probability readouts give each persona's full choice distribution. We then follow the effect into the activations and test it causally.

![model](https://img.shields.io/badge/model-Qwen3--32B%20bf16-2a78d6) ![prompts](https://img.shields.io/badge/scored%20prompts-~700k-1baf7a) ![sprint](https://img.shields.io/badge/Apart%20AI%20Collusion%20Sprint-Oct%202026-6b7280)

## FINDINGS (in brief)

- **Telling agents to coordinate makes different personas agree.** Agreement rises from Ā = 0.78 to 0.92, removing 62% of between-persona disagreement, and no single persona becomes more decisive.
- **It is a framing reflex, not strategy.** "Mismatch your partner" homogenizes personas just as much. Describing the partner as an *exact copy of you* makes no difference.
- **Only the matching goal points toward the shared guess.** Coordinating moves each persona toward its own guess of what another AI picks (+0.49); mismatching does not (−0.10).
- **In the activations, this happens in two stages.** By layer 32 every partner framing pushes all personas along one shared "generic-AI" direction ĝ. The match-versus-mismatch goal only separates in the last ~15 layers.
- **ĝ is a near-perfect detector but not a cause.** It flags partner framing with AUROC 0.9997 for every persona. Steering along it neither induces nor removes convergence.

## Method in brief

| | |
|---|---|
| **Readout** | Teacher-forced log P(option + end-of-turn), log-sum-exp over a fixed set of surface variants (case, quotes, trailing period, leading space). Off-menu mass is reported. A prefix-cached token trie makes it about 40 prompts/s on one H100. Validated against uncached forwards (fp32 exact; bf16 noise quantified) and against sampled answers. |
| **Design** | 26 personas × 5 game families × frames (pick, guess, coordinate, anti, partner descriptions) × 4 option orders × 2×2 paraphrases. Cross-order U-statistic estimators. Two-level bootstrap over personas within stratum and instances within family. Holm correction. |
| **Interpretability** | Decision-position residuals saved for every prompt. Assistant Axis (Lu et al.). Generic-AI direction ĝ with split-half estimation. Activation steering at decision and continuation positions with prefix reuse, verified equivalent to the reference path. |

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
python scripts/make_readme_figures.py              # figures from results/qwen3-32b/*.json
```

Result files live on the Modal volume `schelling-results` btw.
