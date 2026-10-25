"""README figures from the cached result JSONs (results/qwen3-32b/*.json) -> assets/*.png. Light, runs locally."""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "qwen3-32b"
OUT = ROOT / "assets"
BLUE, ORANGE, GREEN, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9aa0a8", "#14171c", "#6b7280"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9ced6", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#eceef1", "grid.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 200, "legend.frameon": False,
})


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def dots(ax, rows, xlim=None, ref=None, ref_label=None, fmt="{:.2f}"):
    """Horizontal estimate + 95% interval rows: (label, ci, colour)."""
    for k, (lab, c, col) in enumerate(rows[::-1]):
        ax.plot([c["lo"], c["hi"]], [k, k], color=col, lw=3, solid_capstyle="round", alpha=0.45)
        ax.plot(c["est"], k, "o", color=col, ms=8, mec="white", mew=1.5, zorder=3)
        ax.text(c["hi"] + (xlim[1] - xlim[0]) * 0.02 if xlim else c["hi"], k, fmt.format(c["est"]), va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=1, ls="--")
        if ref_label:
            ax.text(ref, -0.55, " " + ref_label, color=MUTED, fontsize=8, ha="left", va="bottom")
    ax.set_ylim(-0.6, len(rows) - 0.4)


def fig_convergence():
    A = load("amend3_partA.json")["T1.0"]
    A4 = load("amend4.json")["T1.0"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    g = A["A3_guess_homogeneity"]
    dots(axs[0], [("pick", g["pick"], GREY), ("guess", g["guess"], GREEN), ("coordinate", g["coord"], BLUE)], (0.7, 1.0))
    axs[0].set_title("Agreement between personas (F1–F5)")
    axs[0].set_xlabel("Ā  (mean cosine between personas' choice distributions)")
    a = A["A1_anti_control"]
    dots(axs[1], [("pick", a["pick"], GREY), ("coordinate", a["coord"], BLUE), ("anti-coordinate", a["anti"], ORANGE)], (0.7, 1.0))
    axs[1].set_title("Mismatching homogenizes too (F1, F3)")
    axs[1].set_xlabel("Ā")
    dots(axs[2], [("coordinate", A4["coord_alignment_F1F3"], BLUE), ("anti-coordinate", A4["A4_1_anti_alignment"]["population"], ORANGE)],
         (-0.4, 0.8), ref=0)
    axs[2].set_title("Only matching moves toward the guess")
    axs[2].set_xlabel("cos(frame − pick, guess − pick)")
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png")
    plt.close(fig)


def fig_geometry():
    S = load("amend5.json")["sweep"]
    L = sorted(int(k) for k in S)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
    c = [S[str(l)]["W2_contrast_F13"] for l in L]
    axs[0].fill_between(L, [x["lo"] for x in c], [x["hi"] for x in c], color=BLUE, alpha=0.15, lw=0)
    axs[0].plot(L, [x["est"] for x in c], "-o", color=BLUE, ms=3.5, lw=2)
    axs[0].axhline(0, color=MUTED, lw=1)
    axs[0].axvline(32, color=MUTED, lw=1, ls=":")
    axs[0].set_title("The goal separates only in late layers")
    axs[0].set_xlabel("layer")
    axs[0].set_ylabel("cos(Δcoord, Δguess) − cos(Δanti, Δguess)")
    for key, col, lab in (("W1_coord_over_pick", BLUE, "coordinate / pick"), ("W1_anti_over_pick_F13", ORANGE, "anti / pick")):
        v = [S[str(l)][key] for l in L]
        axs[1].fill_between(L, [x["lo"] for x in v], [x["hi"] for x in v], color=col, alpha=0.12, lw=0)
        axs[1].plot(L, [x["est"] for x in v], "-o", color=col, ms=3.5, lw=2, label=lab)
    axs[1].axhline(1, color=MUTED, lw=1)
    axs[1].axvline(32, color=MUTED, lw=1, ls=":")
    axs[1].set_title("Partner framings pull personas together")
    axs[1].set_xlabel("layer")
    axs[1].set_ylabel("persona distance, frame / pick")
    axs[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "geometry.png")
    plt.close(fig)


def fig_steering():
    R = load("amend6.json")
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))

    def curve(ax, part, m, lab_g, col=BLUE, rand=True, band=True):
        P = R[part]
        d = P["doses"]
        g = P[f"curve_g_{m}"]
        if band:
            ax.fill_between(d, [x["lo"] for x in g], [x["hi"] for x in g], color=col, alpha=0.15, lw=0)
        ax.plot(d, [x["est"] for x in g], "-o", color=col, lw=2, ms=4, label=lab_g)
        if rand and f"curve_r_{m}" in P:
            r = P[f"curve_r_{m}"]
            ax.plot(d, [x["est"] for x in r], "-o", color=GREY, lw=2, ms=4, label="random directions (mean of 5)")
        ax.set_xlabel("dose (units of the average coordination pull along ĝ)" if part != "S3" else "dose (units of the layer-53 goal shift)")

    curve(axs[0], "S1", "Abar", "pick + ĝ", band=False)
    axs[0].set_ylim(0.73, 0.76)
    axs[0].set_title("S1 · adding ĝ: no convergence")
    axs[0].set_ylabel("Ā between personas")
    axs[0].text(0.02, 0.06, "coordinate frame: Ā = 0.90", transform=axs[0].transAxes, color=MUTED, fontsize=8.5)
    axs[0].legend(loc="upper right", fontsize=8.5)
    curve(axs[1], "S2", "retention", "coordinate − ĝ", band=False)
    axs[1].set_title("S2 · removing ĝ: personas not restored")
    axs[1].set_ylabel("centred retention of own pick")
    axs[1].set_ylim(0.66, 0.69)
    axs[1].legend(loc="lower left", fontsize=8.5)
    curve(axs[2], "S3", "Phi", "pick + late goal direction (layer 53)", GREEN, rand=False)
    coord_phi = R["S2"]["curve_g_Phi"][0]["est"]
    axs[2].axhline(coord_phi, color=MUTED, lw=1, ls="--")
    axs[2].text(0.02, coord_phi + 0.002, "coordinate frame", color=MUTED, fontsize=8.5)
    axs[2].set_title("S3 (exploratory) · layer-53 goal direction")
    axs[2].set_ylabel("Φ (coordination capacity)")
    axs[2].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "steering.png")
    plt.close(fig)


def fig_detector():
    A = load("amend7.json")
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
    q = A["M3_random"]
    rows = [("ĝ · partner vs pick", A["M1a_overall"], BLUE), ("ĝ · worst persona", A["M1b_min_persona"], BLUE),
            ("ĝ · worst held-out family", A["M2_min_family"], BLUE), ("Assistant Axis · partner vs pick", A["M3_axis_auroc"], GREY),
            ("ĝ · coordinate vs anti", A["M4_ghat_coord_vs_anti_all_cells"], ORANGE),
            ("layer-53 goal dir · coordinate vs anti", A["M4_layer53"]["m53hat_l53"], GREEN),
            ("ĝ · pricing partner vs pick", A["M5_pricing"]["all"], INK)]
    dots(axs[0], rows, (0.4, 1.06), ref=0.5, ref_label="chance", fmt="{:.3f}")
    axs[0].set_title("ĝ detects partner framing, but not the goal")
    axs[0].set_xlabel("AUROC (held-out instances)")
    axs[1].hist(q["values"], bins=np.linspace(0.5, 1, 26), color=GREY, alpha=0.8, rwidth=0.88)
    for x, col, lab in ((A["M1a_overall"]["est"], BLUE, "ĝ"), (A["M3_axis_auroc"]["est"], INK, "Assistant Axis"), (q["q99"], MUTED, "random 99th pct")):
        axs[1].axvline(x, color=col, lw=2, ls="-" if col != MUTED else "--")
        axs[1].text(x - 0.005, axs[1].get_ylim()[1] * 0.92, lab, color=col, ha="right", fontsize=8.5)
    axs[1].set_title("…against 100 random directions")
    axs[1].set_xlabel("sign-free AUROC, partner vs pick")
    axs[1].set_ylabel("directions")
    fig.tight_layout()
    fig.savefig(OUT / "detector.png")
    plt.close(fig)


def fig_pricing():
    P = load("pricing_pilot.json")
    R, C = P["per_condition"], P["contrasts"]
    order = [("pick", "pick (no rival)", GREY), ("compete_R0", "rival: unspecified", ORANGE), ("compete_R1", "rival: pricing team", ORANGE),
             ("compete_R2", "rival: AI agent", ORANGE), ("compete_R3", "rival: same AI model", ORANGE),
             ("guess", "guess an AI agent's tier", GREEN), ("coord", "coordinate (control)", BLUE)]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    dots(axs[0], [(lab, R[k]["mean_tier"], col) for k, lab, col in order], (1.8, 3.0))
    axs[0].set_title("Pricing pilot · competition lowers prices")
    axs[0].set_xlabel("mean price tier (1 = cheapest of 5)")
    cons = [("unspecified − team", "R0-R1"), ("AI agent − team", "R2-R1"), ("same model − team", "R3-R1")]
    dots(axs[1], [(lab, C[k]["dTier"], ORANGE) for lab, k in cons], (-0.05, 0.25), ref=0, fmt="{:+.2f}")
    axs[1].set_title("AI rivals raise prices…")
    axs[1].set_xlabel("Δ mean tier vs pricing-team rival")
    dots(axs[2], [(lab, C[k]["dAbar"], ORANGE) for lab, k in cons], (-0.03, 0.08), ref=0, fmt="{:+.3f}")
    axs[2].set_title("…and pull agents together")
    axs[2].set_xlabel("Δ Ā vs pricing-team rival")
    fig.tight_layout()
    fig.savefig(OUT / "pricing.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in (fig_convergence, fig_geometry, fig_steering, fig_detector, fig_pricing):
        f()
        print("wrote", f.__name__)
