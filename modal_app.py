"""Modal entry points. Weights and assets live on a persistent volume; results on another.

    modal run --detach modal_app.py::download_weights      # once, CPU only
"""
import modal

app = modal.App("schelling")

weights_vol = modal.Volume.from_name("schelling-weights", create_if_missing=True)
results_vol = modal.Volume.from_name("schelling-results", create_if_missing=True)
hf_secret = modal.Secret.from_name("personas_spar25")

WEIGHTS = "/weights"
RESULTS = "/results"

dl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]==0.36.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODELS = ["Qwen/Qwen3-8B", "Qwen/Qwen3-32B"]
AXIS_REPO = "lu-christina/assistant-axis-vectors"


AXIS_DIR = f"{WEIGHTS}/assistant-axis-vectors/qwen-3-32b"
cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.6.0", "numpy")

PROJ = "/root/proj"
AXIS_SHA = "a98961956072224eaf244eb289d6c01700b63795"
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0", "transformers==4.57.6", "accelerate", "numpy", "scipy", "pandas", "pyarrow", "pyyaml",
        "huggingface_hub==0.36.0", "safetensors", "scikit-learn", "tqdm", "plotly", "matplotlib", "pytest",
    )
    .run_commands(f"pip install --no-deps git+https://github.com/safety-research/assistant-axis@{AXIS_SHA}")
    .env({"HF_HOME": f"{WEIGHTS}/hf_cache", "PYTHONPATH": PROJ, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src", f"{PROJ}/src")
    .add_local_dir("tests", f"{PROJ}/tests")
    .add_local_dir("configs", f"{PROJ}/configs")
    .add_local_dir("data", f"{PROJ}/data")
    .add_local_dir("results/assets", f"{PROJ}/results/assets")
    .add_local_file("pyproject.toml", f"{PROJ}/pyproject.toml")
)


@app.function(image=gpu_image, gpu="L4", volumes={WEIGHTS: weights_vol}, secrets=[hf_secret], timeout=3600)
def run_tests(model: str = "Qwen/Qwen3-0.6B", dtype: str = "float32", k: str = "", files: str = "tests"):
    """Unit tests in the cloud (Brief 9.3). 0.6B in fp32 for exact checks; 8B in bf16 as a sanity pass."""
    import os
    import subprocess

    env = {**os.environ, "SCHELLING_TEST_MODEL": model, "SCHELLING_TEST_DTYPE": dtype}
    cmd = ["python", "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider", *files.split(","), "--durations=8"]
    if k:
        cmd += ["-k", k]
    r = subprocess.run(cmd, cwd=PROJ, env=env, capture_output=True, text=True)
    weights_vol.commit()  # keep the HF cache for the next run
    return r.returncode, r.stdout[-30000:], r.stderr[-4000:]


@app.local_entrypoint()
def tests(model: str = "Qwen/Qwen3-0.6B", dtype: str = "float32", k: str = "", files: str = "tests"):
    code, out, err = run_tests.remote(model, dtype, k, files)
    print(out)
    if code:
        print(err)
    print("exit code", code)


@app.function(image=cpu_image, volumes={WEIGHTS: weights_vol}, timeout=1800, cpu=4, memory=8192)
def inspect_assets(layer: int = 32):
    """CPU only: axis / capping-config structure, capping sign convention, role projections."""
    import glob
    import os

    import torch

    out: dict = {"files": sorted(os.listdir(AXIS_DIR))}
    axis = torch.load(f"{AXIS_DIR}/assistant_axis.pt", map_location="cpu", weights_only=False)
    axis = axis["axis"] if isinstance(axis, dict) else axis
    axis = axis.float()
    out["axis_shape"] = list(axis.shape)
    out["axis_norm_per_layer"] = axis.norm(dim=1).tolist()

    cfg = torch.load(f"{AXIS_DIR}/capping_config.pt", map_location="cpu", weights_only=False)
    out["capping_experiment_ids"] = [e["id"] for e in cfg["experiments"]]
    exp = next(e for e in cfg["experiments"] if e["id"] == "layers_46:54-p0.25")
    caps = []
    for iv in exp["interventions"]:
        vd = cfg["vectors"][iv["vector"]]
        v = vd["vector"].float()
        L = vd["layer"]
        a = axis[L]
        caps.append({
            "vector": iv["vector"], "layer": L, "cap": float(iv["cap"]),
            "cos_with_axis": float(v @ a / (v.norm() * a.norm())), "vec_norm": float(v.norm()),
            "other_keys": sorted(k for k in iv if k not in ("vector", "cap")),
        })
    out["capping_released"] = caps

    default = torch.load(f"{AXIS_DIR}/default_vector.pt", map_location="cpu", weights_only=False)
    default = (default["vector"] if isinstance(default, dict) and "vector" in default else default)
    default = default.float() if torch.is_tensor(default) else default
    out["default_type"] = str(type(default)) + (str(list(default.shape)) if torch.is_tensor(default) else "")

    a_hat = axis[layer] / axis[layer].norm()
    proj = {}
    for f in sorted(glob.glob(f"{AXIS_DIR}/role_vectors/*.pt")):
        rv = torch.load(f, map_location="cpu", weights_only=False)
        rv = rv["vector"] if isinstance(rv, dict) and "vector" in rv else rv
        rv = rv.float()
        proj[os.path.basename(f)[:-3]] = float(rv[layer] @ a_hat)
    out["role_projection_layer"] = layer
    out["role_projections"] = proj
    if torch.is_tensor(default):
        out["default_projection"] = float(default[layer] @ a_hat)
    return out


@app.function(image=dl_image, volumes={WEIGHTS: weights_vol}, secrets=[hf_secret], timeout=3 * 3600, cpu=4)
def download_weights():
    import os
    from huggingface_hub import snapshot_download

    token = os.environ["HF_TOKEN"]
    snapshot_download(
        AXIS_REPO, repo_type="dataset", allow_patterns=["qwen-3-32b/*"],
        local_dir=f"{WEIGHTS}/assistant-axis-vectors", token=token,
    )
    weights_vol.commit()
    print("axis assets done", flush=True)
    for repo in MODELS:
        snapshot_download(
            repo, local_dir=f"{WEIGHTS}/{repo}", token=token,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
        )
        weights_vol.commit()
        print(f"{repo} done", flush=True)


@app.local_entrypoint()
def assets():
    import json
    import pathlib

    out = inspect_assets.remote()
    p = pathlib.Path("results/assets/qwen3-32b_assets.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k not in ("role_projections", "axis_norm_per_layer")})
    print("roles:", len(out["role_projections"]), "->", p)


# ------------------------------------------------------------------ phases
RESULTS_REPO = "soumyadeepbose/schelling-results"  # private HF dataset


def _push(subdir: str):
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=RESULTS_REPO, repo_type="dataset", folder_path=f"{RESULTS}/{subdir}", path_in_repo=subdir)


@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=2 * 3600)
def phase0(n_gof: int = 96, n_jac: int = 8):
    """Dev model on the main GPU type: throughput, Jacobian timing, readout goodness-of-fit gate."""
    import json
    import pathlib
    import random
    import time

    from src import run
    from src.jacobian import option_jacobian
    from src.readout_hf import Item, Scorer
    from src import prompts

    out = pathlib.Path(f"{RESULTS}/qwen3-8b/phase0")
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-8B")
    load_s = time.time() - t
    sc = Scorer(model, tok, max_batch_tokens=16384)
    personas = run.load_personas(["assistant"])
    cells = run.phase_cells(personas, ["F1", "F2", "F3", "F4", "F5", "F2b"], ["pick", "coord"], n_orders=2)
    meta = {"model": "Qwen3-8B", "engine": "hf", "layer_cfg": "", "intervention": "none", "dose": 0.0}
    tp = run.score_cells(sc, cells, meta, out / "dist.parquet")
    rng = random.Random(0)
    jac_cells = rng.sample(cells, n_jac)
    t = time.time()
    for c in jac_cells:
        ids, P = prompts.render(tok, c.system, c.user)
        option_jacobian(sc, Item(ids, P, c.options), layer=18)
    jac_s = (time.time() - t) / n_jac
    gof = run.gof_gate(sc, rng.sample(cells, n_gof))
    summary = {"load_s": load_s, "throughput": tp, "jacobian_s_per_prompt": jac_s, "gof": run.summarize_gof(gof)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "gof.json").write_text(json.dumps(gof))
    results_vol.commit()
    return summary


@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=4 * 3600)
def phase1(gof_frac: float = 0.01, families: str = "F1,F2,F3,F4,F5,F1b,F2b,F4b"):
    """Assistant, pick/guess/coord on every family (+anti on F1, F3), 4 orders, 2x2 paraphrases."""
    import json
    import pathlib
    import random
    import time

    from src import run
    from src.readout_hf import Scorer

    out = pathlib.Path(f"{RESULTS}/qwen3-32b/phase1")
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-32B")
    load_s = time.time() - t
    sc = Scorer(model, tok, max_batch_tokens=8192)
    personas = run.load_personas(["assistant"])
    cells = run.phase_cells(personas, families.split(","), ["pick", "guess", "coord"], n_orders=4)
    meta = {"model": "Qwen3-32B", "engine": "hf", "layer_cfg": "", "intervention": "none", "dose": 0.0}
    tp = run.score_cells(sc, cells, meta, out / "dist.parquet")
    results_vol.commit()
    rng = random.Random(1)
    gof = run.gof_gate(sc, rng.sample(cells, max(20, int(gof_frac * len(cells)))))
    summary = {"load_s": load_s, "throughput": tp, "gof": run.summarize_gof(gof)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "gof.json").write_text(json.dumps(gof))
    results_vol.commit()
    _push("qwen3-32b/phase1")
    return summary


@app.local_entrypoint()
def run_phase(phase: str = "0"):
    import json

    fn = {"0": phase0, "1": phase1}[phase]
    print(json.dumps(fn.remote(), indent=1))


@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=3600)
def precision_check(n_cells: int = 300, n_uncached: int = 40):
    """bf16 path noise against an fp32 reference on Qwen3-8B: per-cell TV, log-prob error, effect on C00."""
    import json
    import random

    import numpy as np
    import torch

    from src import prompts, run
    from src.readout_hf import Item, Scorer, score_uncached

    personas = run.load_personas(["assistant"])
    cells = run.phase_cells(personas, ["F1", "F2", "F3", "F4", "F5"], ["pick", "coord"], n_orders=2)
    cells = random.Random(0).sample(cells, n_cells)
    out = {}
    for dtype in ["float32", "bfloat16"]:
        model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-8B", dtype=dtype)
        items = [Item(*prompts.render(tok, c.system, c.user), c.options) for c in cells]
        out[dtype] = {"batched": [r.logprobs for r in Scorer(model, tok).score(items)],
                      "single": [r.logprobs for r in Scorer(model, tok, max_batch_size=1).score(items[:n_uncached])],
                      "uncached": [np.logaddexp.reduce(np.array(row), axis=1) for row in
                                   (score_uncached(model, tok, it) for it in items[:n_uncached])]}
        del model
        torch.cuda.empty_cache()

    def sm(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    ref = out["float32"]["batched"]
    res = {}
    for name, arrs in [("bf16_batched", out["bfloat16"]["batched"]), ("bf16_single", out["bfloat16"]["single"]),
                       ("bf16_uncached", out["bfloat16"]["uncached"]), ("fp32_uncached", out["float32"]["uncached"])]:
        tv, dlp, dc = [], [], []
        for z, zr in zip(arrs, ref):
            p, pr = sm(z), sm(zr)
            tv.append(0.5 * np.abs(p - pr).sum())
            m = pr > 0.01
            dlp.append(np.abs(z - zr)[m].max())
            dc.append(p @ p - pr @ pr)
        tv, dlp, dc = map(np.array, (tv, dlp, dc))
        res[name] = {"n": len(tv), "tv_median": float(np.median(tv)), "tv_p95": float(np.quantile(tv, .95)), "tv_max": float(tv.max()),
                     "dlogp_median": float(np.median(dlp)), "dlogp_p95": float(np.quantile(dlp, .95)),
                     "dC00_mean": float(dc.mean()), "dC00_abs_median": float(np.median(np.abs(dc))), "dC00_abs_p95": float(np.quantile(np.abs(dc), .95))}
    with open(f"{RESULTS}/qwen3-8b/phase0/precision.json", "w") as f:
        json.dump(res, f, indent=1)
    results_vol.commit()
    return res


@app.local_entrypoint()
def precision():
    import json

    print(json.dumps(precision_check.remote(), indent=1))


# ------------------------------------------------------------------ Phase 2
P2_LAYER = 32
P2_EXTRA = list(range(46, 54))  # released capping layers 46..53


@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=4 * 3600)
def phase2_shard(personas: list[str], shard: int, n_gof: int = 24, max_instances: int = 0, subdir: str = "phase2"):
    """All four strata (Part B): pick/guess/coord on F1-F5, anti on F1/F3, 4 orders, 2x2 paraphrases."""
    import json
    import pathlib
    import random
    import time

    import torch

    from src import run
    from src.readout_hf import Scorer

    out = pathlib.Path(f"{RESULTS}/qwen3-32b/{subdir}")
    out.mkdir(parents=True, exist_ok=True)
    axis = torch.load(f"{AXIS_DIR}/assistant_axis.pt", map_location="cpu", weights_only=False)
    axis = axis["axis"] if isinstance(axis, dict) else axis
    t = time.time()
    model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-32B")
    load_s = time.time() - t
    sc = Scorer(model, tok, max_batch_tokens=8192)
    allp = run.load_personas()
    sel = {k: allp[k] for k in personas}
    strata = {k: v["stratum"] for k, v in sel.items()}
    cells = run.phase_cells(sel, ["F1", "F2", "F3", "F4", "F5"], ["pick", "guess", "coord"], n_orders=4,
                            max_instances=max_instances or None)
    meta = {"model": "Qwen3-32B", "engine": "hf", "layer_cfg": "", "intervention": "none", "dose": 0.0}
    tp = run.score_cells(sc, cells, meta, out / f"dist_shard{shard}.parquet", strata=strata, axis=axis,
                         act_layer=P2_LAYER, extra_layers=P2_EXTRA,
                         extra_keep=lambda c: c.order_id == 0 and c.persona_par == 0 and c.frame_par == 0)
    results_vol.commit()
    gof = run.gof_gate(sc, random.Random(100 + shard).sample(cells, n_gof), seed=1000 * shard)
    summary = {"shard": shard, "personas": personas, "load_s": load_s, "throughput": tp, "gof": run.summarize_gof(gof)}
    (out / f"summary_shard{shard}.json").write_text(json.dumps(summary, indent=1))
    (out / f"gof_shard{shard}.json").write_text(json.dumps(gof))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def run_phase2(n_shards: int = 6):
    import json

    import yaml

    names = list(yaml.safe_load(open("configs/personas.yaml", encoding="utf-8")))
    shards = [names[i::n_shards] for i in range(n_shards)]
    for s in phase2_shard.starmap([(p, i) for i, p in enumerate(shards)]):
        print(json.dumps({k: s[k] for k in ("shard", "personas", "throughput", "gof")}, indent=1))


@app.local_entrypoint()
def pilot2(max_instances: int = 4):
    """Phase 2 pilot: all personas, all conditions, a few instances per family, one GPU. Not confirmatory."""
    import json

    import yaml

    names = list(yaml.safe_load(open("configs/personas.yaml", encoding="utf-8")))
    s = phase2_shard.remote(names, 0, 12, max_instances, "phase2_pilot")
    print(json.dumps({k: s[k] for k in ("load_s", "throughput", "gof")}, indent=1))


@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=8, memory=32768, timeout=3600)
def analyze2_remote(subdir: str = "phase2"):
    import json

    from src.analysis_phase2 import analyze

    rep = analyze(f"{RESULTS}/qwen3-32b/{subdir}", exclude_pilot=(subdir == "phase2"))
    with open(f"{RESULTS}/qwen3-32b/{subdir}/analysis.json", "w") as f:
        json.dump(rep, f, default=float)
    results_vol.commit()
    return json.loads(json.dumps(rep, default=float))


@app.local_entrypoint()
def analyze2(subdir: str = "phase2"):
    import json
    import pathlib

    rep = analyze2_remote.remote(subdir)
    p = pathlib.Path(f"results/qwen3-32b/{subdir}/analysis.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep))
    print(json.dumps({k: rep[k] for k in ("manipulation", "A1", "H2", "sanity")}, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_persona"} for k, v in rep["A2"].items()}, indent=1)[:6000])


@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=8, memory=32768, timeout=3600)
def amend3_remote(with_2b: bool = False):
    import json

    from src.analysis_amend3 import analyze

    base = f"{RESULTS}/qwen3-32b"
    rep = {f"T{T}": analyze(f"{base}/phase2", f"{base}/phase2b" if with_2b else None, T=T) for T in (1.0, 0.3)}
    with open(f"{base}/amend3{'_2b' if with_2b else '_partA'}.json", "w") as f:
        json.dump(rep, f, default=float)
    results_vol.commit()
    return json.loads(json.dumps(rep, default=float))


@app.local_entrypoint()
def amend3(with_2b: bool = False):
    import json
    import pathlib

    rep = amend3_remote.remote(with_2b)
    p = pathlib.Path(f"results/qwen3-32b/amend3{'_2b' if with_2b else '_partA'}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep))
    print("saved", p)


@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=8, memory=32768, timeout=1800)
def flatten_check_remote():
    """Exploratory (not pre-registered): is anti-frame homogenisation an artefact of flattening toward uniform?"""
    import glob

    import numpy as np
    import pandas as pd
    import yaml

    from src.analysis_phase2 import per_order, pilot_instances
    from src.prompts import ROOT

    base = f"{RESULTS}/qwen3-32b/phase2"
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/dist_shard*.parquet"))], ignore_index=True)
    df = df[~df.instance_id.isin(pilot_instances()) & df.family.isin(["F1", "F3"])]
    pcfg = yaml.safe_load((ROOT / "configs/personas.yaml").read_text(encoding="utf-8"))
    subst = [p for p in pcfg if pcfg[p]["stratum"] in ("assistant", "stratified", "theory", "realistic") and p != "facilitator"]
    df = df[df.persona.isin(subst)]
    PO = per_order(df)
    rng = np.random.default_rng(0)
    out = {}
    insts = sorted({(k[1], k[2]) for k in PO})
    per = {fr: [] for fr in ("pick", "coord", "anti")}
    for fr in per:
        for f, iid in insts:
            P = np.stack([PO[(p, f, iid, fr)].mean(0) for p in subst])       # K x n
            n = P.shape[1]
            Z = P - 1.0 / n
            nz = np.linalg.norm(Z, axis=1)
            Cc = (Z @ Z.T) / np.outer(nz, nz)
            Cr = (P @ P.T) / np.outer(np.linalg.norm(P, axis=1), np.linalg.norm(P, axis=1))
            m = ~np.eye(len(subst), dtype=bool)
            per[fr].append({"centred_cos": Cc[m].mean(), "raw_cos": Cr[m].mean(), "selfC": (P ** 2).sum(1).mean(),
                            "dist_from_uniform": nz.mean(), "top_share_agree": np.mean(P.argmax(1)[:, None] == P.argmax(1)[None, :])})
    for fr, rows in per.items():
        d = pd.DataFrame(rows)
        out[fr] = {}
        for c in d.columns:
            v = d[c].values
            dr = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
            out[fr][c] = {"est": float(v.mean()), "lo": float(np.quantile(dr, .025)), "hi": float(np.quantile(dr, .975))}
    for a, b in (("coord", "pick"), ("anti", "pick"), ("coord", "anti")):
        for c in ("centred_cos", "top_share_agree"):
            v = pd.DataFrame(per[a])[c].values - pd.DataFrame(per[b])[c].values
            dr = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
            out[f"{a}-{b} {c}"] = {"est": float(v.mean()), "lo": float(np.quantile(dr, .025)), "hi": float(np.quantile(dr, .975))}
    return out


@app.local_entrypoint()
def flatten_check():
    import json
    import pathlib

    r = flatten_check_remote.remote()
    pathlib.Path("results/qwen3-32b/flatten_check.json").write_text(json.dumps(r))
    for k, v in r.items():
        print(k, {kk: f"{vv['est']:+.3f} [{vv['lo']:+.3f}, {vv['hi']:+.3f}]" for kk, vv in v.items()} if "est" not in v else f"{v['est']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")


# ------------------------------------------------------------------ Phase 2b (PREREG Amendment 3)
@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=4 * 3600)
def phase2b_shard(personas: list[str], shard: int, n_gof: int = 24):
    """Coordinate frame with partner descriptions D2 and D3; confirmatory instances only."""
    import json
    import pathlib
    import random
    import time

    import torch

    from src import run
    from src.analysis_phase2 import pilot_instances
    from src.readout_hf import Scorer

    out = pathlib.Path(f"{RESULTS}/qwen3-32b/phase2b")
    out.mkdir(parents=True, exist_ok=True)
    axis = torch.load(f"{AXIS_DIR}/assistant_axis.pt", map_location="cpu", weights_only=False)
    axis = axis["axis"] if isinstance(axis, dict) else axis
    t = time.time()
    model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-32B")
    load_s = time.time() - t
    sc = Scorer(model, tok, max_batch_tokens=8192)
    allp = run.load_personas()
    sel = {k: allp[k] for k in personas}
    cells = run.phase_cells(sel, ["F1", "F2", "F3", "F4", "F5"], ["coord_d2", "coord_d3"], n_orders=4,
                            anti_families=(), exclude_instances=pilot_instances())
    meta = {"model": "Qwen3-32B", "engine": "hf", "layer_cfg": "", "intervention": "none", "dose": 0.0}
    tp = run.score_cells(sc, cells, meta, out / f"dist_shard{shard}.parquet",
                         strata={k: v["stratum"] for k, v in sel.items()}, axis=axis, act_layer=P2_LAYER)
    results_vol.commit()
    gof = run.gof_gate(sc, random.Random(200 + shard).sample(cells, n_gof), seed=2000 * shard)
    summary = {"shard": shard, "personas": personas, "load_s": load_s, "throughput": tp, "gof": run.summarize_gof(gof)}
    (out / f"summary_shard{shard}.json").write_text(json.dumps(summary, indent=1))
    (out / f"gof_shard{shard}.json").write_text(json.dumps(gof))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def run_phase2b(n_shards: int = 4):
    import json

    import yaml

    cfg = yaml.safe_load(open("configs/personas.yaml", encoding="utf-8"))
    names = [k for k in cfg if k != "facilitator"]
    shards = [names[i::n_shards] for i in range(n_shards)]
    for s in phase2b_shard.starmap([(p, i) for i, p in enumerate(shards)]):
        print(json.dumps({k: s[k] for k in ("shard", "personas", "throughput", "gof")}, indent=1))


@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=8, memory=32768, timeout=3600)
def amend3_selftest_remote():
    """Part B code check: D2 = D3 = copies of D1 (coord) must give zero differences."""
    import glob
    import os

    import pandas as pd

    from src.analysis_amend3 import analyze

    base = f"{RESULTS}/qwen3-32b"
    fake = "/tmp/fake2b"
    os.makedirs(fake, exist_ok=True)
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/phase2/dist_shard*.parquet"))], ignore_index=True)
    c = df[df.frame == "coord"]
    pd.concat([c.assign(frame="coord_d2"), c.assign(frame="coord_d3")]).to_parquet(f"{fake}/dist_shard0.parquet")
    rep = analyze(f"{base}/phase2", fake, T=1.0)
    B = rep["partB"]
    return {k: {kk: B[k][kk] for kk in ("a1_dAbar", "a2_dRetention", "b1_dS", "b2_median_beta_minus_1", "c_d_mean")} for k in ("D3", "D2")} | {"ref": B["reference_D1"], "mechanism": B["mechanism"]}


@app.local_entrypoint()
def amend3_selftest():
    import json

    print(json.dumps(amend3_selftest_remote.remote(), indent=1))


@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=8, memory=32768, timeout=3600)
def amend4_remote():
    import json

    from src.analysis_amend4 import analyze

    base = f"{RESULTS}/qwen3-32b"
    rep = {f"T{T}": analyze(f"{base}/phase2", f"{base}/phase2b", T=T) for T in (1.0, 0.3)}
    with open(f"{base}/amend4.json", "w") as f:
        json.dump(rep, f, default=float)
    results_vol.commit()
    return json.loads(json.dumps(rep, default=float))


@app.local_entrypoint()
def amend4():
    import json
    import pathlib

    rep = amend4_remote.remote()
    pathlib.Path("results/qwen3-32b/amend4.json").write_text(json.dumps(rep))
    print("saved")


# ------------------------------------------------------------------ Amendment 5 Step 0: residual inventory
@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=4, memory=32768, timeout=1800)
def residual_inventory_remote():
    import glob

    import numpy as np
    import pandas as pd

    from src.analysis_phase2 import pilot_instances

    pil = set(pilot_instances())
    out = {}
    for sub in ("phase2", "phase2b"):
        rows = []
        for f in sorted(glob.glob(f"{RESULTS}/qwen3-32b/{sub}/dist_shard*.parquet")):
            d = pd.read_parquet(f, columns=["persona", "frame", "family", "instance_id", "order_id", "persona_par", "frame_par"])
            a = np.load(f[:-8] + "_l32.npy", mmap_mode="r")
            ex = f[:-8] + "_extra.npz"
            d["has_l32"] = True
            d["has_extra"] = False
            if glob.glob(ex):
                z = np.load(ex)
                d.loc[z["rows"], "has_extra"] = True
                exl = sorted(k for k in z.files if k != "rows")
            else:
                exl = []
            rows.append(d)
            out.setdefault(sub, {}).setdefault("shards", []).append(
                {"file": f.rsplit("/", 1)[1], "rows": len(d), "l32_shape": list(a.shape), "l32_dtype": str(a.dtype),
                 "match": len(d) == a.shape[0], "extra_layers": exl, "finite": bool(np.isfinite(a[:2000]).all())})
        d = pd.concat(rows, ignore_index=True)
        d["confirmatory"] = ~d.instance_id.isin(pil)
        g = d.groupby(["frame", "family"]).agg(cells=("persona", "size"), personas=("persona", "nunique"),
                                              insts=("instance_id", "nunique"),
                                              conf_insts=("instance_id", lambda s: s[~s.isin(pil)].nunique()),
                                              orders=("order_id", "nunique"), extra_cells=("has_extra", "sum"))
        out[sub]["by_frame_family"] = {f"{a}|{b}": {k: int(v) for k, v in r.items()} for (a, b), r in g.iterrows()}
        out[sub]["paraphrases"] = sorted(d.paraphrase_ids.unique().tolist()) if "paraphrase_ids" in d else \
            sorted((d.persona_par.astype(str) + "-" + d.frame_par.astype(str)).unique().tolist())
    return out


@app.local_entrypoint()
def residual_inventory():
    import json

    print(json.dumps(residual_inventory_remote.remote(), indent=1))


# ------------------------------------------------------------------ Amendment 5: exploratory layer sweep (capture only)
SWEEP_LAYERS = list(range(4, 64, 4)) + [63]  # every fourth layer (includes 32 as a consistency check) + last


@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=2 * 3600)
def sweep_capture_remote():
    """Decision-position residuals at SWEEP_LAYERS for order 0, paraphrase (0,0), confirmatory instances,
    the 23 substantive personas, pick/guess/coord on F1-F5 and anti on F1/F3 (same subset as the 46-53 capture)."""
    import json
    import pathlib
    import time

    import numpy as np
    import pandas as pd

    from src import run
    from src.analysis_phase2 import SUBST, pilot_instances
    from src.prompts import render
    from src.readout_hf import Item, Scorer

    out = pathlib.Path(f"{RESULTS}/qwen3-32b/amend5_sweep")
    out.mkdir(parents=True, exist_ok=True)
    allp = run.load_personas()
    sel = {k: v for k, v in allp.items() if v["stratum"] in SUBST and k != "facilitator"}
    cells = [c for c in run.phase_cells(sel, ["F1", "F2", "F3", "F4", "F5"], ["pick", "guess", "coord"], n_orders=4,
                                        exclude_instances=pilot_instances())
             if c.order_id == 0 and c.persona_par == 0 and c.frame_par == 0]
    t = time.time()
    model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-32B")
    load_s = time.time() - t
    sc = Scorer(model, tok, max_batch_tokens=16384)
    t = time.time()
    acts = np.zeros((len(cells), len(SWEEP_LAYERS), model.config.hidden_size), np.float16)
    for s in range(0, len(cells), 2048):
        part = cells[s: s + 2048]
        items = [Item(*render(tok, c.system, c.user), c.options, key=i) for i, c in enumerate(part)]
        for j, r in enumerate(sc.capture(items, SWEEP_LAYERS)):
            acts[s + j] = np.stack([r[l] for l in SWEEP_LAYERS]).astype(np.float16)
        print(f"  {s + len(part)}/{len(cells)} cells, {time.time() - t:.0f}s")
    np.save(out / "acts.npy", acts)
    pd.DataFrame([{"persona": c.persona, "frame": c.frame, "family": c.family, "instance_id": c.instance_id,
                   "order_id": c.order_id, "persona_par": c.persona_par, "frame_par": c.frame_par} for c in cells]
                 ).to_parquet(out / "index.parquet")
    summary = {"cells": len(cells), "layers": SWEEP_LAYERS, "load_s": load_s, "seconds": time.time() - t}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def sweep_capture():
    import json

    print(json.dumps(sweep_capture_remote.remote(), indent=1))


@app.function(image=gpu_image, volumes={RESULTS: results_vol, WEIGHTS: weights_vol}, cpu=8, memory=65536, timeout=4 * 3600)
def amend5_remote():
    import json

    import torch

    from src.analysis_amend5 import analyze

    axis = torch.load(f"{AXIS_DIR}/assistant_axis.pt", map_location="cpu", weights_only=False)
    axis = axis["axis"] if isinstance(axis, dict) else axis
    base = f"{RESULTS}/qwen3-32b"
    rep = analyze(f"{base}/phase2", f"{base}/amend5_sweep", axis[32].float().numpy(), log=lambda m: print(m, flush=True))
    with open(f"{base}/amend5.json", "w") as f:
        json.dump(rep, f, default=float)
    results_vol.commit()
    return json.loads(json.dumps(rep, default=float))


@app.local_entrypoint()
def amend5():
    import json
    import pathlib

    rep = amend5_remote.remote()
    pathlib.Path("results/qwen3-32b/amend5.json").write_text(json.dumps(rep))
    print("saved")


# ------------------------------------------------------------------ Pricing pilot (exploratory, not pre-registered)
@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=2 * 3600)
def pricing_pilot_remote(n_gof: int = 24):
    import json
    import pathlib
    import random
    import time

    import torch

    from src import pricing, run
    from src.readout_hf import Scorer

    out = pathlib.Path(f"{RESULTS}/qwen3-32b/pricing_pilot")
    out.mkdir(parents=True, exist_ok=True)
    axis = torch.load(f"{AXIS_DIR}/assistant_axis.pt", map_location="cpu", weights_only=False)
    axis = axis["axis"] if isinstance(axis, dict) else axis
    t = time.time()
    model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-32B")
    load_s = time.time() - t
    sc = Scorer(model, tok, max_batch_tokens=8192)
    sel = run.load_personas(pricing.load_cfg()["agents"])
    cells = pricing.build_cells({k: v["prompts"] for k, v in sel.items()})
    meta = {"model": "Qwen3-32B", "engine": "hf", "layer_cfg": "", "intervention": "none", "dose": 0.0}
    tp = run.score_cells(sc, cells, meta, out / "dist_shard0.parquet", strata={k: v["stratum"] for k, v in sel.items()},
                         axis=axis, act_layer=32)
    results_vol.commit()
    gof = run.gof_gate(sc, random.Random(300).sample(cells, n_gof), seed=3000)
    summary = {"load_s": load_s, "throughput": tp, "gof": run.summarize_gof(gof)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "gof.json").write_text(json.dumps(gof))
    results_vol.commit()
    return summary


@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=8, memory=32768, timeout=3600)
def pricing_analyze_remote():
    import json

    from src.analysis_pricing import analyze, generic_ai_direction

    base = f"{RESULTS}/qwen3-32b"
    g = generic_ai_direction(f"{base}/phase2")
    rep = analyze(f"{base}/pricing_pilot", g)
    with open(f"{base}/pricing_pilot.json", "w") as f:
        json.dump(rep, f, default=float)
    results_vol.commit()
    return json.loads(json.dumps(rep, default=float))


@app.local_entrypoint()
def pricing_pilot(score: bool = True):
    import json
    import pathlib

    if score:
        print(json.dumps(pricing_pilot_remote.remote(), indent=1))
    rep = pricing_analyze_remote.remote()
    pathlib.Path("results/qwen3-32b/pricing_pilot.json").write_text(json.dumps(rep))
    print("saved")


# ------------------------------------------------------------------ Amendment 6: steering along g
A6 = f"{RESULTS}/qwen3-32b/amend6"


@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=4, memory=32768, timeout=3600)
def amend6_directions_remote():
    import os

    from src.amend6 import directions, save_directions

    os.makedirs(A6, exist_ok=True)
    vec, info = directions(f"{RESULTS}/qwen3-32b/phase2")
    save_directions(vec, info, A6)
    results_vol.commit()
    return {k: v for k, v in info.items() if k != "split"}


@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=3600)
def amend6_bench_remote(n_equiv: int = 48, n_tp: int = 230):
    """Steering equivalence (prompt reuse vs reference path, bf16) and throughput. Half-A cells, random directions only."""
    import json
    import random
    import time

    import numpy as np
    import torch

    from src import amend6, run
    from src.prompts import render
    from src.readout_hf import Item, Scorer

    vec = np.load(f"{A6}/directions.npz")
    info = json.load(open(f"{A6}/directions.json"))
    u, u53 = info["u"], info["u53"]
    t = time.time()
    model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-32B")
    load_s = time.time() - t
    sc = Scorer(model, tok, max_batch_tokens=8192)
    R = torch.tensor(vec["random"], device="cuda")
    item = lambda c: Item(*render(tok, c.system, c.user), c.options)

    # ---------------- equivalence (criterion fixed before running: every delta median TV <= 0.02 and q95 <= 0.05)
    pool = amend6.cells(("pick", "coord"), "A")
    eq = random.Random(61).sample(pool, n_equiv)
    items = [item(c) for c in eq]
    deltas = [None, (32, u * R[0]), (32, 2 * u * R[0]), (53, 2 * u53 * R[1])]
    names = ["dose 0", "random 1u @32", "random 2u @32", "random 2u53 @53"]
    fast = sc.score_steered(items, deltas)
    dist = lambda r: np.append(np.exp(r.logprobs), max(r.offmenu, 0.0))
    equiv = {}
    for k, dl in enumerate(deltas):
        sc.steer = dl
        ref = sc.score(items)
        sc.steer = None
        tv = [0.5 * np.abs(dist(fast[i][k]) - dist(ref[i])).sum() for i in range(len(items))]
        dlp = [float(np.max(np.abs(fast[i][k].logprobs - ref[i].logprobs)[np.exp(ref[i].logprobs) > 1e-3])) for i in range(len(items))]
        eff = [0.5 * np.abs(dist(fast[i][k]) - dist(fast[i][0])).sum() for i in range(len(items))]
        equiv[names[k]] = {"tv_median": float(np.median(tv)), "tv_q95": float(np.quantile(tv, .95)), "tv_max": float(np.max(tv)),
                           "max_dlogp_median": float(np.median(dlp)), "max_dlogp_max": float(np.max(dlp)),
                           "steering_effect_tv_median": float(np.median(eff)), "offmenu_mean": float(np.mean([fast[i][k].offmenu for i in range(len(items))])),
                           "pass": bool(np.median(tv) <= 0.02 and np.quantile(tv, .95) <= 0.05)}
    equiv_pass = all(v["pass"] for v in equiv.values())

    # ---------------- throughput: per-prompt cost of the prompt pass and of each steered evaluation
    bench = {"load_s": load_s}
    for fr in ("pick", "coord"):
        cs = random.Random(62).sample(amend6.cells((fr,), "A"), n_tp)
        its = [item(c) for c in cs]
        sc.score_steered(its[:8], [None])  # warm-up
        times = {}
        for D in (1, 13):
            dl = [None] + [(32, (0.5 + 0.25 * j) * u * R[j % 5]) for j in range(D - 1)]
            torch.cuda.synchronize()
            t = time.time()
            sc.score_steered(its, dl)
            torch.cuda.synchronize()
            times[D] = time.time() - t
        c_delta = (times[13] - times[1]) / (12 * n_tp)
        bench[fr] = {"t1": times[1], "t13": times[13], "c_delta": c_delta, "c_prefix": times[1] / n_tp - c_delta}
    t = time.time()
    run.gof_gate(sc, random.Random(63).sample(pool, 4), seed=6000, steer=(32, u * R[2]))
    bench["audit_s"] = (time.time() - t) * 6
    pred = {I: amend6.predicted_cost(bench, I) for I in (18, 14, 10)}
    rep = {"equivalence": equiv, "equivalence_pass": equiv_pass, "bench": bench, "predicted": pred,
           "u": u, "u53": u53}
    with open(f"{A6}/bench.json", "w") as f:
        json.dump(rep, f, indent=1, default=float)
    results_vol.commit()
    return json.loads(json.dumps(rep, default=float))


@app.local_entrypoint()
def amend6_bench():
    import json
    import pathlib

    info = amend6_directions_remote.remote()
    print(json.dumps(info, indent=1))
    rep = amend6_bench_remote.remote()
    pathlib.Path("results/qwen3-32b/amend6_bench.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=3600)
def amend6_diag_remote(n_equiv: int = 48):
    """Diagnostic A: reuse path and reference path, each against a full uncached bf16 forward, on the benchmark cells.
    Criterion (owner, before any steered outcome data): per delta, p95 TV(reuse, uncached) <= p95 TV(reference, uncached) + 0.01."""
    import json
    import random

    import numpy as np
    import torch

    from src import amend6, run
    from src.prompts import render
    from src.readout_hf import Item, Scorer, score_uncached_batched

    vec = np.load(f"{A6}/directions.npz")
    info = json.load(open(f"{A6}/directions.json"))
    u, u53 = info["u"], info["u53"]
    model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-32B")
    sc = Scorer(model, tok, max_batch_tokens=8192)
    R = torch.tensor(vec["random"], device="cuda")
    eq = random.Random(61).sample(amend6.cells(("pick", "coord"), "A"), n_equiv)       # same cells as the benchmark
    items = [Item(*render(tok, c.system, c.user), c.options) for c in eq]
    deltas = [None, (32, u * R[0]), (32, 2 * u * R[0]), (53, 2 * u53 * R[1])]
    names = ["dose 0", "random 1u @32", "random 2u @32", "random 2u53 @53"]
    fast = sc.score_steered(items, deltas)

    def dist_lp(lp):
        p = np.exp(np.asarray(lp))
        return np.append(p, max(1 - p.sum(), 0.0))

    out = {}
    for k, dl in enumerate(deltas):
        sc.steer = dl
        ref = sc.score(items)
        sc.steer = None
        unc = [dist_lp([np.logaddexp.reduce(np.array(v)) for v in score_uncached_batched(model, tok, it, steer=dl)]) for it in items]
        tv_reuse = [0.5 * np.abs(dist_lp(fast[i][k].logprobs) - unc[i]).sum() for i in range(len(items))]
        tv_ref = [0.5 * np.abs(dist_lp(ref[i].logprobs) - unc[i]).sum() for i in range(len(items))]
        tv_rr = [0.5 * np.abs(dist_lp(fast[i][k].logprobs) - dist_lp(ref[i].logprobs)).sum() for i in range(len(items))]
        q = lambda x, a: float(np.quantile(x, a))
        out[names[k]] = {"reuse_vs_uncached": {"median": q(tv_reuse, .5), "p95": q(tv_reuse, .95), "max": max(tv_reuse)},
                         "reference_vs_uncached": {"median": q(tv_ref, .5), "p95": q(tv_ref, .95), "max": max(tv_ref)},
                         "reuse_vs_reference": {"median": q(tv_rr, .5), "p95": q(tv_rr, .95), "max": max(tv_rr)},
                         "pass": bool(q(tv_reuse, .95) <= q(tv_ref, .95) + 0.01)}
    rep = {"per_delta": out, "pass": all(v["pass"] for v in out.values())}
    with open(f"{A6}/diag.json", "w") as f:
        json.dump(rep, f, indent=1, default=float)
    results_vol.commit()
    return json.loads(json.dumps(rep, default=float))


@app.local_entrypoint()
def amend6_diag():
    import json
    import pathlib

    rep = amend6_diag_remote.remote()
    pathlib.Path("results/qwen3-32b/amend6_diag.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=4 * 3600)
def amend6_shard(personas: list[str], shard: int, n_per_family: int = 18, audit: bool = False, half: str = "B", tag: str = ""):
    """Steered readouts on held-out (half B) cells, orders 1-2, paraphrase (0,0), via prompt reuse (dose 0 included).
    pick: dose 0, +g (S1), +m53 at layer 53 (S3), +random (S1 control); coord: dose 0, -g (S2), -random (S2 control)."""
    import json
    import random
    import time

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from src import amend6, run
    from src.prompts import render
    from src.readout_hf import Item, Scorer

    vec = np.load(f"{A6}/directions.npz")
    info = json.load(open(f"{A6}/directions.json"))
    u, u53 = info["u"], info["u53"]
    g = torch.tensor(vec["ghat"], device="cuda")
    m53 = torch.tensor(vec["m53hat"], device="cuda")
    R = torch.tensor(vec["random"], device="cuda")
    t = time.time()
    model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-32B")
    load_s = time.time() - t
    sc = Scorer(model, tok, max_batch_tokens=8192)
    plan = {"pick": [("dose0", "none", 0.0, None)]
                    + [("S1", "g", d, (32, d * u * g)) for d in amend6.DOSES]
                    + [("S3", "m53", d, (53, d * u53 * m53)) for d in amend6.DOSES]
                    + [("S1", f"r{k}", d, (32, d * u * R[k])) for k in range(amend6.N_RANDOM) for d in amend6.DOSES],
            "coord": [("dose0", "none", 0.0, None)]
                     + [("S2", "g", d, (32, -d * u * g)) for d in amend6.DOSES]
                     + [("S2", f"r{k}", d, (32, -d * u * R[k])) for k in range(amend6.N_RANDOM) for d in amend6.DOSES]}
    t = time.time()
    writer = None
    out = f"{A6}/{tag}steer_shard{shard}.parquet"
    for fr, conds in plan.items():
        cs = [c for c in amend6.cells((fr,), half, n_per_family) if c.persona in personas]
        for s in range(0, len(cs), 1024):
            part = cs[s: s + 1024]
            res = sc.score_steered([Item(*render(tok, c.system, c.user), c.options) for c in part], [x[3] for x in conds])
            recs = [{"persona": c.persona, "frame": c.frame, "family": c.family, "instance_id": c.instance_id,
                     "order_id": c.order_id, "part": cd[0], "direction": cd[1], "dose": cd[2],
                     "logprobs": r[k].logprobs.tolist(), "offmenu": r[k].offmenu}
                    for c, r in zip(part, res) for k, cd in enumerate(conds)]
            tbl = pa.Table.from_pylist(recs)
            writer = writer or pq.ParquetWriter(out, tbl.schema)
            writer.write_table(tbl)
            print(f"  {fr} {s + len(part)}/{len(cs)} prompts, {time.time() - t:.0f}s", flush=True)
    writer.close()
    summary = {"shard": shard, "personas": personas, "load_s": load_s, "seconds": time.time() - t}
    if audit:
        cs = amend6.cells(("pick",), half, n_per_family)
        gof = run.gof_gate(sc, random.Random(64).sample(cs, min(24, len(cs))), seed=6400, steer=(32, 1.0 * u * g))
        summary["audit_g1u"] = run.summarize_gof(gof)
        with open(f"{A6}/{tag}audit_g1u.json", "w") as f:
            json.dump(gof, f)
    with open(f"{A6}/{tag}summary_shard{shard}.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    results_vol.commit()
    return summary


@app.local_entrypoint()
def amend6_run(n_shards: int = 4):
    import json

    from src.analysis_amend5 import substantive

    subst, stratum = substantive()
    order = sorted(subst, key=lambda p: (stratum[p], p))
    shards = [order[k::n_shards] for k in range(n_shards)]
    for s in amend6_shard.starmap([(p, k, 18, k == 0) for k, p in enumerate(shards)]):
        print(json.dumps(s, default=float))


@app.local_entrypoint()
def amend6_smoke():
    """Half-A smoke test of the runner (no held-out data): 1 persona, 1 instance per family, audit on 4 cells."""
    import json

    print(json.dumps(amend6_shard.remote(["writer"], 99, 1, True, "A", "smoke_"), default=float))


@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=8, memory=32768, timeout=3 * 3600)
def amend6_analyze_remote():
    import json

    from src.analysis_amend6 import analyze

    rep = analyze(A6, f"{RESULTS}/qwen3-32b/phase2")
    info = json.load(open(f"{A6}/directions.json"))
    rep["u"], rep["u53"] = info["u"], info["u53"]
    try:
        rep["audit_g1u"] = json.load(open(f"{A6}/summary_shard0.json")).get("audit_g1u")
    except FileNotFoundError:
        pass
    with open(f"{RESULTS}/qwen3-32b/amend6.json", "w") as f:
        json.dump(rep, f, default=float)
    results_vol.commit()
    return json.loads(json.dumps(rep, default=float))


@app.local_entrypoint()
def amend6_analyze():
    import json
    import pathlib

    rep = amend6_analyze_remote.remote()
    pathlib.Path("results/qwen3-32b/amend6.json").write_text(json.dumps(rep))
    print("saved")


# ------------------------------------------------------------------ Amendment 7: g as a detector (CPU)
@app.function(image=gpu_image, volumes={RESULTS: results_vol, WEIGHTS: weights_vol}, cpu=8, memory=32768, timeout=2 * 3600)
def amend7_remote():
    import json

    import torch

    from src.analysis_amend7 import analyze

    axis = torch.load(f"{AXIS_DIR}/assistant_axis.pt", map_location="cpu", weights_only=False)
    axis = axis["axis"] if isinstance(axis, dict) else axis
    base = f"{RESULTS}/qwen3-32b"
    rep = analyze(f"{base}/phase2", A6, f"{base}/pricing_pilot", axis[32].float().numpy())
    with open(f"{base}/amend7.json", "w") as f:
        json.dump(rep, f, default=float)
    results_vol.commit()
    return json.loads(json.dumps(rep, default=float))


@app.local_entrypoint()
def amend7():
    import json
    import pathlib

    rep = amend7_remote.remote()
    pathlib.Path("results/qwen3-32b/amend7.json").write_text(json.dumps(rep))
    print("saved")


# ------------------------------------------------------------------ Amendment 8 Part B (Tier B): human partner, length, non-social goal
A8B_FRAMES = ["coord_human", "guess_human", "pick_long", "goal_nonsocial"]
A8B_NONE_FRAMES = ["pick", "guess", "coord", "coord_human", "pick_long"]
NONE_PERSONA = {"none": {"stratum": "none", "prompts": [""]}}


@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=4 * 3600)
def phase2c_shard(personas: list[str], shard: int, include_none: bool = False, n_gof: int = 24):
    """Tier B: 23 substantive personas in coord_human / guess_human / pick_long / goal_nonsocial (F1-F5) and anti_human
    (F1, F3); the none persona in pick / guess / coord / coord_human / pick_long. Confirmatory instances, 4 orders, 2x2
    paraphrases; layer-32 residuals for every prompt."""
    import json
    import pathlib
    import random
    import time

    import torch

    from src import run
    from src.analysis_phase2 import pilot_instances
    from src.readout_hf import Scorer

    out = pathlib.Path(f"{RESULTS}/qwen3-32b/phase2c")
    out.mkdir(parents=True, exist_ok=True)
    axis = torch.load(f"{AXIS_DIR}/assistant_axis.pt", map_location="cpu", weights_only=False)
    axis = axis["axis"] if isinstance(axis, dict) else axis
    t = time.time()
    model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-32B")
    load_s = time.time() - t
    sc = Scorer(model, tok, max_batch_tokens=8192)
    pil = pilot_instances()
    sel = run.load_personas(personas)
    fams = ["F1", "F2", "F3", "F4", "F5"]
    cells = run.phase_cells(sel, fams, A8B_FRAMES, n_orders=4, anti_families=(), exclude_instances=pil)
    cells += run.phase_cells(sel, ["F1", "F3"], ["anti_human"], n_orders=4, anti_families=(), exclude_instances=pil)
    strata = {k: v["stratum"] for k, v in sel.items()}
    if include_none:
        cells += run.phase_cells(NONE_PERSONA, fams, A8B_NONE_FRAMES, n_orders=4, anti_families=(), exclude_instances=pil)
        strata["none"] = "none"
    meta = {"model": "Qwen3-32B", "engine": "hf", "layer_cfg": "", "intervention": "none", "dose": 0.0}
    tp = run.score_cells(sc, cells, meta, out / f"dist_shard{shard}.parquet", strata=strata, axis=axis, act_layer=32)
    results_vol.commit()
    gof = run.gof_gate(sc, random.Random(800 + shard).sample(cells, n_gof), seed=8000 * (shard + 1))
    summary = {"shard": shard, "personas": personas, "include_none": include_none, "cells": len(cells), "load_s": load_s,
               "throughput": tp, "gof": run.summarize_gof(gof)}
    (out / f"summary_shard{shard}.json").write_text(json.dumps(summary, indent=1))
    (out / f"gof_shard{shard}.json").write_text(json.dumps(gof))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def run_phase2c(n_shards: int = 6):
    import json

    from src.analysis_amend5 import substantive

    subst, stratum = substantive()
    order = sorted(subst, key=lambda p: (stratum[p], p))
    shards = [order[k::n_shards] for k in range(n_shards)]
    args = [(p, k, k == n_shards - 1) for k, p in enumerate(shards)]
    for s in phase2c_shard.starmap(args):
        print(json.dumps({k: s[k] for k in ("shard", "personas", "cells", "throughput", "gof")}, default=float))


# ------------------------------------------------------------------ Amendment 8 Part C0: sweep recapture (forward pass only)
@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=2 * 3600)
def c0_capture_remote():
    """Layers 0-3 on the Amendment 5 sweep cells; all sweep layers (0-3, 4..60, 63, 46-53) on pick cells with order 0,
    persona paraphrase 0, frame paraphrase 1."""
    import json
    import pathlib
    import time

    import numpy as np
    import pandas as pd

    from src import run
    from src.analysis_phase2 import SUBST, pilot_instances
    from src.prompts import render
    from src.readout_hf import Item, Scorer

    out = pathlib.Path(f"{RESULTS}/qwen3-32b/amend8_c0")
    out.mkdir(parents=True, exist_ok=True)
    allp = run.load_personas()
    sel = {k: v for k, v in allp.items() if v["stratum"] in SUBST and k != "facilitator"}
    pil = pilot_instances()
    fams = ["F1", "F2", "F3", "F4", "F5"]
    base = run.phase_cells(sel, fams, ["pick", "guess", "coord"], n_orders=4, exclude_instances=pil)
    sweep_cells = [c for c in base if c.order_id == 0 and c.persona_par == 0 and c.frame_par == 0]
    pick1 = [c for c in base if c.frame == "pick" and c.order_id == 0 and c.persona_par == 0 and c.frame_par == 1]
    jobs = {"sweep_l0_3": (sweep_cells, [0, 1, 2, 3]),
            "pick_par1": (pick1, [0, 1, 2, 3] + SWEEP_LAYERS + [46, 47, 49, 50, 51, 53])}
    t = time.time()
    model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-32B")
    load_s = time.time() - t
    sc = Scorer(model, tok, max_batch_tokens=16384)
    summary = {"load_s": load_s}
    for name, (cs, layers) in jobs.items():
        t = time.time()
        acts = np.zeros((len(cs), len(layers), model.config.hidden_size), np.float16)
        for s in range(0, len(cs), 2048):
            part = cs[s: s + 2048]
            items = [Item(*render(tok, c.system, c.user), c.options, key=i) for i, c in enumerate(part)]
            for j, r in enumerate(sc.capture(items, layers)):
                acts[s + j] = np.stack([r[l] for l in layers]).astype(np.float16)
            print(f"  {name} {s + len(part)}/{len(cs)} cells, {time.time() - t:.0f}s", flush=True)
        np.save(out / f"{name}_acts.npy", acts)
        pd.DataFrame([{"persona": c.persona, "frame": c.frame, "family": c.family, "instance_id": c.instance_id,
                       "order_id": c.order_id, "persona_par": c.persona_par, "frame_par": c.frame_par} for c in cs]
                     ).to_parquet(out / f"{name}_index.parquet")
        summary[name] = {"cells": len(cs), "layers": layers, "seconds": time.time() - t}
    # decision-token identity check for the embedding layer (A11-ii)
    last_ids = {render(tok, c.system, c.user)[0][-1] for c in sweep_cells[:500] + pick1[:500]}
    summary["decision_token_ids"] = sorted(last_ids)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def c0_capture():
    import json

    print(json.dumps(c0_capture_remote.remote(), indent=1))


# ------------------------------------------------------------------ Amendment 8 Part C2: cross-model baseline (Qwen3-14B)
@app.function(image=gpu_image, volumes={WEIGHTS: weights_vol}, secrets=[hf_secret], timeout=3600, cpu=4, memory=16384)
def download_14b():
    import os

    from huggingface_hub import snapshot_download

    snapshot_download("Qwen/Qwen3-14B", local_dir=f"{WEIGHTS}/Qwen/Qwen3-14B", token=os.environ["HF_TOKEN"],
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    weights_vol.commit()
    return "ok"


@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=2 * 3600)
def c2_remote(n_gof: int = 24):
    import json
    import pathlib
    import random
    import time

    from src import run
    from src.analysis_phase2 import pilot_instances
    from src.readout_hf import Scorer

    out = pathlib.Path(f"{RESULTS}/qwen3-14b/amend8_c2")
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-14B")
    load_s = time.time() - t
    sc = Scorer(model, tok, max_batch_tokens=16384)
    sel = run.load_personas(["assistant"])
    cells = run.phase_cells(sel, ["F1", "F2", "F3", "F4", "F5"], ["pick", "coord"], n_orders=4, anti_families=(),
                            exclude_instances=pilot_instances())
    meta = {"model": "Qwen3-14B", "engine": "hf", "layer_cfg": "", "intervention": "none", "dose": 0.0}
    tp = run.score_cells(sc, cells, meta, out / "dist_shard0.parquet", strata={"assistant": "assistant"})
    gof = run.gof_gate(sc, random.Random(900).sample(cells, n_gof), seed=9000)
    summary = {"cells": len(cells), "load_s": load_s, "throughput": tp, "gof": run.summarize_gof(gof)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def c2_run():
    import json

    print(download_14b.remote())
    print(json.dumps(c2_remote.remote(), indent=1, default=float))


# ------------------------------------------------------------------ Amendment 8 Part C1: steering positive controls
A8C1 = f"{RESULTS}/qwen3-32b/amend8_c1"


@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=4, memory=32768, timeout=3600)
def c1_prep_remote():
    """Half-A mean Delta-coord at layer 32 (all cells, 23 personas) and the stored coord residuals of the half-B
    cells (orders 1-2, paraphrase (0,0)) used as patch targets."""
    import glob
    import os

    import numpy as np
    import pandas as pd

    from src.amend6 import ORDERS, split
    from src.analysis_amend5 import COLS, substantive

    os.makedirs(A8C1, exist_ok=True)
    subst, _ = substantive()
    sp = split()
    half = {iid: h for f in sp for h in ("A", "B") for iid in sp[f][h]}
    acc, cnt, tg, keys = {}, {}, [], []
    for fn in sorted(glob.glob(f"{RESULTS}/qwen3-32b/phase2/dist_shard*.parquet")):
        d = pd.read_parquet(fn, columns=COLS)
        d["half"] = d.instance_id.map(half)
        d = d[d.persona.isin(subst) & d.half.notna()]
        a = np.load(fn[:-8] + "_l32.npy", mmap_mode="r")
        for fr in ("pick", "coord"):
            rows = d[(d.half == "A") & (d.frame == fr)].index.to_numpy()
            s = np.zeros(a.shape[1])
            for k in range(0, len(rows), 4096):
                s += np.asarray(a[rows[k: k + 4096]], np.float64).sum(0)
            acc[fr] = acc.get(fr, 0.0) + s
            cnt[fr] = cnt.get(fr, 0) + len(rows)
        m = d[(d.half == "B") & (d.frame == "coord") & d.order_id.isin(ORDERS) & (d.persona_par == 0) & (d.frame_par == 0)]
        tg.append(np.asarray(a[m.index.to_numpy()], np.float32))
        keys += list(zip(m.persona, m.instance_id, m.order_id))
    dcoord = acc["coord"] / cnt["coord"] - acc["pick"] / cnt["pick"]
    np.savez(f"{A8C1}/prep.npz", dcoord=dcoord.astype(np.float32), targets=np.concatenate(tg),
             keys=np.array(["|".join(map(str, k)) for k in keys]))
    results_vol.commit()
    return {"norm_dcoord": float(np.linalg.norm(dcoord)), "n_targets": len(keys)}


@app.function(image=gpu_image, gpu="H100", volumes={WEIGHTS: weights_vol, RESULTS: results_vol}, secrets=[hf_secret],
              timeout=3 * 3600)
def c1_remote():
    """(i) patch the decision-position residual of half-B pick cells with the stored coord residual of the same cell;
    (iii) add d x mean Delta-coord (d = 1, 2) and 5 norm-matched random directions (d = 1) at the decision position only;
    (ii) ablate g_A at every position of layers 20-40 in the coord frame (standard path)."""
    import json
    import time

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from src import amend6, run
    from src.prompts import render
    from src.readout_hf import Item, Scorer

    vec = np.load(f"{A6}/directions.npz")
    prep = np.load(f"{A8C1}/prep.npz")
    tindex = {k: i for i, k in enumerate(prep["keys"])}
    g = torch.tensor(vec["ghat"], device="cuda")
    dco = torch.tensor(prep["dcoord"], device="cuda")
    R = torch.tensor(vec["random"], device="cuda")
    t = time.time()
    model, tok = run.load_model(f"{WEIGHTS}/Qwen/Qwen3-32B")
    load_s = time.time() - t
    sc = Scorer(model, tok, max_batch_tokens=8192)
    summary = {"load_s": load_s}

    def write(path, recs_iter):
        writer = None
        for recs in recs_iter:
            tbl = pa.Table.from_pylist(recs)
            writer = writer or pq.ParquetWriter(path, tbl.schema)
            writer.write_table(tbl)
        writer.close()

    # (i) + (iii): pick frame, prompt reuse
    cs = amend6.cells(("pick",), "B")
    conds = [("patch", 1.0), ("dcoord", 1.0), ("dcoord", 2.0)] + [(f"r{k}", 1.0) for k in range(amend6.N_RANDOM)]
    nd = float(dco.norm())

    def gen_pick():
        for s in range(0, len(cs), 1024):
            part = cs[s: s + 1024]
            items = [Item(*render(tok, c.system, c.user), c.options, key=s + j) for j, c in enumerate(part)]
            targets = {s + j: torch.tensor(prep["targets"][tindex[f"{c.persona}|{c.instance_id}|{c.order_id}"]], device="cuda")
                       for j, c in enumerate(part)}
            deltas = [(32, targets, "replace"), (32, dco, "decision"), (32, 2 * dco, "decision")] + \
                     [(32, nd * R[k], "decision") for k in range(amend6.N_RANDOM)]
            res = sc.score_steered(items, deltas)
            print(f"  pick {s + len(part)}/{len(cs)}, {time.time() - t0:.0f}s", flush=True)
            yield [{"persona": c.persona, "frame": c.frame, "family": c.family, "instance_id": c.instance_id,
                    "order_id": c.order_id, "condition": cd[0], "dose": cd[1], "logprobs": r[k].logprobs.tolist(),
                    "offmenu": r[k].offmenu} for c, r in zip(part, res) for k, cd in enumerate(conds)]

    t0 = time.time()
    write(f"{A8C1}/pick_patch_dcoord.parquet", gen_pick())
    summary["pick_seconds"] = time.time() - t0

    # (ii): coord frame, directional ablation of g over layers 20-40, all positions (standard path)
    def ablate(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        gg = g.to(h.dtype)
        h = h - (h @ gg)[..., None] * gg
        return (h, *out[1:]) if isinstance(out, tuple) else h

    cs2 = amend6.cells(("coord",), "B")
    sc.cache_tag = ("ablate_g_20_40",)
    sc._prefix_cache.clear()
    hooks = [model.model.layers[l].register_forward_hook(ablate) for l in range(20, 41)]

    def gen_coord():
        for s in range(0, len(cs2), 1024):
            part = cs2[s: s + 1024]
            res = sc.score([Item(*render(tok, c.system, c.user), c.options) for c in part])
            print(f"  coord-ablate {s + len(part)}/{len(cs2)}, {time.time() - t0:.0f}s", flush=True)
            yield [{"persona": c.persona, "frame": c.frame, "family": c.family, "instance_id": c.instance_id,
                    "order_id": c.order_id, "condition": "ablate_g_20_40", "dose": 1.0, "logprobs": r.logprobs.tolist(),
                    "offmenu": r.offmenu} for c, r in zip(part, res)]

    t0 = time.time()
    try:
        write(f"{A8C1}/coord_ablate.parquet", gen_coord())
    finally:
        for h in hooks:
            h.remove()
    summary["coord_seconds"] = time.time() - t0
    with open(f"{A8C1}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    results_vol.commit()
    return summary


@app.local_entrypoint()
def c1_run():
    import json

    print(json.dumps(c1_prep_remote.remote(), indent=1))
    print(json.dumps(c1_remote.remote(), indent=1))


# ------------------------------------------------------------------ Amendment 8 Part A (CPU)
@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=8, memory=65536, timeout=4 * 3600)
def amend8a_remote(old: dict):
    import json

    from src.analysis_amend8a import analyze

    base = f"{RESULTS}/qwen3-32b"
    rep = analyze(f"{base}/phase2", f"{base}/phase2b", f"{base}/phase1/dist.parquet", old)
    with open(f"{base}/amend8_partA_behaviour.json", "w") as f:
        json.dump(rep, f, default=float)
    results_vol.commit()
    return json.loads(json.dumps(rep, default=float))


@app.local_entrypoint()
def amend8a():
    import json
    import pathlib

    R = pathlib.Path("results/qwen3-32b")
    old = {"phase2": json.loads((R / "phase2/analysis.json").read_text()),
           "amend3": json.loads((R / "amend3_partA.json").read_text())["T1.0"],
           "amend3_2b": json.loads((R / "amend3_2b.json").read_text())["T1.0"],
           "amend4": json.loads((R / "amend4.json").read_text())["T1.0"]}
    rep = amend8a_remote.remote(old)
    (R / "amend8_partA_behaviour.json").write_text(json.dumps(rep))
    print("saved")


@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=8, memory=98304, timeout=4 * 3600)
def amend8a_act_remote(amend5: dict):
    import json

    from src.analysis_amend8a_act import analyze

    base = f"{RESULTS}/qwen3-32b"
    rep = analyze(base, amend5)
    with open(f"{base}/amend8_partA_activations.json", "w") as f:
        json.dump(rep, f, default=float)
    results_vol.commit()
    return json.loads(json.dumps(rep, default=float))


@app.local_entrypoint()
def amend8a_act():
    import json
    import pathlib

    R = pathlib.Path("results/qwen3-32b")
    rep = amend8a_act_remote.remote(json.loads((R / "amend5.json").read_text()))
    (R / "amend8_partA_activations.json").write_text(json.dumps(rep))
    print("saved")


@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=8, memory=65536, timeout=4 * 3600)
def amend8b_remote():
    import json

    from src.analysis_amend8b import analyze

    base = f"{RESULTS}/qwen3-32b"
    rep = analyze(base)
    with open(f"{base}/amend8_partB.json", "w") as f:
        json.dump(rep, f, default=float)
    results_vol.commit()
    return json.loads(json.dumps(rep, default=float))


@app.function(image=gpu_image, volumes={RESULTS: results_vol}, cpu=8, memory=65536, timeout=3 * 3600)
def amend8c_remote():
    import json

    from src.analysis_amend8c import analyze

    base = f"{RESULTS}/qwen3-32b"
    rep = analyze(base, f"{RESULTS}/qwen3-14b/amend8_c2")
    with open(f"{base}/amend8_partC.json", "w") as f:
        json.dump(rep, f, default=float)
    results_vol.commit()
    return json.loads(json.dumps(rep, default=float))


@app.local_entrypoint()
def amend8bc(part: str = "b"):
    import json
    import pathlib

    fn, name = (amend8b_remote, "amend8_partB.json") if part == "b" else (amend8c_remote, "amend8_partC.json")
    rep = fn.remote()
    pathlib.Path(f"results/qwen3-32b/{name}").write_text(json.dumps(rep))
    print("saved")
