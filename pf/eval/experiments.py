"""Run Stage C experiments from configs/experiments/*.yaml -> results/<ID>/episodes.csv + summary.csv.

Every method in an experiment runs on the SAME scenario seeds (same layouts, people, and perception noise
streams); only the component under test changes.
"""
from __future__ import annotations

import copy
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pf.config import CONFIG_DIR, REPO_ROOT, deep_merge, load_config
from pf.control.brain import FollowerBrain
from pf.eval.episode import run_episode
from pf.eval.metrics import episode_metrics
from pf.eval.scenarios import build
from pf.prediction.base import HORIZON, K, ConstantVelocity, Forecast
from pf.prediction.data import true_to_odom

EXP_DIR = CONFIG_DIR / "experiments"
METRICS = ["tracking_rate", "success", "turn_lag", "lost_events", "recovery_rate", "recovery_time",
           "wrong_switches", "person_collisions", "obstacle_collisions", "personal_space_compliance",
           "min_dist_other", "smoothness"]

_PRED_CACHE: dict = {}


def get_predictor(name):
    if name == "P0":
        return ConstantVelocity()
    if name not in _PRED_CACHE:
        from pf.prediction.model import LearnedPredictor
        _PRED_CACHE[name] = LearnedPredictor(REPO_ROOT / "models" / f"predictor_{name}.pt")
    return _PRED_CACHE[name]


def oracle_future_fn(world):
    """E6: the TRUE next 2 s of the target (people simulated ahead on a copy; the robot held in place),
    expressed in the robot's odometry frame. Not a method: an upper bound on prediction."""
    def fn(t):
        p = world.people
        memo = {id(p.map): p.map, id(p.grid): p.grid, id(p.planner): p.planner}
        ghost = copy.deepcopy(p, memo)
        ti = world.target_index
        fut = []
        for _ in range(HORIZON):
            ghost.step(world.robot.pose[:2], world.robot.radius)
            fut.append(ghost.pos[ti].copy())
        fut = true_to_odom(np.asarray(fut), world.robot.pose, world.robot.odom)
        modes = np.repeat(fut[None], K, axis=0)
        return Forecast(modes, np.array([1.0, 0.0, 0.0]), np.full(K, np.log(0.1)), t)
    return fn


def run_one(args):
    exp_id, method, scenario, seed = args
    cfg = load_config()
    over = {}
    if "fov_deg" in method:
        over["robot"] = {"fov_deg": float(method["fov_deg"])}
    if "noise_scale" in method:
        over["perception"] = {"noise_scale": float(method["noise_scale"])}
    cfg = deep_merge(cfg, over)
    t0 = time.time()
    try:
        world = build(cfg, scenario, seed)
        pred = method.get("predictor", "P0")
        oracle = oracle_future_fn(world) if pred == "oracle" else None
        brain = FollowerBrain(cfg, method["controller"], predictor=None if pred == "oracle" else get_predictor(pred),
                              memory=method.get("memory", "M2"), search=method.get("search", "S1"),
                              delta_s=method.get("delta"), oracle_future=oracle)
        log = run_episode(cfg, world, brain, seed)
        m = episode_metrics(log)
        m["error"] = ""
    except Exception as e:  # recorded, never hidden
        m = {"error": repr(e)[:300]}
    m.update(experiment=exp_id, method=method["name"], scenario=scenario, seed=seed, wall_s=time.time() - t0)
    return m


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(df):
    rows = []
    for (method, scenario), g in df.groupby(["method", "scenario"], sort=False):
        r = {"method": method, "scenario": scenario, "n": len(g)}
        for mtr in METRICS:
            if mtr in g:
                mean, lo, hi = boot_ci(g[mtr].astype(float))
                r[mtr], r[mtr + "_lo"], r[mtr + "_hi"] = mean, lo, hi
        rows.append(r)
    # pooled over scenarios
    for method, g in df.groupby("method", sort=False):
        r = {"method": method, "scenario": "ALL", "n": len(g)}
        for mtr in METRICS:
            if mtr in g:
                mean, lo, hi = boot_ci(g[mtr].astype(float))
                r[mtr], r[mtr + "_lo"], r[mtr + "_hi"] = mean, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def run_experiment(path: Path, procs=8, n_override=None, out_root=None):
    spec = yaml.safe_load(open(path))
    exp_id = spec["id"]
    n = n_override or spec["seeds"]["n"]
    seeds = list(range(spec["seeds"]["start"], spec["seeds"]["start"] + n))
    assert all(s >= 1000 for s in seeds), "experiments run on held-out test seeds only"
    jobs = [(exp_id, m, sc, s) for m in spec["methods"] for sc in spec["scenarios"] for s in seeds]
    out = Path(out_root or REPO_ROOT / "results") / exp_id
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(procs) as pool:
        rows = pool.map(run_one, jobs, chunksize=2)
    df = pd.DataFrame(rows)
    df.to_csv(out / "episodes.csv", index=False)
    errs = df[df["error"].astype(str) != ""]
    ok = df[df["error"].astype(str) == ""]
    summ = summarize(ok)
    summ.to_csv(out / "summary.csv", index=False)
    meta = {"id": exp_id, "config": str(path.relative_to(REPO_ROOT)), "spec": spec, "seeds": [seeds[0], seeds[-1]],
            "episodes": len(df), "failed_episodes": len(errs), "wall_minutes": (time.time() - t0) / 60,
            "world_config": load_config()}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(f"[{exp_id}] {len(df)} episodes ({len(errs)} failed) in {(time.time() - t0) / 60:.1f} min -> {out}")
    if len(errs):
        print(errs[["method", "scenario", "seed", "error"]].head(10).to_string())
    return summ
