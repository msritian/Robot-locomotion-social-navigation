"""Association-only evaluation of the memory variants (M5) and threshold tuning on TRAINING seeds.

Episodes: target + a look-alike distractor (cos 0.85-0.95) that starts next to the target and walks to
the same goal (then wanders), plus `n_extra` random others. The robot uses the naive ground-truth chase
so that only association is under test here (the real controllers come in M7).

    python -m pf.memory.evaluate --tune      # grid search on seeds 0-99 -> results/M5/tune.csv
    python -m pf.memory.evaluate             # compare M0-M3 with the configured thresholds
"""
from __future__ import annotations

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from pf.config import REPO_ROOT, deep_merge, load_config
from pf.memory.target import VARIANTS, TargetMemory
from pf.perception import Perception
from pf.world.demo import naive_chase
from pf.world.sim import World


def lookalike_world(cfg, seed, n_extra=1):
    """The look-alike starts 3-6 m from the target and outside the initial central view (so the t = 0
    initialization of Section 6.1 is unambiguous), and walks to the target's first goal, so it converges on
    and crosses the target's path."""
    w = World(cfg).reset(seed, n_others=0)
    ti = w.target_index
    ag = w.people.agents[ti]
    tp = w.people.pos[ti]
    rp = w.robot.pose
    pts = w.walkable_points(0.45)
    d_t = np.hypot(*(pts - tp).T)
    bearing = np.abs(np.arctan2(pts[:, 1] - rp[1], pts[:, 0] - rp[0]) - rp[2])
    bearing = np.abs((bearing + np.pi) % (2 * np.pi) - np.pi)
    cand = pts[(d_t > 3.0) & (d_t < 6.0) & (bearing > np.deg2rad(50))]
    if len(cand) == 0:
        cand = pts[d_t > 3.0]
    p = cand[w.rng.integers(len(cand))]
    goal = w.map.pois[ag.next_goal] if ag.next_goal is not None else w.map.pois[0]
    base = w.app.lookalike_base(w.app.base[w.app_idx[ti]])
    w.add_person(p, target=False, base=base, route=[(goal, 1.0)], start_pause=0.0)
    pts = w.walkable_points(0.45)
    for _ in range(n_extra):
        w.add_person(pts[w.rng.integers(len(pts))], start_pause=0.0)
    return w


def run_episode(cfg, seed, variant, n_extra=1, steps=600):
    w = lookalike_world(cfg, seed, n_extra)
    ti = w.target_index
    per = Perception(cfg, seed)
    mem = TargetMemory(cfg["memory"], variant)
    sel, tvis = [], []
    for _ in range(steps):
        w.step(naive_chase(w))
        obs = per.step(w)
        tvis.append(any(t["_gt_id"] == ti and t["time_since_seen_s"] == 0 for t in obs["tracks"]))
        if not mem.initialized:
            mem.init(obs)
            sel.append(-2)
            continue
        t = mem.step(obs, w.dt)
        sel.append(-2 if t is None else t["_gt_id"])
    sel, tvis = np.array(sel), np.array(tvis)
    has = sel != -2
    correct = (sel == ti)
    wrong = has & ~correct
    switches, run = 0, 0
    for x in wrong:
        run = run + 1 if x else 0
        if run == int(round(1.0 / w.dt)) + 1:     # wrong for > 1 s continuously
            switches += 1
    return {"seed": seed, "variant": variant,
            "precision": correct.sum() / max(has.sum(), 1),
            "recall": correct.sum() / max(tvis.sum(), 1),
            "wrong_frac": wrong.sum() / max(has.sum(), 1),
            "wrong_switches": switches, "mem_size": mem.memory.shape[0], "n_added": mem.n_added,
            "wrong_at_end": bool(wrong[-10:].any())}


def _job(args):
    cfg, seed, variant = args
    return run_episode(cfg, seed, variant)


def evaluate(cfg, seeds, variants, procs=8):
    jobs = [(cfg, s, v) for v in variants for s in seeds]
    with Pool(procs) as pool:
        return pd.DataFrame(pool.map(_job, jobs, chunksize=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--seeds", type=int, default=100, help="number of training seeds (from 0)")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "M5"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = load_config()
    seeds = list(range(args.seeds))
    assert max(seeds) < base["maps"]["test_seed_start"], "tuning must use training seeds only"
    if args.tune:
        grid = {"w_app": [0.6, 0.8, 0.9], "tau_accept": [0.7, 0.8, 0.85, 0.9], "tau_reacquire": [0.85, 0.9],
                "tau_high": [0.88]}
        rows = []
        for vals in itertools.product(*grid.values()):
            over = dict(zip(grid, vals))
            over["w_motion"] = round(1.0 - over["w_app"], 3)
            cfg = deep_merge(base, {"memory": over})
            df = evaluate(cfg, seeds, ["M2"])
            r = {**over, **df[["precision", "recall", "wrong_frac", "wrong_switches"]].mean().to_dict()}
            r["score"] = r["recall"] - 2.0 * r["wrong_frac"]   # objective: keep the target, penalize wrong picks
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        res = pd.DataFrame(rows).sort_values("score", ascending=False)
        res.to_csv(out / "tune.csv", index=False)
        print("best:\n", res.head(5).to_string())
    else:
        df = evaluate(base, seeds, list(VARIANTS))
        df.to_csv(out / "variants_train.csv", index=False)
        summ = df.groupby("variant")[["precision", "recall", "wrong_frac", "wrong_switches", "mem_size",
                                     "wrong_at_end"]].mean()
        print(summ.to_string())
        with open(out / "variants_train_summary.json", "w") as f:
            json.dump({"seeds": [seeds[0], seeds[-1]], "memory_cfg": base["memory"],
                       "summary": summ.reset_index().to_dict(orient="records")}, f, indent=1)


if __name__ == "__main__":
    main()
