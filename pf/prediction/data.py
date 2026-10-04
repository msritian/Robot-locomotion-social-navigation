"""Predictor dataset (Section 7.5): run the simulator with the robot following under C0 (+P0, M2, S1),
record the target's ESTIMATED 1 s history (what the brain sees) and the TRUE next 2 s (labels),
both in the robot's odometry frame at the sample time.

    python -m pf.prediction.data --split train --episodes 520 --seed0 0      # train seeds 0-519
    python -m pf.prediction.data --split val   --episodes 80  --seed0 900    # val seeds 900-979
    python -m pf.prediction.data --split test  --episodes 200 --seed0 1000   # held-out test layouts

Samples are kept only when the history window and the current selection belong to the true target
(ground truth is used here only to build clean labels) and at least 3 of the 10 history steps are visible.
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from pf.config import REPO_ROOT, load_config, split_of
from pf.control.brain import FollowerBrain
from pf.perception import Perception
from pf.prediction.base import HIST, HORIZON
from pf.world.robot import wrap
from pf.world.sim import World

OUT_DIR = REPO_ROOT / "results" / "M6" / "raw"


def true_to_odom(p_true, true_pose, odom):
    """Map true-world points into the robot's odometry frame using the pose pair at the same time."""
    c, s = np.cos(true_pose[2]), np.sin(true_pose[2])
    d = p_true - true_pose[:2]
    rel = np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)
    c, s = np.cos(odom[2]), np.sin(odom[2])
    return odom[:2] + np.stack([c * rel[..., 0] - s * rel[..., 1], s * rel[..., 0] + c * rel[..., 1]], axis=-1)


def collect_episode(seed: int, cfg=None):
    cfg = cfg or load_config()
    w = World(cfg).reset(seed, n_others=int(np.random.default_rng([seed, 5]).integers(0, 5)))
    ti = w.target_index
    per = Perception(cfg, seed)
    brain = FollowerBrain(cfg, "C0", memory="M2", search="S1")
    steps = w.max_steps
    H = {k: [] for k in ("pos", "vel", "body", "head", "vis", "sel_ok", "odom", "true_pose")}
    for _ in range(steps):
        obs = per.step(w)
        cmd = brain.step(obs)
        sel = brain.selected
        h = brain.hist
        H["pos"].append(h.pos[-1].copy()); H["vel"].append(h.vel[-1].copy())
        H["body"].append(h.body[-1]); H["head"].append(h.head[-1]); H["vis"].append(h.vis[-1])
        H["sel_ok"].append(sel is None or sel["_gt_id"] == ti)
        H["odom"].append(w.robot.odom.copy()); H["true_pose"].append(w.robot.pose.copy())
        w.step(cmd)
    H = {k: np.asarray(v) for k, v in H.items()}
    # true target positions/headings at the time of each observation (world history is logged after
    # each step, so the state seen at step k is hist[k-1]; prepend the initial state)
    # (the target pauses for >= 0.5 s at t = 0, so hist[0] equals the initial state)
    hp, hh = np.asarray(w.hist["pos"])[:, ti], np.asarray(w.hist["heading"])[:, ti]
    tpos = np.concatenate([hp[:1], hp], axis=0)
    thead = np.concatenate([hh[:1], hh])
    samples = {k: [] for k in ("pos", "vel", "body", "head", "vis", "fut", "turn", "cur_true")}
    for t in range(HIST - 1, steps - HORIZON):
        win = slice(t - HIST + 1, t + 1)
        vis = H["vis"][win]
        if not H["vis"][t] or vis.sum() < 3 or not H["sel_ok"][win].all():
            continue
        # labels: true positions at t+1..t+20 in the odometry frame of time t
        fut = true_to_odom(tpos[t + 1:t + 1 + HORIZON], H["true_pose"][t], H["odom"][t])
        samples["pos"].append(H["pos"][win]); samples["vel"].append(H["vel"][win])
        samples["body"].append(H["body"][win]); samples["head"].append(H["head"][win])
        samples["vis"].append(vis); samples["fut"].append(fut)
        samples["cur_true"].append(true_to_odom(tpos[t], H["true_pose"][t], H["odom"][t]))
        samples["turn"].append(abs(wrap(thead[t + HORIZON] - thead[t])) > np.deg2rad(45))
    out = {k: np.asarray(v) for k, v in samples.items()}
    out["seed"] = np.full(len(out["fut"]), seed)
    return out


def _job(seed):
    return collect_episode(seed)


def collect(seeds, procs=8):
    with Pool(procs) as pool:
        parts = pool.map(_job, seeds, chunksize=4)
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts if len(p["fut"])]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed0", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config()
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    want = "test" if args.split == "test" else "train"
    assert all(split_of(s, cfg) == want for s in seeds), f"{args.split} must use {want} seeds"
    data = collect(seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / f"{args.split}.npz", **data)
    print(f"{args.split}: {len(data['fut'])} samples from {len(seeds)} episodes (seeds {seeds[0]}-{seeds[-1]}), "
          f"turn subset {int(data['turn'].sum())}")


if __name__ == "__main__":
    main()
