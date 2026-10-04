"""M4 deliverable: tracks-over-truth GIF. Circles = true people, x + ID = confirmed tracks
(world estimate drawn in the TRUE frame by correcting for odometry drift, so only perception error shows).
The robot uses the naive ground-truth chase from pf.world.demo (display only).

    python -m pf.perception.demo [--family office] [--seed 4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pf.config import REPO_ROOT, load_config
from pf.perception import Perception
from pf.world.demo import naive_chase
from pf.world.render import OTHER_C, TARGET_C, TopDownRenderer, save_gif
from pf.world.sim import World


def odom_to_true(p, odom, true_pose):
    """Map a point from the odometry frame to the true world frame (display only)."""
    c, s = np.cos(odom[2]), np.sin(odom[2])
    d = p - odom[:2]
    rel = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    c, s = np.cos(true_pose[2]), np.sin(true_pose[2])
    return true_pose[:2] + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="office")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "M4"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    w = World(cfg).reset(args.seed, args.family, n_others=4)
    per = Perception(cfg, args.seed)
    rend = TopDownRenderer(w.map, cfg["robot"]["fov_deg"], cfg["robot"]["cam_range"][1])
    colors = [TARGET_C if t else OTHER_C for t in w.people.is_target]
    frames, errs = [], []
    for _ in range(int(args.seconds / w.dt)):
        w.step(naive_chase(w))
        obs = per.step(w)
        pts_t, pts_o, labels = [], [], []
        for tr in obs["tracks"]:
            p = odom_to_true(tr["pos_world_est"], w.robot.odom, w.robot.pose)
            (pts_t if tr["_gt_id"] == w.target_index else pts_o).append(p)
            if tr["_gt_id"] >= 0 and tr["time_since_seen_s"] == 0:
                errs.append(np.hypot(*(p - w.people.pos[tr["_gt_id"]])))
        ovs = [("points", pts_o or np.zeros((0, 2)), dict(marker="x", c="k", s=40)),
               ("points", pts_t or np.zeros((0, 2)), dict(marker="x", c=TARGET_C, s=60))]
        ids = {tr["_gt_id"]: str(tr["track_id"]) for tr in obs["tracks"] if tr["_gt_id"] >= 0}
        labels = [("id " + ids[i]) if i in ids else "" for i in range(w.people.n)]
        txt = (f"M4 tracks over truth | {args.family} seed {args.seed} | t={w.t:4.1f}s | swaps so far {per.tracker.n_swaps}\n"
               f"circles = truth, x = confirmed tracks (label = track ID on the true person); robot = naive GT chase (demo)")
        frames.append(rend.draw(w.robot.pose, w.robot.camera_pose(), w.people.pos, w.people.heading,
                                w.people.head, colors, labels=labels, text=txt, overlays=ovs))
    rend.close()
    save_gif(frames, out / "tracks_over_truth.gif", fps=10)
    meta = {"family": args.family, "seed": args.seed, "seconds": args.seconds,
            "track_pos_err_median_m": float(np.median(errs)), "track_pos_err_p90_m": float(np.percentile(errs, 90)),
            "n_swaps": per.tracker.n_swaps}
    with open(out / "tracks_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(meta)


if __name__ == "__main__":
    main()
