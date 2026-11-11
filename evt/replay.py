"""Offline replay of recorded EVT brain inputs (<episode>_raw.npy) through tracker + memory + brain, open loop.
Finds why the target gets dropped while it is detected. No Habitat needed.

    python evt/replay.py results/evt/rec/out/ours/stt [--variant base] [--verbose 3]
"""
import argparse
import collections
import glob
import os

import numpy as np

from evt.evt_cfg import EVT_OVERRIDES, STEP_S, VARIANTS
from pf.config import deep_merge, load_config
from pf.control.brain import FollowerBrain
from pf.perception.sensor import Detections
from pf.perception.tracker import Tracker
from pf.prediction.base import ConstantVelocity


def dets(r):
    n = len(r["gid"])
    return Detections(pos_robot=r["pos"].reshape(n, 2), body_heading_rel=np.full(n, np.nan), head_rel=np.full(n, np.nan),
                      appearance=r["app"].astype(np.float64).reshape(n, 256), gt_id=r["gid"].astype(int),
                      meas_sigma=0.05 + 0.03 * np.hypot(*r["pos"].reshape(n, 2).T))


def replay(path, cfg, verbose=0):
    raw = np.load(path, allow_pickle=True)
    main = None
    brain = FollowerBrain(cfg, "C1", predictor=ConstantVelocity(), memory="M2", search="S1")
    trk = Tracker(cfg["tracker"], STEP_S, 1.0, np.random.default_rng(0))
    why = collections.Counter()
    prev = None
    for k, r in enumerate(raw):
        if r["init"] and len(r["gid"]):
            main = int(r["gid"][0])
        d = dets(r)
        tracks = trk.update(d, r["odom"])
        mem = brain.memory
        brain.step({"tracks": tracks, "robot_odom": r["odom"], "ray_angles": np.linspace(-np.pi / 4, np.pi / 4, 32),
                    "depth_rays": r["rays"].astype(float)})
        if main is None or not mem.initialized:
            continue
        det_t = main in d.gt_id
        seen_t = [t for t in tracks if t["_gt_id"] == main and t["time_since_seen_s"] == 0.0]
        sel_ok = brain.selected is not None and brain.selected["_gt_id"] == main
        if det_t and not sel_ok:
            if not seen_t:
                conf = [t for t in trk.tracks if t.gt_id == main]
                reason = "tracker: target detected but no confirmed seen track" + (" (unconfirmed new track)" if any(not t.confirmed for t in conf) else "")
            else:
                reason = f"memory rejected (s={mem.last_scores.get('s', 0):.2f} app={mem.last_scores.get('s_app', 0):.2f} mot={mem.last_scores.get('s_motion', 0):.2f}, lost={mem.lost})"
                reason = "memory rejected" + (" while lost" if mem.lost else "")
            why[reason] += 1
            if verbose and prev != reason:
                p = d.pos_robot[list(d.gt_id).index(main)]
                print(f"  {os.path.basename(path)} step {k}: {reason}  det@{np.round(p, 2)} scores={ {a: round(b, 2) for a, b in mem.last_scores.items()} }")
            prev = reason
        else:
            prev = None
    return why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--verbose", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(overrides=deep_merge(EVT_OVERRIDES, VARIANTS[a.variant]))
    tot = collections.Counter()
    files = sorted(glob.glob(os.path.join(a.root, "*", "*_raw.npy")))
    for i, f in enumerate(files):
        tot += replay(f, cfg, verbose=a.verbose if i < 6 else 0)
    print(f"{len(files)} episodes, steps where the detected target was not selected:")
    for k, v in tot.most_common():
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main()
