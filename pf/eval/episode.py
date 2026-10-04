"""Run one closed-loop Tier 1 episode and log everything metrics need (ground truth stays here)."""
from __future__ import annotations

import numpy as np

from pf.control.brain import FollowerBrain
from pf.perception import Perception, ground_truth_from_world
from pf.perception.sensor import visibility


def run_episode(cfg, world, brain: FollowerBrain, seed: int, noise_scale=None, steps=None, render=None,
                on_step=None):
    """world: a reset World (scenario already set up). Returns a log dict of arrays."""
    per = Perception(cfg, seed, noise_scale)
    steps = steps or world.max_steps
    ti = world.target_index
    fov = np.deg2rad(cfg["robot"]["fov_deg"])
    rmin, rmax = cfg["robot"]["cam_range"]
    L = {k: [] for k in ("robot", "cmd", "target", "people", "sel_gt", "in_fov", "visible", "state",
                         "collided", "goal")}
    for k in range(steps):
        obs = per.step(world)
        cmd = brain.step(obs)
        gt = ground_truth_from_world(world)
        # target "in view" geometry for metrics (true camera, before the robot moves this step)
        tp = world.people.pos[ti]
        d = tp - gt.cam_pose[:2]
        dist = np.hypot(*d)
        bearing = np.arctan2(d[1], d[0]) - gt.cam_pose[2]
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        L["in_fov"].append(bool(abs(bearing) <= fov / 2 and rmin <= dist <= rmax))
        L["visible"].append(bool(per.last_info["visible"][ti]))
        L["sel_gt"].append(-2 if brain.selected is None else int(brain.selected["_gt_id"]))
        L["state"].append(brain.state)
        L["goal"].append(np.full(2, np.nan) if brain.goal is None else np.asarray(brain.goal, float))
        if render is not None:
            render(world, brain, obs)
        world.step(cmd)
        L["robot"].append(world.robot.pose.copy())
        L["cmd"].append(world.robot.last_cmd.copy())
        L["target"].append(world.people.pos[ti].copy())
        L["people"].append(world.people.pos.copy())
        L["collided"].append(world.robot.collided)
        if on_step is not None:
            on_step(world, brain, obs)
    out = {k: np.asarray(v) for k, v in L.items()}
    out["target_index"] = ti
    out["heading_target"] = np.asarray(world.hist["heading"])[:, ti]
    out["dt"] = world.dt
    out["robot_person_d"] = np.asarray(world.hist["robot_person_d"][-len(out["robot"]):])
    out["n_swaps"] = per.tracker.n_swaps
    return out
