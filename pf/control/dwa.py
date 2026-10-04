"""Shared Dynamic Window Approach local planner (Section 8.1). Every controller only picks the goal
(and a look point to keep in view); this planner turns it into (vx, vy, wz).

Uses only robot-side information: odometry, its own command history (to estimate its velocity with the
response model), the depth-ray LocalMap, and tracks.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from pf.control.local_map import LocalMap
from pf.world.robot import ResponseModel, integrate_pose, wrap


class DWAPlanner:
    def __init__(self, cfg: dict, origin_xy):
        self.dc = dc = cfg["dwa"]
        rc = cfg["robot"]
        self.dt = cfg["sim"]["dt"]
        self.model = ResponseModel(rc, self.dt)
        self.radius = rc["radius"]
        self.fov = np.deg2rad(rc["fov_deg"])
        self.map = LocalMap(dc, origin_xy, rc["cam_range"][1])
        self.T = int(round(dc["horizon_s"] / self.dt))
        self.v_est = np.zeros(3)
        self.queue = deque([np.zeros(3)] * self.model.latency_steps)
        self.prev_cmd = np.zeros(3)
        self.acc = np.array([rc["acc_lin"], rc["acc_lin"], rc["acc_ang"]])
        self.span = self.model.hi - self.model.lo
        self.last_info = {}

    def observe(self, obs):
        self.map.update(obs["robot_odom"], obs["ray_angles"], obs["depth_rays"])

    def _samples(self):
        dc = self.dc
        lo = np.maximum(self.model.lo, self.v_est - self.acc * dc["window_s"])
        hi = np.minimum(self.model.hi, self.v_est + self.acc * dc["window_s"])
        axes = [np.linspace(l, h, n) for l, h, n in zip(lo, hi, (dc["n_vx"], dc["n_vy"], dc["n_wz"]))]
        axes = [np.unique(np.append(a, 0.0)) if l <= 0 <= h else a for a, l, h in zip(axes, lo, hi)]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        return g

    def plan(self, odom, goal, look, people, target=None):
        """odom (3,), goal (2,) or None, look (2,) or None: odometry-frame points.
        people: list of (pos (2,), vel (2,)) for NON-target tracks; target: (pos, vel) or None.
        Returns the command (3,)."""
        dc = self.dc
        cmds = self._samples()
        S, T = len(cmds), self.T
        pose = np.repeat(np.asarray(odom, float)[None], S, axis=0)
        v = np.repeat(self.v_est[None], S, axis=0)
        pending = list(self.queue)
        traj = np.zeros((S, T, 3))
        for k in range(T):
            eff = pending[k] if k < len(pending) else cmds
            v = self.model.step_vel(v, eff)
            pose = integrate_pose(pose, v, self.dt)
            traj[:, k] = pose
        xy = traj[..., :2]
        self.map.prepare(odom[:2])
        clear = self.map.clearance(xy.reshape(-1, 2)).reshape(S, T)
        min_clear = clear.min(axis=1) - self.radius
        ok = min_clear >= dc["collide_margin"]
        cost = np.zeros(S)
        cost += dc["w_clear"] * np.clip((dc["clear_soft"] - min_clear) / dc["clear_soft"], 0, 1) ** 2
        tsec = (np.arange(T) + 1) * self.dt
        if people:
            P = np.stack([p for p, _ in people])
            V = np.stack([vv for _, vv in people])
            fut = P[None, :, :] + tsec[:, None, None] * V[None, :, :]            # (T, P, 2)
            d = np.linalg.norm(xy[:, :, None, :] - fut[None], axis=-1)          # (S, T, P)
            soft, hard = dc["social_soft"], dc["social_hard"]
            pen = np.clip((soft - d) / (soft - hard), 0, 1) + 10.0 * (d < hard)
            cost += dc["w_social"] * pen.mean(axis=1).sum(axis=1)
            ok &= d.min(axis=(1, 2)) >= dc["person_collide"]
        if target is not None:
            tp, tv = target
            fut = tp[None] + tsec[:, None] * tv[None]                              # (T, 2)
            dt_ = np.linalg.norm(xy - fut[None], axis=-1)
            ok &= dt_.min(axis=1) >= dc["person_collide"]
        if goal is not None:
            # distance at the end of the horizon, plus a bit of the average (prefers getting there sooner)
            dg = np.linalg.norm(xy - np.asarray(goal)[None, None], axis=-1)
            cost += dc["w_goal"] * (0.7 * dg[:, -1] + 0.3 * dg.mean(axis=1))
        if look is not None:
            dl = np.asarray(look)[None] - xy[:, -1]
            ang = np.abs(wrap(np.arctan2(dl[:, 1], dl[:, 0]) - traj[:, -1, 2]))
            cost += dc["w_heading"] * (ang / (self.fov / 2)) ** 2
        cost += dc["w_smooth"] * np.abs((cmds - self.prev_cmd) / self.span).sum(axis=1)
        if ok.any():
            cost[~ok] = np.inf
            cmd = cmds[int(np.argmin(cost))]
            fallback = False
        else:
            # no safe sample: stop and rotate toward the look point (rotation in place cannot hit walls)
            wz = 0.0
            if look is not None:
                dl = np.asarray(look) - odom[:2]
                wz = float(np.clip(1.5 * wrap(np.arctan2(dl[1], dl[0]) - odom[2]), self.model.lo[2], self.model.hi[2]))
            cmd = np.array([0.0, 0.0, wz])
            fallback = True
        self.last_info = {"fallback": fallback, "n_ok": int(ok.sum()), "best_traj": traj[int(np.argmin(cost))] if not fallback else None}
        return cmd

    def commit(self, cmd):
        """Record the command actually sent (updates the internal velocity estimate)."""
        cmd = self.model.clamp(np.asarray(cmd, float))
        self.queue.append(cmd)
        eff = self.queue.popleft()
        self.v_est = self.model.step_vel(self.v_est, eff)
        self.prev_cmd = cmd
