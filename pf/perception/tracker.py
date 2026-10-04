"""Multi-object tracker (Section 5.3): constant-velocity Kalman filter per track in the robot's
odometry (world-estimate) frame, Hungarian association with a Mahalanobis gate, confirmation after
2 consecutive hits, deletion after 1 s unseen, plus injected ID swaps between close people.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from pf.perception.sensor import Detections
from pf.world.robot import wrap


@dataclass
class _Track:
    track_id: int
    x: np.ndarray               # (4,) [x, y, vx, vy] world-estimate frame
    P: np.ndarray               # (4, 4)
    hits: int = 1
    confirmed: bool = False
    age_s: float = 0.0
    time_since_seen_s: float = 0.0
    body_heading: float = np.nan
    head_yaw: float = np.nan
    appearance: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gt_id: int = -1             # METRICS ONLY: true person of the last associated detection


class Tracker:
    def __init__(self, tc: dict, dt: float, noise_scale: float, rng: np.random.Generator):
        self.tc = tc
        self.dt = dt
        self.s = noise_scale
        self.rng = rng
        self.tracks: list[_Track] = []
        self.next_id = 0
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        self.F = F
        q = tc["q_acc"] ** 2
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        self.Q = G @ G.T * q
        self.H = np.eye(2, 4)
        self.n_swaps = 0

    def reset(self):
        self.tracks, self.next_id, self.n_swaps = [], 0, 0

    # ------------------------------------------------------------------ main update
    def update(self, det: Detections, odom: np.ndarray, gt_pos: np.ndarray | None = None) -> list[dict]:
        tc = self.tc
        # detections -> odometry (world-estimate) frame
        c, s = np.cos(odom[2]), np.sin(odom[2])
        z = np.stack([odom[0] + c * det.pos_robot[:, 0] - s * det.pos_robot[:, 1],
                      odom[1] + s * det.pos_robot[:, 0] + c * det.pos_robot[:, 1]], axis=1) \
            if len(det.pos_robot) else np.zeros((0, 2))
        sig = np.maximum(det.meas_sigma, tc["min_meas_sigma"])
        # predict
        for t in self.tracks:
            t.x = self.F @ t.x
            t.P = self.F @ t.P @ self.F.T + self.Q
            t.age_s += self.dt
            t.time_since_seen_s += self.dt
        # associate (Hungarian on Mahalanobis distance, gated)
        nT, nD = len(self.tracks), len(z)
        matched_t, matched_d = [], []
        if nT and nD:
            cost = np.full((nT, nD), 1e6)
            for i, t in enumerate(self.tracks):
                S_base = t.P[:2, :2]
                for j in range(nD):
                    S = S_base + np.eye(2) * sig[j] ** 2
                    r = z[j] - t.x[:2]
                    m2 = float(r @ np.linalg.solve(S, r))
                    if m2 <= tc["gate_chi2"]:
                        cost[i, j] = m2
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e6:
                    matched_t.append(i)
                    matched_d.append(j)
        # update matched
        for i, j in zip(matched_t, matched_d):
            t = self.tracks[i]
            R = np.eye(2) * sig[j] ** 2
            S = self.H @ t.P @ self.H.T + R
            K = t.P @ self.H.T @ np.linalg.inv(S)
            t.x = t.x + K @ (z[j] - self.H @ t.x)
            t.P = (np.eye(4) - K @ self.H) @ t.P
            t.hits += 1
            if t.hits >= tc["confirm_hits"]:
                t.confirmed = True
            t.time_since_seen_s = 0.0
            t.body_heading = wrap(odom[2] + det.body_heading_rel[j])
            t.head_yaw = wrap(odom[2] + det.head_rel[j]) if np.isfinite(det.head_rel[j]) else np.nan
            t.appearance = det.appearance[j]
            t.gt_id = int(det.gt_id[j])
        # drop tentative tracks that missed; confirmed tracks after delete_after_s unseen
        matched_set = set(matched_t)
        keep = []
        for i, t in enumerate(self.tracks):
            if not t.confirmed and i not in matched_set:
                continue
            if t.time_since_seen_s > tc["delete_after_s"] + 1e-9:
                continue
            keep.append(t)
        self.tracks = keep
        # new tentative tracks from unmatched detections
        used = set(matched_d)
        for j in range(nD):
            if j in used:
                continue
            P = np.diag([sig[j] ** 2, sig[j] ** 2, tc["init_vel_sigma"] ** 2, tc["init_vel_sigma"] ** 2])
            t = _Track(self.next_id, np.array([z[j, 0], z[j, 1], 0.0, 0.0]), P,
                       body_heading=wrap(odom[2] + det.body_heading_rel[j]),
                       head_yaw=wrap(odom[2] + det.head_rel[j]) if np.isfinite(det.head_rel[j]) else np.nan,
                       appearance=det.appearance[j], gt_id=int(det.gt_id[j]))
            self.next_id += 1
            if tc["confirm_hits"] <= 1:
                t.confirmed = True
            self.tracks.append(t)
        if gt_pos is not None:
            self._inject_swaps(gt_pos)
        return self.output(odom)

    def _inject_swaps(self, gt_pos):
        """With prob swap_p * noise_scale per step, swap the IDs of two confirmed, currently seen tracks
        whose true people are within swap_dist of each other (models real tracker ID switches)."""
        p = min(1.0, self.tc["swap_p"] * self.s)
        if p <= 0:
            return
        live = [t for t in self.tracks if t.confirmed and t.time_since_seen_s == 0.0 and t.gt_id >= 0]
        done = set()
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                ta, tb = live[a], live[b]
                if ta.track_id in done or tb.track_id in done:
                    continue
                if np.hypot(*(gt_pos[ta.gt_id] - gt_pos[tb.gt_id])) < self.tc["swap_dist"] and self.rng.random() < p:
                    ta.track_id, tb.track_id = tb.track_id, ta.track_id
                    ta.age_s, tb.age_s = tb.age_s, ta.age_s
                    done.update((ta.track_id, tb.track_id))
                    self.n_swaps += 1

    def output(self, odom) -> list[dict]:
        c, s = np.cos(odom[2]), np.sin(odom[2])
        out = []
        for t in self.tracks:
            if not t.confirmed:
                continue
            d = t.x[:2] - odom[:2]
            out.append({
                "track_id": t.track_id,
                "pos_robot": np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]]),
                "pos_world_est": t.x[:2].copy(),
                "vel_world_est": t.x[2:].copy(),
                "body_heading_world_est": float(t.body_heading),
                "head_yaw_world_est": float(t.head_yaw),
                "appearance": t.appearance,
                "age_s": t.age_s,
                "time_since_seen_s": t.time_since_seen_s,
                "pos_sigma": float(np.sqrt(0.5 * (t.P[0, 0] + t.P[1, 1]))),
                "_gt_id": t.gt_id,          # METRICS ONLY: never read by any method
            })
        return out
