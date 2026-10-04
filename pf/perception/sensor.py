"""Perception simulator (Section 5): ground truth -> noisy person detections + depth rays.

Works on a plain `GroundTruth` record so the same code serves the 2D simulator (Stage C) and Isaac
Sim (Stage D, D1 mode), where `los_fn` can be replaced by 3D ray casts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pf.world.appearance import relative_view_angle
from pf.world.robot import wrap


@dataclass
class GroundTruth:
    cam_pose: np.ndarray        # (3,) true camera x, y, yaw (with gait sway)
    robot_pose: np.ndarray      # (3,) true robot pose (camera nominally mounted here)
    odom: np.ndarray            # (3,) robot's own (drifting) pose estimate
    pos: np.ndarray             # (N, 2) people positions
    heading: np.ndarray         # (N,) body heading (world)
    head: np.ndarray            # (N,) head yaw (world)
    person_radius: float
    los_fn: Callable            # (origin (2,), points (N,2)) -> bool (N,) wall-free line of sight
    ray_fn: Callable            # (origin (2,), angles (K,), max_range) -> distances (K,)
    app_fn: Callable            # (person idx (n,), view angle (n,), extra_sigma, noise_scale) -> (n, D)
    app_dim: int = 32


@dataclass
class Detections:
    pos_robot: np.ndarray       # (n, 2) measured, robot frame
    body_heading_rel: np.ndarray  # (n,) measured, relative to robot yaw
    head_rel: np.ndarray        # (n,) measured, relative to robot yaw, NaN if unreadable
    appearance: np.ndarray      # (n, D)
    gt_id: np.ndarray           # (n,) ground-truth person index, -1 for false positives (METRICS ONLY)
    meas_sigma: np.ndarray      # (n,) position sigma used (tracker uses it as R)


def segment_point_dist(a, b, p):
    """Distance from points p (..., 2) to segment a->b (2,)."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return np.linalg.norm(p - a, axis=-1)
    t = np.clip(((p - a) @ ab) / L2, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1)


def visibility(cam_pose, pos, radius, fov, rng_min, rng_max, los_fn, margin=0.1):
    """Return (visible (N,), partial (N,), dist (N,), bearing (N,) in camera frame).

    visible: center inside FOV and range, wall-free LOS, LOS not through another person's body circle.
    partial: visible but the LOS passes within `margin` of another person's circle.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
    n = len(pos)
    cam = cam_pose[:2]
    d = pos - cam
    dist = np.hypot(d[:, 0], d[:, 1])
    bearing = wrap(np.arctan2(d[:, 1], d[:, 0]) - cam_pose[2])
    vis = (np.abs(bearing) <= fov / 2) & (dist >= rng_min) & (dist <= rng_max)
    partial = np.zeros(n, dtype=bool)
    if vis.any():
        idx = np.flatnonzero(vis)
        vis[idx] = los_fn(cam, pos[idx])
    for i in np.flatnonzero(vis):
        others = np.delete(np.arange(n), i)
        if len(others) == 0:
            continue
        # only people closer to the camera than person i can occlude it
        o = others[dist[others] < dist[i]]
        if len(o) == 0:
            continue
        sd = segment_point_dist(cam, pos[i], pos[o])
        if np.any(sd < radius):
            vis[i] = False
        elif np.any(sd < radius + margin):
            partial[i] = True
    return vis, partial, dist, bearing


class PerceptionSensor:
    def __init__(self, pc: dict, rc: dict, rng: np.random.Generator):
        self.pc = pc
        self.fov = np.deg2rad(rc["fov_deg"])
        self.rng_min, self.rng_max = rc["cam_range"]
        self.rng = rng
        self.s = float(pc["noise_scale"])

    def p_detect(self, dist):
        pc = self.pc
        f = np.clip((dist - pc["near_m"]) / (pc["far_m"] - pc["near_m"]), 0.0, 1.0)
        p = pc["p_det_near"] + f * (pc["p_det_far"] - pc["p_det_near"])
        return 1.0 - self.s * (1.0 - p)   # noise_scale 0 -> perfect detection

    def pos_sigma(self, dist):
        return self.s * (self.pc["pos_sigma0"] + self.pc["pos_sigma_k"] * dist)

    def detect(self, gt: GroundTruth) -> tuple[Detections, dict]:
        pc, rng, s = self.pc, self.rng, self.s
        vis, partial, dist, bearing = visibility(gt.cam_pose, gt.pos, gt.person_radius, self.fov,
                                                 self.rng_min, self.rng_max, gt.los_fn, pc["partial_occ_margin"])
        p = self.p_detect(dist) * np.where(partial, pc["partial_occ_factor"] if s > 0 else 1.0, 1.0)
        det = vis & (rng.random(len(dist)) < p)
        idx = np.flatnonzero(det)
        n = len(idx)
        # Measurement in the (swayed) camera frame, interpreted as if the camera were at its nominal mount
        # (= robot pose): this is how gait sway leaks into the observations.
        c, si = np.cos(gt.cam_pose[2]), np.sin(gt.cam_pose[2])
        d = gt.pos[idx] - gt.cam_pose[:2]
        cam_xy = np.stack([c * d[:, 0] + si * d[:, 1], -si * d[:, 0] + c * d[:, 1]], axis=1)
        sig = self.pos_sigma(dist[idx])
        meas = cam_xy + rng.normal(size=(n, 2)) * sig[:, None]
        yaw_off = gt.cam_pose[2]
        bh = wrap(gt.heading[idx] - yaw_off + rng.normal(0, s * np.deg2rad(pc["body_heading_sigma_deg"]), n))
        hd = wrap(gt.head[idx] - yaw_off + rng.normal(0, s * np.deg2rad(pc["head_yaw_sigma_deg"]), n))
        hd = np.where(rng.random(n) < s * pc["head_nan_p"], np.nan, hd)
        view = relative_view_angle(gt.pos[idx], gt.heading[idx], gt.cam_pose[:2])
        app = gt.app_fn(idx, view, pc["app_extra_sigma"], s) if n else np.zeros((0, gt.app_dim))
        gt_id = idx.copy()
        # false positives: random position inside the FOV/range, random embedding
        n_fp = rng.poisson(s * pc["fp_rate"]) if s > 0 else 0
        if n_fp:
            r = rng.uniform(self.rng_min, self.rng_max, n_fp)
            b = rng.uniform(-self.fov / 2, self.fov / 2, n_fp)
            fp_xy = np.stack([r * np.cos(b), r * np.sin(b)], axis=1)
            emb = rng.normal(size=(n_fp, gt.app_dim))
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            meas = np.vstack([meas, fp_xy])
            bh = np.concatenate([bh, rng.uniform(-np.pi, np.pi, n_fp)])
            hd = np.concatenate([hd, rng.uniform(-np.pi, np.pi, n_fp)])
            app = np.vstack([app, emb])
            gt_id = np.concatenate([gt_id, -np.ones(n_fp, dtype=int)])
            sig = np.concatenate([sig, self.pos_sigma(r)])
        info = {"visible": vis, "partial": partial, "dist": dist, "bearing": bearing}
        return Detections(meas, bh, hd, app, gt_id, sig), info

    def depth_rays(self, gt: GroundTruth) -> np.ndarray:
        """32 rays across the FOV: distance to walls/obstacles (people excluded), noisy, max 8 m.
        Rays are cast from the true (swayed) camera, reported at the nominal angles."""
        k = self.pc["n_depth_rays"]
        rel = np.linspace(-self.fov / 2, self.fov / 2, k)
        d = gt.ray_fn(gt.cam_pose[:2], gt.cam_pose[2] + rel, self.rng_max)
        sig = self.s * (self.pc["depth_sigma0"] + self.pc["depth_sigma_k"] * d)
        return np.clip(d + self.rng.normal(size=k) * sig, 0.0, self.rng_max)

    @property
    def ray_angles(self):
        return np.linspace(-self.fov / 2, self.fov / 2, self.pc["n_depth_rays"])
