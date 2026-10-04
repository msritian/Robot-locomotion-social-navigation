"""Stage C perception: 'people as numbers' (Idea 7). `Perception.step(world)` -> obs dict (Section 5.4)."""
from __future__ import annotations

import numpy as np

from pf.perception.sensor import GroundTruth, PerceptionSensor, visibility
from pf.perception.tracker import Tracker

__all__ = ["Perception", "GroundTruth", "PerceptionSensor", "Tracker", "visibility", "ground_truth_from_world"]


def ground_truth_from_world(world) -> GroundTruth:
    grid = world.grid
    pr = world.people.radius

    def los_fn(origin, pts):
        return grid.segment_free(np.broadcast_to(origin, pts.shape), pts, 0.0)

    def app_fn(idx, view, extra_sigma, noise_scale):
        a = np.asarray([world.app_idx[i] for i in idx], dtype=int)
        return world.app.observe(a, view, extra_sigma, noise_scale)

    return GroundTruth(
        cam_pose=world.robot.camera_pose(), robot_pose=world.robot.pose.copy(), odom=world.robot.odom.copy(),
        pos=world.people.pos, heading=world.people.heading, head=world.people.head, person_radius=pr,
        los_fn=los_fn, ray_fn=grid.raycast, app_fn=app_fn, app_dim=world.app.dim)


class Perception:
    """Sensor + tracker. Holds no ground truth except `_gt_id` tags used by metrics."""

    def __init__(self, cfg: dict, seed: int, noise_scale: float | None = None):
        pc = dict(cfg["perception"])
        if noise_scale is not None:
            pc["noise_scale"] = noise_scale
        self.noise_scale = pc["noise_scale"]
        self.sensor = PerceptionSensor(pc, cfg["robot"], np.random.default_rng([seed, 23]))
        self.tracker = Tracker(cfg["tracker"], cfg["sim"]["dt"], pc["noise_scale"], np.random.default_rng([seed, 29]))
        self.last_info = None
        self.last_det = None

    def step_gt(self, gt: GroundTruth) -> dict:
        det, info = self.sensor.detect(gt)
        tracks = self.tracker.update(det, gt.odom, gt_pos=gt.pos)
        self.last_info, self.last_det = info, det
        return {"tracks": tracks, "depth_rays": self.sensor.depth_rays(gt), "robot_odom": gt.odom.copy(),
                "ray_angles": self.sensor.ray_angles}

    def step(self, world) -> dict:
        return self.step_gt(ground_truth_from_world(world))
