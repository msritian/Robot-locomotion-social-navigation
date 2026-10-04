"""K1 stand-in (Section 4.3): omnidirectional base with first-order lag, latency, accel/vel limits,
gait sway on the camera, and drifting odometry.

All parameters come from the `robot` config block. They are ASSUMPTIONS until replaced with the
measured walker response (configs/walker_response.yaml, milestone M10).
"""
from __future__ import annotations

from collections import deque

import numpy as np


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class ResponseModel:
    """Command -> body velocity dynamics. Shared by the simulator and the DWA rollouts."""

    def __init__(self, rc: dict, dt: float):
        self.dt = dt
        self.lo = np.array([rc["vx_range"][0], rc["vy_range"][0], rc["wz_range"][0]], dtype=np.float64)
        self.hi = np.array([rc["vx_range"][1], rc["vy_range"][1], rc["wz_range"][1]], dtype=np.float64)
        tau = rc["tau"]
        tau = np.broadcast_to(np.asarray(tau, dtype=np.float64), (3,)).copy()
        self.alpha = 1.0 - np.exp(-dt / tau)  # exact discretization of the first-order lag
        acc = np.array([rc["acc_lin"], rc["acc_lin"], rc["acc_ang"]], dtype=np.float64)
        self.dv_max = acc * dt
        self.latency_steps = int(round(rc["latency"] / dt))

    def clamp(self, cmd):
        return np.clip(cmd, self.lo, self.hi)

    def step_vel(self, v, cmd):
        """v, cmd: (..., 3) body-frame velocity and (already delayed) command."""
        dv = self.alpha * (self.clamp(cmd) - v)
        dv = np.clip(dv, -self.dv_max, self.dv_max)
        return v + dv


def integrate_pose(pose, v, dt):
    """pose (..., 3) = (x, y, yaw); v (..., 3) body-frame (vx, vy, wz). Midpoint-yaw integration."""
    yaw_mid = pose[..., 2] + 0.5 * v[..., 2] * dt
    c, s = np.cos(yaw_mid), np.sin(yaw_mid)
    out = np.empty_like(pose)
    out[..., 0] = pose[..., 0] + (c * v[..., 0] - s * v[..., 1]) * dt
    out[..., 1] = pose[..., 1] + (s * v[..., 0] + c * v[..., 1]) * dt
    out[..., 2] = wrap(pose[..., 2] + v[..., 2] * dt)
    return out


class Robot:
    def __init__(self, rc: dict, dt: float, rng: np.random.Generator):
        self.rc = rc
        self.dt = dt
        self.rng = rng
        self.model = ResponseModel(rc, dt)
        self.radius = rc["radius"]
        self.fov = np.deg2rad(rc["fov_deg"])
        self.cam_range = tuple(rc["cam_range"])
        self.pose = np.zeros(3)
        self.vel = np.zeros(3)
        self.odom = np.zeros(3)
        self.t = 0.0
        self._queue: deque = deque()
        # per-episode systematic odometry errors (see ASSUMPTIONS.md A-ROB-ODOM)
        self.odom_scale = 1.0 + rng.normal(0.0, rc["odom_scale_sigma"])
        self.odom_yaw_sigma = np.deg2rad(rc["odom_yaw_per_m_deg"])
        self.sway_phase = rng.uniform(0, 2 * np.pi)
        self.last_cmd = np.zeros(3)
        self.collided = False

    def reset(self, pose):
        self.pose = np.asarray(pose, dtype=np.float64).copy()
        self.odom = self.pose.copy()
        self.vel[:] = 0.0
        self.t = 0.0
        self._queue = deque([np.zeros(3)] * self.model.latency_steps)
        self.last_cmd = np.zeros(3)

    def step(self, cmd, grid):
        """Advance one control step. Collisions with obstacles block motion (and are flagged)."""
        cmd = self.model.clamp(np.asarray(cmd, dtype=np.float64))
        self.last_cmd = cmd
        self._queue.append(cmd)
        eff = self._queue.popleft()
        self.vel = self.model.step_vel(self.vel, eff)
        new = integrate_pose(self.pose, self.vel, self.dt)
        self.collided = bool(grid.clearance(new[:2]) < self.radius)
        if self.collided:
            # blocked: keep rotation, cancel translation, kill linear velocity
            new[:2] = self.pose[:2]
            self.vel[:2] = 0.0
        self._update_odom(self.pose, new)
        self.pose = new
        self.t += self.dt

    def _update_odom(self, old, new):
        d_world = new[:2] - old[:2]
        dist = float(np.hypot(*d_world))
        c, s = np.cos(old[2]), np.sin(old[2])
        d_body = np.array([c * d_world[0] + s * d_world[1], -s * d_world[0] + c * d_world[1]])
        d_body = d_body * self.odom_scale + self.rng.normal(0.0, self.rc["odom_step_sigma"] * dist + 1e-12, 2)
        dyaw = wrap(new[2] - old[2]) + self.rng.normal(0.0, self.odom_yaw_sigma * np.sqrt(dist) + 1e-12)
        ce, se = np.cos(self.odom[2]), np.sin(self.odom[2])
        self.odom[0] += ce * d_body[0] - se * d_body[1]
        self.odom[1] += se * d_body[0] + ce * d_body[1]
        self.odom[2] = wrap(self.odom[2] + dyaw)

    def camera_pose(self):
        """Camera (x, y, yaw) in the world, including gait sway (only while moving)."""
        speed = np.hypot(self.vel[0], self.vel[1]) + 0.3 * abs(self.vel[2])
        amp = min(1.0, speed / 0.2)  # sway fades out when standing
        ph = 2 * np.pi * self.rc["sway_hz"] * self.t + self.sway_phase
        lat = amp * self.rc["sway_lat"] * np.sin(ph)
        dyaw = amp * np.deg2rad(self.rc["sway_yaw_deg"]) * np.sin(ph + 0.5 * np.pi)
        x, y, yaw = self.pose
        return np.array([x - np.sin(yaw) * lat, y + np.cos(yaw) * lat, wrap(yaw + dyaw)])
