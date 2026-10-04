"""Predictor interface (Section 7) + target history buffer + P0 (constant velocity).

History: the target's last 1.0 s (10 steps) of ESTIMATED states in the odometry frame:
  pos (10, 2), vel (10, 2), body heading (10,), head yaw (10,) (NaN allowed), visible mask (10,).
Missing steps are filled with the last value and masked (Section 7.2).
Forecast: modes (K=3, 20, 2) odometry-frame positions for t+0.1 ... t+2.0 s, probs (K,), log_sigma (K,).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIST = 10
HORIZON = 20
K = 3


@dataclass
class Forecast:
    modes: np.ndarray        # (K, HORIZON, 2)
    probs: np.ndarray        # (K,)
    log_sigma: np.ndarray    # (K,)
    t0: float                # time the forecast was made (s)

    @property
    def best(self) -> int:
        return int(np.argmax(self.probs))

    def at(self, dt_ahead: float, mode: int | None = None):
        """Position of a mode `dt_ahead` seconds after t0 (clamped to the horizon), and its sigma.
        log_sigma is the mode's uncertainty at the 2 s horizon; it grows linearly from 0 with time."""
        m = self.best if mode is None else mode
        k = int(np.clip(round(dt_ahead / 0.1) - 1, 0, HORIZON - 1))
        sig = float(np.exp(self.log_sigma[m])) * min(max(dt_ahead, 0.1), HORIZON * 0.1) / (HORIZON * 0.1)
        return self.modes[m, k], sig


class TargetHistory:
    def __init__(self):
        self.pos = np.zeros((HIST, 2))
        self.vel = np.zeros((HIST, 2))
        self.body = np.zeros(HIST)
        self.head = np.full(HIST, np.nan)
        self.vis = np.zeros(HIST, dtype=bool)
        self.n_seen = 0

    def push(self, track: dict | None):
        self.pos = np.roll(self.pos, -1, axis=0)
        self.vel = np.roll(self.vel, -1, axis=0)
        self.body = np.roll(self.body, -1)
        self.head = np.roll(self.head, -1)
        self.vis = np.roll(self.vis, -1)
        if track is not None:
            self.pos[-1] = track["pos_world_est"]
            self.vel[-1] = track["vel_world_est"]
            self.body[-1] = track["body_heading_world_est"]
            self.head[-1] = track["head_yaw_world_est"]
            self.vis[-1] = True
            if self.n_seen == 0:      # first observation: back-fill so the buffer is well defined
                self.pos[:] = self.pos[-1]
                self.vel[:] = self.vel[-1]
                self.body[:] = self.body[-1]
                self.head[:] = self.head[-1]
            self.n_seen += 1
        else:                          # missing: repeat the last value, masked
            self.pos[-1] = self.pos[-2]
            self.vel[-1] = self.vel[-2]
            self.body[-1] = self.body[-2]
            self.head[-1] = self.head[-2]
            self.vis[-1] = False

    def snapshot(self) -> dict:
        return {"pos": self.pos.copy(), "vel": self.vel.copy(), "body": self.body.copy(),
                "head": self.head.copy(), "vis": self.vis.copy()}


def smoothed_velocity(hist: dict) -> np.ndarray:
    """Least-squares velocity over the visible samples of the 1 s window (>= 3 needed), else the
    tracker's latest estimate. Young tracks have very noisy Kalman velocities."""
    vis = np.flatnonzero(hist["vis"])
    if len(vis) >= 3:
        t = vis * 0.1
        A = np.stack([t - t.mean(), np.ones_like(t)], axis=1)
        coef, *_ = np.linalg.lstsq(A, hist["pos"][vis], rcond=None)
        return coef[0]
    i = vis[-1] if len(vis) else HIST - 1
    return hist["vel"][i]


class ConstantVelocity:
    """P0: extrapolate the last estimated velocity (smoothed over the 1 s window). Single mode
    (duplicated with probability 0). sigma2: assumed 2 s-horizon uncertainty (m)."""
    name = "P0"

    def __init__(self, sigma2: float = 1.0):
        self.sigma2 = sigma2

    def predict(self, hist: dict, t0: float) -> Forecast:
        last = np.flatnonzero(hist["vis"])
        i = last[-1] if len(last) else HIST - 1
        p, v = hist["pos"][i], smoothed_velocity(hist)
        # positions relative to "now": if the last visible sample is old, extrapolate from it
        lag = (HIST - 1 - i) * 0.1
        t = (np.arange(1, HORIZON + 1) * 0.1 + lag)[:, None]
        traj = p[None] + t * v[None]
        modes = np.repeat(traj[None], K, axis=0)
        return Forecast(modes, np.array([1.0, 0.0, 0.0]), np.full(K, np.log(self.sigma2 + 0.5 * lag)), t0)
