"""Learned multi-modal predictor (Section 7): small GRU over the 1 s history, K = 3 modes x 20 steps,
mode probabilities and a per-mode log sigma. Variants: P1 (position/velocity only), P2 (+ body heading
and head yaw relative to the walking direction).

Frame (rotation invariance, Section 7.2): origin = last observed position; x axis = walking direction
(least-squares velocity over the window if > 0.15 m/s, else the last body-heading estimate).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pf.prediction.base import HIST, HORIZON, K, Forecast

VARIANT_FEATURES = {"P1": 5, "P2": 10}


def frame_of(pos, vel, body, vis):
    """Per-sample frame (origin (N,2), angle (N,)). Arrays: pos/vel (N,10,2), body (N,10), vis (N,10)."""
    N = len(pos)
    last = HIST - 1 - np.argmax(vis[:, ::-1], axis=1)                  # index of last visible step
    origin = pos[np.arange(N), last]
    # least-squares velocity over visible steps
    t = np.arange(HIST) * 0.1
    w = vis.astype(float)
    sw = np.maximum(w.sum(1), 1e-9)
    tm = (w * t).sum(1) / sw
    pm = (w[..., None] * pos).sum(1) / sw[:, None]
    num = (w[..., None] * (t[None, :, None] - tm[:, None, None]) * (pos - pm[:, None])).sum(1)
    den = (w * (t[None] - tm[:, None]) ** 2).sum(1)[:, None]
    v = np.where(den > 1e-9, num / np.maximum(den, 1e-9), vel[np.arange(N), last])
    sp = np.hypot(v[:, 0], v[:, 1])
    ang = np.where(sp > 0.15, np.arctan2(v[:, 1], v[:, 0]), body[np.arange(N), last])
    return origin, ang


def rotate(xy, ang):
    """Rotate (N, ..., 2) by -ang (world -> local)."""
    c, s = np.cos(ang), np.sin(ang)
    shape = (-1,) + (1,) * (xy.ndim - 2)
    c, s = c.reshape(shape), s.reshape(shape)
    return np.stack([c * xy[..., 0] + s * xy[..., 1], -s * xy[..., 0] + c * xy[..., 1]], axis=-1)


def unrotate(xy, ang):
    return rotate(xy, -ang)


def features(d: dict, variant: str):
    """Build (N, 10, F) input features and the frame. d: dict with pos, vel, body, head, vis."""
    pos, vel, body, head, vis = d["pos"], d["vel"], d["body"], d["head"], d["vis"]
    origin, ang = frame_of(pos, vel, body, vis)
    p = rotate(pos - origin[:, None], ang)
    v = rotate(vel, ang)
    m = vis.astype(np.float32)[..., None]
    f = [p, v, m]
    if variant == "P2":
        rb = body - ang[:, None]
        hm = np.isfinite(head)
        rh = np.where(hm, head, 0.0) - ang[:, None]
        f += [np.sin(rb)[..., None], np.cos(rb)[..., None],
              (np.sin(rh) * hm)[..., None], (np.cos(rh) * hm)[..., None], hm[..., None].astype(float)]
    x = np.concatenate(f, axis=-1).astype(np.float32)
    # masked (missing) steps: features already hold the last value; also zero them out with the mask flag
    return x, origin, ang


class PredNet(nn.Module):
    def __init__(self, n_in, hidden=128):
        super().__init__()
        self.inp = nn.Linear(n_in, 64)
        self.gru = nn.GRU(64, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, K * HORIZON * 2 + K + K))

    def forward(self, x):
        h, _ = self.gru(torch.relu(self.inp(x)))
        o = self.head(h[:, -1])
        traj = o[:, : K * HORIZON * 2].view(-1, K, HORIZON, 2)
        traj = torch.cumsum(traj, dim=2)          # predict per-step displacements
        logits = o[:, K * HORIZON * 2: K * HORIZON * 2 + K]
        log_sigma = o[:, -K:].clamp(-3.0, 2.0)
        return traj, logits, log_sigma


def wta_loss(traj, logits, log_sigma, y, w_cls=0.5, w_nll=0.1):
    """Winner-takes-all regression on the best mode + CE on mode probabilities + NLL for log sigma."""
    err = torch.linalg.norm(traj - y[:, None], dim=-1)            # (B, K, H)
    ade = err.mean(-1)                                             # (B, K)
    best = ade.argmin(1)
    idx = torch.arange(len(y))
    reg = ade[idx, best].mean()
    cls = nn.functional.cross_entropy(logits, best)
    ls = log_sigma[idx, best]
    e2 = (err[idx, best] ** 2).mean(-1).detach()                   # sigma models the RMS error of the best mode
    nll = (e2 / (2 * torch.exp(2 * ls)) + ls).mean()
    return reg + w_cls * cls + w_nll * nll, reg


class LearnedPredictor:
    """Wraps a trained PredNet for the brain: predict(hist_snapshot, t0) -> Forecast in the odometry frame."""

    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.variant = ck["variant"]
        self.name = self.variant
        self.net = PredNet(VARIANT_FEATURES[self.variant], ck.get("hidden", 128))
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        torch.set_num_threads(1)

    @torch.no_grad()
    def predict_batch(self, d: dict):
        x, origin, ang = features(d, self.variant)
        traj, logits, log_sigma = self.net(torch.from_numpy(x))
        probs = torch.softmax(logits, -1).numpy()
        traj = traj.numpy()
        world = unrotate(traj, ang) + origin[:, None, None]
        # sigma: RMS error over the horizon -> convert to the 2 s-horizon value used by Forecast.at (~1.5x)
        return world, probs, log_sigma.numpy() + np.log(1.5)

    def predict(self, hist: dict, t0: float) -> Forecast:
        d = {k: v[None] for k, v in hist.items()}
        last = np.flatnonzero(hist["vis"])
        lag = (HIST - 1 - last[-1]) * 0.1 if len(last) else 0.0
        modes, probs, ls = self.predict_batch(d)
        modes = modes[0]
        if lag > 0:   # history ends with missing steps: shift the forecast so index 0 is "now + 0.1 s"
            k = int(round(lag / 0.1))
            modes = np.concatenate([modes[:, k:], np.repeat(modes[:, -1:], k, axis=1)], axis=1)
        return Forecast(modes, probs[0], ls[0], t0)
