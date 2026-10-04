"""Synthetic appearance embeddings (Section 4.4).

obs = normalize(base + view_scale * view_component(relative_view_angle) + noise)
`view_component` is a fixed, smooth, random function of the view angle per person (low-order
Fourier series without a constant term, random 32-D coefficients, unit RMS norm over angles),
so the base embedding stays the person's average look.
"""
from __future__ import annotations

import numpy as np


def _unit(v, axis=-1):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)


class AppearanceModel:
    def __init__(self, ac: dict, rng: np.random.Generator):
        self.ac = ac
        self.dim = ac["dim"]
        self.K = ac["view_harmonics"]
        self.rng = rng
        self.base = np.zeros((0, self.dim))
        self.coef = np.zeros((0, 2 * self.K, self.dim))

    def _noise_std(self, sigma):
        if self.ac["noise_convention"] == "total":
            return sigma / np.sqrt(self.dim)
        return sigma

    def add_person(self, base=None) -> int:
        if base is None:
            base = _unit(self.rng.normal(size=self.dim))
        coef = self.rng.normal(size=(2 * self.K, self.dim))
        # Mean over angles of |sum_k a_k cos + b_k sin|^2 is 0.5 * sum |coef|^2 -> normalize to 1.
        coef /= np.sqrt(0.5 * np.sum(coef ** 2))
        self.base = np.vstack([self.base, base])
        self.coef = np.concatenate([self.coef, coef[None]], axis=0)
        return len(self.base) - 1

    def lookalike_base(self, of_base: np.ndarray) -> np.ndarray:
        """Base embedding with cosine similarity to `of_base` drawn from lookalike_cos (0.85-0.95)."""
        cos = self.rng.uniform(*self.ac["lookalike_cos"])
        r = self.rng.normal(size=self.dim)
        r -= r.dot(of_base) * of_base
        r = _unit(r)
        return cos * of_base + np.sqrt(1 - cos ** 2) * r

    def view_component(self, idx, angle):
        """idx: (n,) person indices, angle: (n,) relative view angle (0 = seen from the front)."""
        idx = np.asarray(idx)
        angle = np.asarray(angle, dtype=np.float64)
        feats = []
        for k in range(1, self.K + 1):
            feats += [np.cos(k * angle), np.sin(k * angle)]
        F = np.stack(feats, axis=-1)                        # (n, 2K)
        return np.einsum("nk,nkd->nd", F, self.coef[idx])

    def observe(self, idx, angle, extra_sigma: float = 0.0, noise_scale: float = 1.0):
        """Observed embedding(s). extra_sigma: perception-level per-detection noise (Section 5.2)."""
        idx = np.atleast_1d(idx)
        angle = np.atleast_1d(angle)
        v = self.base[idx] + self.ac["view_scale"] * self.view_component(idx, angle)
        s = noise_scale * np.hypot(self._noise_std(self.ac["noise_sigma"]), self._noise_std(extra_sigma))
        if s > 0:
            v = v + self.rng.normal(0.0, s, size=v.shape)
        return _unit(v)


def relative_view_angle(person_pos, person_heading, cam_pos):
    """Angle between the person's facing direction and the direction from the person to the camera.
    0 = camera sees the person's front, pi = back view."""
    d = np.asarray(cam_pos)[..., :2] - np.asarray(person_pos)
    ang = np.arctan2(d[..., 1], d[..., 0])
    return (ang - person_heading + np.pi) % (2 * np.pi) - np.pi
