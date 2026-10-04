"""Target memory and association (Section 6, Idea 4a).

Selection score per track: s = w_app * s_app + w_motion * s_motion
  s_app    = mean of the top-3 cosine similarities between the track's embedding and the memory
  s_motion = exp(-0.5 * d^2 / sigma^2), d = distance to the expected target position (predictor forecast
             for this time step if given, else the last known position propagated with its velocity)
Memory variants: M0 tracker-ID only, M1 fixed, M2 gated continual (proposed), M3 ungated continual.
Never reads `_gt_id`.
"""
from __future__ import annotations

import numpy as np

VARIANTS = ("M0", "M1", "M2", "M3")


class TargetMemory:
    def __init__(self, mc: dict, variant: str | None = None):
        self.mc = mc
        self.variant = variant or mc["variant"]
        assert self.variant in VARIANTS, self.variant
        self.reset()

    def reset(self):
        self.memory = np.zeros((0, 0))
        self.target_id: int | None = None      # M0: followed tracker ID
        self.last_pos = None                   # last confirmed target state (odometry/world-estimate frame)
        self.last_vel = np.zeros(2)
        self.time_since_seen = np.inf
        self.selected = None                   # track dict selected this step (or None)
        self.n_added = 0
        self.last_scores = {}

    # ------------------------------------------------------------------ init
    def init(self, obs: dict) -> bool:
        """t = 0: the confirmed track closest to the image center is the target (Section 6.1)."""
        tracks = obs["tracks"]
        if not tracks:
            return False
        bearing = [abs(np.arctan2(t["pos_robot"][1], t["pos_robot"][0])) for t in tracks]
        t = tracks[int(np.argmin(bearing))]
        self.memory = t["appearance"][None].copy()
        self.target_id = t["track_id"]
        self._accept(t)
        return True

    @property
    def initialized(self) -> bool:
        return self.memory.shape[0] > 0

    @property
    def lost(self) -> bool:
        return self.time_since_seen > self.mc["lost_after_s"]

    # ------------------------------------------------------------------ scores
    def s_app(self, emb: np.ndarray) -> np.ndarray:
        """emb (n, D) -> (n,) mean of the top-k cosine similarities to the memory."""
        sims = emb @ self.memory.T
        k = min(self.mc["top_k"], sims.shape[1])
        return np.sort(sims, axis=1)[:, -k:].mean(axis=1)

    def expected(self, forecast=None):
        """(position, sigma) where the target should be now."""
        mc = self.mc
        dt = 0.0 if not np.isfinite(self.time_since_seen) else self.time_since_seen
        # uncertainty keeps growing while the target is unseen (also beyond the forecast horizon)
        grow = mc["sigma_min"] + mc["sigma_growth"] * dt
        if forecast is not None:
            return np.asarray(forecast[0]), max(grow, float(forecast[1]))
        if self.last_pos is None:
            return None, None
        return self.last_pos + self.last_vel * dt, grow

    # ------------------------------------------------------------------ per-step
    def step(self, obs: dict, dt: float, forecast=None):
        """Select the target among the current tracks. Returns the selected track dict or None.
        `forecast`: optional (pos (2,), sigma) from the predictor for the current time."""
        mc = self.mc
        tracks = [t for t in obs["tracks"] if t["time_since_seen_s"] == 0.0]   # seen this step
        self.time_since_seen += dt
        self.selected = None
        if not tracks or not self.initialized:
            return None
        if self.variant == "M0":
            return self._step_m0(obs["tracks"])

        emb = np.stack([t["appearance"] for t in tracks])
        pos = np.stack([t["pos_world_est"] for t in tracks])
        sa = self.s_app(emb)
        mu, sig = self.expected(forecast)
        if mu is None:
            sm = np.zeros(len(tracks))
        else:
            d2 = np.sum((pos - mu) ** 2, axis=1)
            sm = np.exp(-0.5 * d2 / sig ** 2)
        s = mc["w_app"] * sa + mc["w_motion"] * sm
        i = int(np.argmax(s))
        self.last_scores = {"s": float(s[i]), "s_app": float(sa[i]), "s_motion": float(sm[i])}
        ok = s[i] > mc["tau_accept"]
        if self.lost:
            ok = ok and sa[i] > mc["tau_reacquire"]
        if not ok:
            return None
        t = tracks[i]
        self._update_memory(t, sa[i], sm[i], pos, i)
        self._accept(t)
        return t

    def _step_m0(self, all_tracks):
        """M0: follow the tracker ID from t = 0; if that ID disappears, take the nearest track."""
        by_id = {t["track_id"]: t for t in all_tracks}
        t = by_id.get(self.target_id)
        if t is None:
            seen = [t for t in all_tracks if t["time_since_seen_s"] == 0.0]
            if not seen or self.last_pos is None:
                return None
            mu, _ = self.expected()
            t = min(seen, key=lambda tr: np.hypot(*(tr["pos_world_est"] - mu)))
            self.target_id = t["track_id"]
        if t["time_since_seen_s"] > 0.0:
            return None
        self._accept(t)
        return t

    def _accept(self, t):
        self.selected = t
        self.last_pos = np.asarray(t["pos_world_est"], float).copy()
        self.last_vel = np.asarray(t["vel_world_est"], float).copy()
        self.time_since_seen = 0.0
        self.target_id = t["track_id"]

    def _update_memory(self, t, s_app, s_motion, pos, i):
        mc = self.mc
        e = t["appearance"]
        if self.variant == "M1":
            return
        if self.variant == "M3":
            self._add(e)
            return
        # M2: gated
        if s_app <= mc["tau_high"] or s_motion <= mc["motion_min"]:
            return
        others = np.delete(pos, i, axis=0)
        if len(others) and np.min(np.hypot(*(others - pos[i]).T)) < mc["isolation_m"]:
            return
        if np.max(self.memory @ e) >= mc["novelty_cos"]:
            return
        self._add(e)

    def _add(self, e):
        self.n_added += 1
        if self.memory.shape[0] < self.mc["max_size"]:
            self.memory = np.vstack([self.memory, e])
            return
        if self.variant == "M3":
            # ungated continual memory = rolling window of the most recent embeddings (ASSUMPTIONS A-MEM-2)
            self.memory = np.vstack([self.memory[1:], e])
            return
        # M2: replace the most redundant stored embedding (highest similarity to another stored one)
        G = self.memory @ self.memory.T
        np.fill_diagonal(G, -np.inf)
        j = int(np.argmax(G.max(axis=1)))
        self.memory[j] = e
