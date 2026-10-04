"""Following brain (Tier 1): perception obs -> target memory -> predictor -> goal (C0/C1) -> DWA.

Configured by: controller C0|C1, predictor (P0 object or learned), memory M0-M3, search S0|S1, delta_s.
Works only on robot-side information (obs dict); never touches ground truth.
"""
from __future__ import annotations

import numpy as np

from pf.control.dwa import DWAPlanner
from pf.control.search import Search
from pf.memory.target import TargetMemory
from pf.prediction.base import ConstantVelocity, TargetHistory
from pf.world.robot import wrap


class FollowerBrain:
    def __init__(self, cfg: dict, controller="C1", predictor=None, memory="M2", search="S1",
                 delta_s: float | None = None, oracle_future=None):
        assert controller in ("C0", "C1")
        self.cfg = cfg
        self.dt = cfg["sim"]["dt"]
        self.fc = cfg["follow"]
        self.controller = controller
        self.predictor = predictor or ConstantVelocity()
        self.memory = TargetMemory(cfg["memory"], memory)
        self.search = Search(search, self.dt)
        self.delta = self.fc["delta_s"] if delta_s is None else delta_s
        self.oracle_future = oracle_future     # E6 only: callable(t) -> Forecast from TRUE future positions
        self.dwa = None
        self.hist = TargetHistory()
        self.t = 0.0
        self.forecast = None                   # latest forecast (made while the target was visible)
        self.state = "init"
        self.goal = self.look = None
        self.selected = None

    # ------------------------------------------------------------------ helpers
    def _follow_point(self, pos, vel, robot_xy):
        d = self.fc["distance"]
        sp = np.hypot(*vel)
        if sp >= self.fc["min_walk_speed"]:
            u = vel / sp
        else:
            u = pos - robot_xy
            n = np.hypot(*u)
            u = u / n if n > 1e-6 else np.array([1.0, 0.0])
        return pos - d * u

    def _safe_goal(self, goal, target_pos):
        """If the follow point lies in a known obstacle, slide it toward the target until free."""
        lm = self.dwa.map
        for a in np.linspace(0.0, 1.0, 11):
            g = (1 - a) * goal + a * target_pos
            if lm.clearance(g[None])[0] > self.dwa.radius + 0.1:
                return g
        return goal

    def _make_forecast(self):
        if self.oracle_future is not None:
            return self.oracle_future(self.t)
        return self.predictor.predict(self.hist.snapshot(), self.t)

    # ------------------------------------------------------------------ main
    def step(self, obs: dict) -> np.ndarray:
        odom = obs["robot_odom"]
        if self.dwa is None:
            self.dwa = DWAPlanner(self.cfg, odom[:2])
        self.dwa.observe(obs)
        self.t += self.dt
        if not self.memory.initialized:
            self.memory.init(obs)
            self.dwa.commit(np.zeros(3))
            return np.zeros(3)

        # expected target position for association: forecast made at the last-seen time, propagated
        fc_assoc = None
        if self.forecast is not None:   # same association for every controller (fair comparison)
            p, s = self.forecast.at(self.t - self.forecast.t0)
            fc_assoc = (p, s)
        sel = self.memory.step(obs, self.dt, forecast=fc_assoc)
        self.selected = sel
        self.hist.push(sel)
        sel_id = sel["track_id"] if sel is not None else None
        others = [(t["pos_world_est"], t["vel_world_est"]) for t in obs["tracks"]
                  if t["track_id"] != sel_id and t["time_since_seen_s"] <= 0.5]

        if sel is not None:
            if self.search.active:
                self.search.stop()
            self.forecast = self._make_forecast()
            pos, vel = np.asarray(sel["pos_world_est"]), np.asarray(sel["vel_world_est"])
            if self.controller == "C0" or self.delta <= 0:
                goal = self._follow_point(pos, vel, odom[:2])
            else:
                f = self.forecast
                m = f.best
                k = int(np.clip(round(self.delta / self.dt) - 1, 0, f.modes.shape[1] - 1))
                p_pred = f.modes[m, k]
                prev = f.modes[m, k - 1] if k > 0 else pos
                v_pred = (p_pred - prev) / self.dt
                goal = self._follow_point(p_pred, v_pred, odom[:2])
            self.goal = self._safe_goal(goal, pos)
            self.look = pos
            self.state = "follow"
            cmd = self.dwa.plan(odom, self.goal, self.look, others, target=(pos, vel))
        elif not self.memory.lost:
            # briefly unseen (< 0.5 s): keep heading for the last goal, look at the propagated estimate
            mu, _ = self.memory.expected(fc_assoc)
            self.look = mu
            self.state = "coast"
            cmd = self.dwa.plan(odom, self.goal, self.look, others)
        else:
            if not self.search.active:
                fc = (self.forecast.modes, self.forecast.probs) if self.forecast is not None else None
                self.search.start(odom, self.memory.last_pos, odom[2], fc)
            self.state = "search:" + self.search.phase
            act = self.search.step(odom, self.memory.time_since_seen)
            if act[0] == "cmd":
                cmd = act[1]
                self.goal, self.look = None, None
            else:
                _, self.goal, self.look = act
                cmd = self.dwa.plan(odom, self.goal, self.look, others)
        self.dwa.commit(cmd)
        return cmd
