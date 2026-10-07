"""Search when the target is lost (Section 8.4, Idea 4b). Triggered after > 0.5 s unseen.

S0 stop-and-scan: stop, rotate toward the side where the target was last seen, then a full 360 deg turn
                  (and keep scanning).
S1 predictive:    (1) move toward the last forecast's most likely position, propagated along its trajectory
                  for the elapsed time; (2) after 3 s, go to the last-seen position and scan; (3) then try the
                  other forecast modes in order of probability; (4) give up after 20 s (stop and wait).
Re-acquisition (the stricter appearance check) is done by TargetMemory; search only moves the robot.
Each step returns either ("goal", goal_xy, look_xy) for the DWA, or ("cmd", cmd) for in-place rotation.
"""
from __future__ import annotations

import numpy as np

from pf.world.robot import wrap

SCAN_WZ = 0.8          # rad/s in-place scanning rate
MODE_TIME = 3.0        # s spent on each forecast mode
GIVE_UP = 20.0         # s


class Search:
    def __init__(self, variant: str, dt: float):
        assert variant in ("S0", "S1"), variant
        self.variant = variant
        self.dt = dt
        self.active = False
        self.patrol_pois = None      # optional (showcase): odometry-frame POIs to patrol after giving up

    def start(self, odom, last_pos, last_seen_odom_yaw, forecast):
        """forecast: (modes (K, H, 2), probs (K,)) made at the last-seen time, or None."""
        self.active = True
        self.t = 0.0
        self.last_pos = np.asarray(last_pos, float)
        self.forecast = forecast
        self.phase = "predict" if (self.variant == "S1" and forecast is not None) else "scan_side"
        self.phase_t = 0.0
        self.rotated = 0.0
        rel = wrap(np.arctan2(*(self.last_pos - odom[:2])[::-1]) - odom[2])
        self.dir = 1.0 if rel >= 0 else -1.0
        self.mode_order = list(np.argsort(-forecast[1])) if forecast is not None else []
        self.mode_i = 0

    def stop(self):
        self.active = False

    def _scan(self, odom):
        """Rotate toward the last-seen side first, then keep turning (full 360 deg and beyond)."""
        self.rotated += abs(SCAN_WZ) * self.dt
        return ("cmd", np.array([0.0, 0.0, self.dir * SCAN_WZ]))

    def step(self, odom, elapsed_since_seen):
        self.t += self.dt
        self.phase_t += self.dt
        if self.variant == "S0":
            return self._scan(odom)
        # ---- S1
        if self.t > GIVE_UP:
            if self.patrol_pois is None or len(self.patrol_pois) == 0:
                self.phase = "give_up"
                return ("cmd", np.zeros(3))
            # patrol (showcase extension, not in the spec's S1): visit POIs nearest to where the target was headed
            if self.phase != "patrol":
                ref = self.forecast[0][self.mode_order[0]][-1] if self.forecast is not None else self.last_pos
                d = np.hypot(*(np.asarray(self.patrol_pois) - ref).T)
                self.patrol_order = list(np.argsort(d))
                self.patrol_i, self.phase, self.phase_t = 0, "patrol", 0.0
            goal = np.asarray(self.patrol_pois[self.patrol_order[self.patrol_i % len(self.patrol_order)]])
            if np.hypot(*(goal - odom[:2])) < 0.8 or self.phase_t > 15.0:
                self.patrol_i += 1
                self.phase_t = 0.0
            return ("goal", goal, goal)
        if self.phase == "predict":
            modes, probs = self.forecast
            m = modes[self.mode_order[0]]
            k = min(len(m) - 1, max(0, int(round(elapsed_since_seen / self.dt)) - 1))
            goal = m[k]
            if self.phase_t >= MODE_TIME:
                self.phase, self.phase_t = "goto_last", 0.0
            return ("goal", goal, goal)
        if self.phase == "goto_last":
            if np.hypot(*(self.last_pos - odom[:2])) < 0.6 or self.phase_t > 4.0:
                self.phase, self.phase_t, self.rotated = "scan_last", 0.0, 0.0
            else:
                return ("goal", self.last_pos, self.last_pos)
        if self.phase == "scan_last":
            if self.rotated < 2 * np.pi:
                return self._scan(odom)
            self.phase, self.phase_t, self.mode_i = "other_modes", 0.0, 1
        if self.phase == "other_modes":
            if self.forecast is None or self.mode_i >= len(self.mode_order):
                self.phase = "scan_side"
                return self._scan(odom)
            m = self.forecast[0][self.mode_order[self.mode_i]]
            goal = m[-1]
            if self.phase_t >= MODE_TIME or np.hypot(*(goal - odom[:2])) < 0.5:
                self.mode_i += 1
                self.phase_t = 0.0
            return ("goal", goal, goal)
        return self._scan(odom)
