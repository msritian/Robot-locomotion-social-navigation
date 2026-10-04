"""Tier 1 world: map + robot stand-in + people + appearance, stepped at 10 Hz.

Ground truth lives here and is only read by perception (to produce noisy observations) and by
metrics / training labels. Methods never see `World` directly.
"""
from __future__ import annotations

import numpy as np

from pf.world.appearance import AppearanceModel
from pf.world.maps import FAMILIES, Map, generate_map
from pf.world.people import People
from pf.world.planning import PathPlanner
from pf.world.robot import Robot, wrap

_PLANNER_CACHE: dict = {}


def planner_for(m: Map, pc: dict) -> PathPlanner:
    """Planners (with their POI-path caches) are reused across episodes on the same map."""
    key = (m.family, m.seed)
    if key not in _PLANNER_CACHE:
        if len(_PLANNER_CACHE) > 64:
            _PLANNER_CACHE.clear()
        _PLANNER_CACHE[key] = PathPlanner(m.grid, pc["plan_inflation"], pc["plan_res"])
    return _PLANNER_CACHE[key]


class World:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dt = cfg["sim"]["dt"]
        self.max_steps = int(round(cfg["sim"]["episode_s"] / self.dt))

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int, family: str | None = None, n_others: int | None = None,
              world_map: Map | None = None, spawn: bool = True):
        """Default random episode. Scenario scripts call reset(spawn=False) and place things themselves."""
        self.seed = seed
        self.rng = np.random.default_rng([seed, 7])
        if world_map is None:
            family = family or FAMILIES[int(self.rng.integers(len(FAMILIES)))]
            world_map = generate_map(family, seed, self.cfg)
        self.map = world_map
        self.grid = world_map.grid
        self.robot = Robot(self.cfg["robot"], self.dt, np.random.default_rng([seed, 11]))
        self.people = People(world_map, self.cfg["people"], self.dt, np.random.default_rng([seed, 13]),
                             planner=planner_for(world_map, self.cfg["people"]))
        self.app = AppearanceModel(self.cfg["appearance"], np.random.default_rng([seed, 17]))
        self.app_idx: list[int] = []
        self.step_i = 0
        self.t = 0.0
        self.hist = {"robot": [], "odom": [], "cmd": [], "pos": [], "vel": [], "heading": [], "head": [],
                     "collided": [], "robot_person_d": []}
        if spawn:
            if n_others is None:
                n_others = int(self.rng.integers(0, 5))
            self.spawn_default(n_others)
        return self

    def add_person(self, pos, target=False, base=None, **kw) -> int:
        i = self.people.add(pos, target=target, **kw)
        self.app_idx.append(self.app.add_person(base))
        return i

    def walkable_points(self, clearance=0.4):
        free = ~self.grid.inflated(clearance)
        iy, ix = np.nonzero(free)
        return np.stack([(ix + 0.5) * self.grid.res, (iy + 0.5) * self.grid.res], axis=1)

    def spawn_default(self, n_others: int):
        """Target 1.5-2.5 m in front of the robot inside the FOV (Section 6.1), others elsewhere."""
        rng = self.rng
        pts = self.walkable_points(0.45)
        fov = np.deg2rad(self.cfg["robot"]["fov_deg"])
        for _ in range(2000):
            tp = pts[rng.integers(len(pts))]
            th = rng.uniform(-np.pi, np.pi)
            d = rng.uniform(1.5, 2.5)
            rp = tp - d * np.array([np.cos(th), np.sin(th)])
            if self.grid.clearance(rp) < 0.45:
                continue
            if not self.grid.segment_free(rp, tp, 0.25):
                continue
            yaw = th + rng.uniform(-1, 1) * (fov / 2 - np.deg2rad(15))
            break
        else:
            raise RuntimeError("could not place robot/target")
        self.robot.reset(np.array([rp[0], rp[1], wrap(yaw)]))
        self.add_person(tp, target=True, heading=th, start_pause=float(rng.uniform(0.5, 2.0)))
        for _ in range(n_others):
            for _try in range(500):
                p = pts[rng.integers(len(pts))]
                if np.hypot(*(p - rp)) > 2.5 and np.hypot(*(p - tp)) > 1.5 and \
                        all(np.hypot(*(p - q)) > 1.0 for q in self.people.pos) and \
                        not self._in_initial_view(p):
                    break
            self.add_person(p, target=False, heading=rng.uniform(-np.pi, np.pi),
                            start_pause=float(rng.uniform(0.0, 1.0)))

    def _in_initial_view(self, p, margin_deg=10.0):
        """True if p could be seen at t = 0 (inside FOV + margin, in range, wall-free line of sight).
        Others start out of view so that the t = 0 target choice (Section 6.1) is unambiguous."""
        x, y, yaw = self.robot.pose
        d = np.asarray(p) - [x, y]
        b = abs((np.arctan2(d[1], d[0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
        fov = np.deg2rad(self.cfg["robot"]["fov_deg"] + 2 * margin_deg)
        if b > fov / 2 or np.hypot(*d) > self.cfg["robot"]["cam_range"][1]:
            return False
        return bool(self.grid.segment_free(np.array([x, y]), np.asarray(p, float), 0.0))

    @property
    def target_index(self) -> int:
        return int(np.flatnonzero(self.people.is_target)[0])

    # ------------------------------------------------------------------ step
    def step(self, cmd):
        self.robot.step(cmd, self.grid)
        # robot-person distances right after the robot moved, before people react: person collisions are
        # measured here so they reflect the robot's own motion (people never walk into the robot)
        rp = np.hypot(*(self.people.pos - self.robot.pose[:2]).T) if self.people.n else np.zeros(0)
        self.people.step(self.robot.pose[:2], self.robot.radius)
        self.step_i += 1
        self.t += self.dt
        h = self.hist
        h["robot"].append(self.robot.pose.copy())
        h["odom"].append(self.robot.odom.copy())
        h["cmd"].append(self.robot.last_cmd.copy())
        h["pos"].append(self.people.pos.copy())
        h["vel"].append(self.people.vel.copy())
        h["heading"].append(self.people.heading.copy())
        h["head"].append(self.people.head.copy())
        h["collided"].append(self.robot.collided)
        h["robot_person_d"].append(rp)
        return self.step_i >= self.max_steps

    def history_arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.hist.items()}
