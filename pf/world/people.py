"""Simulated people (Section 4.4): purposeful POI-to-POI walking on A* paths, simple social force,
body heading, and the SYNTHETIC head-turn lead cue + random glances.

Positions/velocities/headings are stored as arrays (N, ...) so perception can read them in one go.
Per-person path bookkeeping is in small `_Agent` records.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pf.world.planning import PathPlanner
from pf.world.robot import wrap


@dataclass
class _Agent:
    path: np.ndarray | None = None
    wp: int = 1                      # index of the next path vertex
    goal_poi: int | None = None
    cur_poi: int | None = None
    pause_left: float = 0.0
    next_path: np.ndarray | None = None
    next_goal: int | None = None
    lead_s: float = 0.5              # head lead for the upcoming turn
    lead_for: tuple = (-1, -1)       # (path id, vertex) the lead sample belongs to
    glance_left: float = 0.0
    glance_total: float = 0.0
    glance_off: float = 0.0
    route: list = field(default_factory=list)   # scripted [(point(2,), pause_s), ...]
    stuck_t: float = 0.0
    best_goal_d: float = np.inf
    path_id: int = 0
    head_mode: str = "body"          # body | lead | glance (for logging / tests)


class People:
    def __init__(self, world_map, pc: dict, dt: float, rng: np.random.Generator, planner: PathPlanner | None = None):
        self.map = world_map
        self.grid = world_map.grid
        self.pc = pc
        self.dt = dt
        self.rng = rng
        self.planner = planner or PathPlanner(self.grid, pc["plan_inflation"], pc["plan_res"])
        self.radius = pc["radius"]
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.heading = np.zeros(0)
        self.head = np.zeros(0)
        self.pref = np.zeros(0)
        self.is_target = np.zeros(0, dtype=bool)
        self.agents: list[_Agent] = []
        self._path_counter = 0
        self.t = 0.0

    @property
    def n(self):
        return len(self.agents)

    # ------------------------------------------------------------------ spawning
    def add(self, pos, heading=0.0, speed=None, target=False, poi=None, route=None, start_pause=0.0):
        lo, hi = self.pc["target_speed"] if target else self.pc["other_speed"]
        speed = float(self.rng.uniform(lo, hi)) if speed is None else float(speed)
        self.pos = np.vstack([self.pos, np.asarray(pos, float)[None]])
        self.vel = np.vstack([self.vel, np.zeros((1, 2))])
        self.heading = np.append(self.heading, heading)
        self.head = np.append(self.head, heading)
        self.pref = np.append(self.pref, speed)
        self.is_target = np.append(self.is_target, target)
        ag = _Agent(cur_poi=poi, route=list(route or []), pause_left=start_pause)
        self.agents.append(ag)
        i = self.n - 1
        if start_pause > 0:
            self._plan_next(i)
        else:
            self._start_next(i)
        return i

    # ------------------------------------------------------------------ goals / paths
    def _choose_goal(self, i):
        ag = self.agents[i]
        if ag.route:
            pt, pause = ag.route[0]
            return None, np.asarray(pt, float)
        n = len(self.map.pois)
        choices = [k for k in range(n) if k != ag.cur_poi]
        # prefer goals that are not trivially close
        far = [k for k in choices if np.hypot(*(self.map.pois[k] - self.pos[i])) > 2.0]
        k = int(self.rng.choice(far or choices))
        return k, self.map.pois[k]

    def _plan(self, i, start, start_poi):
        """Plan from `start` to a new goal. Returns (path, goal_poi) or (None, None)."""
        for _ in range(5):
            gk, gpt = self._choose_goal(i)
            key = (start_poi, gk) if (start_poi is not None and gk is not None) else None
            path = self.planner.path(start, gpt, key=key)
            if path is not None and len(path) >= 2:
                return path, gk
            if self.agents[i].route:   # scripted goal unreachable: drop it
                self.agents[i].route.pop(0)
        return None, None

    def _plan_next(self, i):
        ag = self.agents[i]
        ag.next_path, ag.next_goal = self._plan(i, self.pos[i], ag.cur_poi)

    def _start_next(self, i):
        ag = self.agents[i]
        if ag.next_path is None:
            self._plan_next(i)
        ag.path, ag.goal_poi = ag.next_path, ag.next_goal
        ag.next_path = ag.next_goal = None
        ag.wp = 1
        ag.stuck_t = 0.0
        ag.best_goal_d = np.inf
        self._path_counter += 1
        ag.path_id = self._path_counter

    def _arrive(self, i):
        ag = self.agents[i]
        if ag.route:
            _, pause = ag.route.pop(0)
            ag.cur_poi = None
        else:
            lo, hi = self.pc["pause_s"]
            pause = float(self.rng.uniform(lo, hi))
            ag.cur_poi = ag.goal_poi
        ag.pause_left = pause
        ag.path = None
        self._plan_next(i)

    # ------------------------------------------------------------------ geometry helpers
    @staticmethod
    def _carrot(path, wp, p, look):
        """Point `look` meters ahead along the path from the projection of p on segment wp-1 -> wp."""
        a, b = path[wp - 1], path[wp]
        ab = b - a
        L = np.hypot(*ab)
        s = 0.0 if L < 1e-9 else float(np.clip(np.dot(p - a, ab) / (L * L), 0.0, 1.0)) * L
        remaining = look
        seg = wp
        pos_on = s
        while True:
            a, b = path[seg - 1], path[seg]
            L = np.hypot(*(b - a))
            if pos_on + remaining <= L or seg == len(path) - 1:
                f = min(1.0, (pos_on + remaining) / L) if L > 1e-9 else 1.0
                return a + (b - a) * f
            remaining -= L - pos_on
            pos_on = 0.0
            seg += 1

    def _upcoming_turn(self, i):
        """(time until the body starts turning, new direction) for the next >30 deg path corner,
        or for the start of the next path when pausing. None if no such turn is coming."""
        ag = self.agents[i]
        thr = np.deg2rad(self.pc["lead_turn_deg"])
        if ag.path is None:
            if ag.next_path is None or len(ag.next_path) < 2:
                return None
            d = ag.next_path[1] - ag.next_path[0]
            new = np.arctan2(d[1], d[0])
            if abs(wrap(new - self.heading[i])) > thr:
                return ag.pause_left, new, ("pause", ag.path_id)
            return None
        path, wp = ag.path, ag.wp
        p = self.pos[i]
        speed = max(self.pref[i], 0.1)
        look = self.pc["lookahead"]
        # Reference direction = current body heading. A "turn" is the first vertex where the path
        # direction differs from it by > 30 deg; it is anchored at the start of the bend (first vertex
        # with a noticeable direction change), since bends can be split across several vertices.
        a0 = self.heading[i]
        dist = np.hypot(*(path[wp] - p))
        bend_start, bend_dist = None, 0.0
        prev = np.arctan2(*(path[wp] - p)[::-1])
        for k in range(wp, len(path) - 1):
            dout = path[k + 1] - path[k]
            a_out = np.arctan2(dout[1], dout[0])
            if bend_start is None and abs(wrap(a_out - prev)) > np.deg2rad(5):
                bend_start, bend_dist = k, dist
            if abs(wrap(a_out - a0)) > thr:
                if bend_start is None or dist - bend_dist > 1.5:
                    bend_start, bend_dist = k, dist
                return max(0.0, (bend_dist - look) / speed), a_out, (ag.path_id, bend_start)
            prev = a_out
            dist += np.hypot(*dout)
            if dist > 3.0 * speed + look + 1.5:
                break
        return None

    # ------------------------------------------------------------------ step
    def step(self, robot_xy=None, robot_radius=0.2):
        pc, dt = self.pc, self.dt
        n = self.n
        if n == 0:
            return
        v_des = np.zeros((n, 2))
        for i, ag in enumerate(self.agents):
            if ag.path is None:
                ag.pause_left -= dt
                if ag.pause_left <= 0:
                    self._start_next(i)
                if ag.path is None:
                    continue
            # advance waypoints
            while ag.wp < len(ag.path) - 1:
                a, b = ag.path[ag.wp - 1], ag.path[ag.wp]
                ab = b - a
                past = np.dot(self.pos[i] - a, ab) >= np.dot(ab, ab) - 1e-9
                if past or np.hypot(*(b - self.pos[i])) < 0.15:
                    ag.wp += 1
                else:
                    break
            goal = ag.path[-1]
            dg = np.hypot(*(goal - self.pos[i]))
            if dg < 0.15:
                self._arrive(i)
                continue
            c = self._carrot(ag.path, ag.wp, self.pos[i], pc["lookahead"])
            d = c - self.pos[i]
            dn = np.hypot(*d)
            if dn > 1e-6:
                v_des[i] = d / dn * self.pref[i] * min(1.0, dg / 0.5 + 0.3)
            # stuck detection -> replan to a new goal
            if dg < ag.best_goal_d - 0.3:
                ag.best_goal_d, ag.stuck_t = dg, 0.0
            else:
                ag.stuck_t += dt
                if ag.stuck_t > pc["stuck_replan_s"]:
                    ag.cur_poi = None
                    ag.route = ag.route[1:] if ag.route else ag.route
                    self._plan_next(i)
                    self._start_next(i)

        F = self._forces(v_des, robot_xy, robot_radius)
        moving = np.array([ag.path is not None for ag in self.agents])
        v = self.vel + ((v_des - self.vel) / pc["tau_vel"] + F) * dt
        v[~moving] = 0.0
        sp = np.hypot(v[:, 0], v[:, 1])
        vmax = pc["max_speed_factor"] * self.pref
        scale = np.where(sp > vmax, vmax / np.maximum(sp, 1e-9), 1.0)
        v *= scale[:, None]
        new = self.pos + v * dt
        # wall guard: slide along walls, never enter them
        clear = self.grid.clearance(new)
        bad = clear < 0.9 * self.radius
        if bad.any():
            g = self.grid.clearance_grad(self.pos[bad])
            gn = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
            vb = v[bad]
            into = np.minimum((vb * gn).sum(1), 0.0)
            vb = vb - into[:, None] * gn
            nb = self.pos[bad] + vb * dt
            still = self.grid.clearance(nb) < 0.9 * self.radius
            vb[still] = 0.0
            nb[still] = self.pos[bad][still]
            v[bad], new[bad] = vb, nb
        self.pos, self.vel = new, v
        self._update_orientation(moving)
        self.t += dt

    def _forces(self, v_des, robot_xy, robot_radius):
        pc = self.pc
        n = self.n
        F = np.zeros((n, 2))
        r = self.radius
        if n > 1:
            diff = self.pos[:, None, :] - self.pos[None, :, :]       # i - j
            d = np.linalg.norm(diff, axis=-1) + np.eye(n) * 1e9
            nij = diff / np.maximum(d, 1e-9)[..., None]
            mag = pc["agent_A"] * np.exp((2 * r - d) / pc["agent_B"])
            mag = np.where(d < pc["agent_range"], mag, 0.0)
            # anisotropy: people react more to others in front of them
            spd = np.linalg.norm(v_des, axis=1, keepdims=True)
            e = np.where(spd > 1e-6, v_des / np.maximum(spd, 1e-9), 0.0)
            cos_phi = -(e[:, None, :] * nij).sum(-1)                   # j in front of i -> +1
            w = 0.3 + 0.7 * (1 + cos_phi) / 2
            F += ((w * mag)[..., None] * nij).sum(1)
            # pass-on-the-right tangential term for people ahead (breaks head-on deadlocks)
            right = np.stack([e[:, 1], -e[:, 0]], axis=-1)
            ahead = (cos_phi > 0.3) & (d < 1.5)
            tmag = 0.6 * pc["agent_A"] * np.exp((2 * r - d) / (3 * pc["agent_B"])) * ahead
            F += tmag.sum(1)[:, None] * right
        # walls
        c = self.grid.clearance(self.pos)
        near = c < pc["wall_range"]
        if near.any():
            g = self.grid.clearance_grad(self.pos[near])
            gn = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
            F[near] += pc["wall_A"] * np.exp((r - c[near]) / pc["wall_B"])[:, None] * gn
        # robot: minimal repulsion only to avoid physical overlap
        if robot_xy is not None:
            diff = self.pos - np.asarray(robot_xy)[None, :2]
            d = np.linalg.norm(diff, axis=1)
            close = d < r + robot_radius + 0.15
            if close.any():
                nr = diff[close] / np.maximum(d[close], 1e-9)[:, None]
                F[close] += pc["robot_A"] * np.exp((r + robot_radius - d[close]) / pc["robot_B"]).clip(max=20.0)[:, None] * nr
        return F

    def _update_orientation(self, moving):
        pc, dt = self.pc, self.dt
        sp = np.hypot(self.vel[:, 0], self.vel[:, 1])
        walk_dir = np.arctan2(self.vel[:, 1], self.vel[:, 0])
        turn = wrap(walk_dir - self.heading)
        step = np.clip(pc["body_smooth"] * turn, -pc["body_rate"] * dt, pc["body_rate"] * dt)
        self.heading = wrap(self.heading + np.where(sp > 0.1, step, 0.0))
        # head target
        head_tgt = self.heading.copy()
        for i, ag in enumerate(self.agents):
            up = self._upcoming_turn(i)
            if up is not None:
                t_turn, new_dir, key = up
                if key != ag.lead_for:
                    ag.lead_for = key
                    ag.lead_s = float(self.rng.uniform(*pc["lead_s"]))
                if t_turn <= ag.lead_s:
                    head_tgt[i] = new_dir
                    ag.head_mode = "lead"
                    ag.glance_left = 0.0
                    continue
            # random glances while walking (cue imperfection)
            if ag.glance_left > 0:
                ag.glance_left -= dt
                half = ag.glance_total / 2
                if ag.glance_left > half:
                    head_tgt[i] = self.heading[i] + ag.glance_off
                ag.head_mode = "glance"
            elif moving[i] and sp[i] > 0.1 and self.rng.random() < pc["glance_rate_hz"] * dt:
                ag.glance_total = ag.glance_left = float(self.rng.uniform(*pc["glance_s"]))
                ag.glance_off = np.deg2rad(self.rng.uniform(*pc["glance_deg"])) * self.rng.choice([-1, 1])
                head_tgt[i] = self.heading[i] + ag.glance_off
                ag.head_mode = "glance"
            else:
                ag.head_mode = "body"
        dh = np.clip(wrap(head_tgt - self.head), -pc["head_rate"] * dt, pc["head_rate"] * dt)
        self.head = wrap(self.head + dh)
