"""Fixed scenario suite (Section 9). Each builder takes (cfg, seed) and returns a reset World with the
robot, target, and other people placed and scripted. Test seeds 1000-1199 give held-out layouts.

T1 sharp turn        corridor loop: target walks 4-6 m straight, turns 90 deg at a corner
T2 door exit         office: target leaves a room through the door and turns immediately into the corridor
T3 crossing occluder open hall: a person crosses between robot and target 1-2 m in front of the robot
T4 look-alike        a look-alike (cos 0.85-0.95) walks to the target's goal, crossing its path
T5 disappearance     office: target walks from the corridor into a side room and steps out of view for 3-6 s
T6 crowded corridor  corridor loop (1.6-2.0 m): 6-10 others walking in both directions (Section 19.1)
T7 random mix        random layout family, 0-8 others, random target routes (Section 19.1)
T8 dense open space  open hall: 10-15 others crossing the target's path from several directions (Section 19.1)
"""
from __future__ import annotations

import numpy as np

from pf.config import deep_merge
from pf.world.maps import FAMILIES, generate_map
from pf.world.sim import World

SCENARIOS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _place_robot(w, tp, direction, rng, d_range=(1.5, 2.5)):
    """Robot 1.5-2.5 m behind the target (opposite to `direction`), facing it, clear LOS."""
    u = _unit(direction)
    for _ in range(200):
        d = rng.uniform(*d_range)
        lat = rng.uniform(-0.15, 0.15)
        rp = tp - d * u + lat * np.array([-u[1], u[0]])
        if w.grid.clearance(rp) >= 0.45 and w.grid.segment_free(rp, tp, 0.25):
            yaw = np.arctan2(*(tp - rp)[::-1]) + rng.uniform(-0.15, 0.15)
            w.robot.reset(np.array([rp[0], rp[1], yaw]))
            return True
    return False


def _add_others(w, n, rng, avoid, min_d=2.5):
    pts = w.walkable_points(0.45)
    for _ in range(n):
        for _try in range(500):
            p = pts[rng.integers(len(pts))]
            if all(np.hypot(*(p - a)) > min_d for a in avoid) and not w._in_initial_view(p) and \
                    all(np.hypot(*(p - q)) > 1.0 for q in w.people.pos):
                break
        w.add_person(p, heading=rng.uniform(-np.pi, np.pi), start_pause=float(rng.uniform(0, 1)))


def _new_world(cfg, seed, family, map_override=None, attempt=0):
    """Map from `seed` (keeps the train/test split); placement randomness from (seed, attempt)."""
    c = deep_merge(cfg, {"maps": map_override}) if map_override else cfg
    m = generate_map(family, seed, c)
    w = World(cfg).reset(seed, world_map=m, spawn=False)
    w.rng = np.random.default_rng([seed, 7, attempt])
    return w


# ---------------------------------------------------------------------------- T1
def _loop_turn_world(cfg, seed, map_override=None, attempt=0):
    w = _new_world(cfg, seed, "loop", map_override, attempt)
    rng = w.rng
    p = w.map.params
    t = cfg["maps"]["wall_thickness"]
    ox, oy = p["outer"]
    h = p["corridor_w"] / 2
    x0, y0, x1, y1 = t, t, t + ox, t + oy
    corners = [np.array(c) for c in [(x0 + h, y0 + h), (x1 - h, y0 + h), (x1 - h, y1 - h), (x0 + h, y1 - h)]]
    c = int(rng.integers(4))
    prev, cur, nxt = corners[c - 1], corners[c], corners[(c + 1) % 4]
    d_in, d_out = _unit(cur - prev), _unit(nxt - cur)
    L = min(rng.uniform(4.0, 6.0), np.hypot(*(cur - prev)) - 3.0)
    tp = cur - d_in * L
    return w, rng, tp, d_in, cur, d_out, nxt


def build_T1(cfg, seed, attempt=0):
    w, rng, tp, d_in, cur, d_out, nxt = _loop_turn_world(cfg, seed, attempt=attempt)
    if not _place_robot(w, tp, d_in, rng):
        raise RuntimeError("T1 placement failed")
    leg2 = cur + d_out * min(5.0, np.hypot(*(nxt - cur)) - 0.5)
    w.add_person(tp, target=True, heading=np.arctan2(*d_in[::-1]), start_pause=float(rng.uniform(0.5, 1.0)),
                 route=[(cur, 0.0), (leg2, float(rng.uniform(1, 3)))])
    _add_others(w, int(rng.integers(0, 2)), rng, [tp, w.robot.pose[:2]])
    return w


# ---------------------------------------------------------------------------- T2 / T5 (office)
def _office_room(w, rng):
    reg = w.map.regions
    t = w.cfg["maps"]["wall_thickness"]
    cor = reg["corridor"]
    i = int(rng.integers(len(reg["rooms"])))
    room, door = np.asarray(reg["rooms"][i]), np.asarray(reg["doors"][i])
    above = door[1] > cor[3] - 1e-6          # room above the corridor
    s = 1.0 if above else -1.0               # +y points into the room if above
    door_in = door + np.array([0.0, s * (t / 2 + 0.7)])
    door_out = door - np.array([0.0, s * (t / 2 + 0.6)])
    return room, door, door_in, door_out, s, cor


def build_T2(cfg, seed, attempt=0):
    w = _new_world(cfg, seed, "office", attempt=attempt)
    rng = w.rng
    for _ in range(20):
        room, door, door_in, door_out, s, cor = _office_room(w, rng)
        # target starts deep in the room, 2.5-3.5 m from the door, robot behind it (further inside)
        tp = door_in + np.array([rng.uniform(-0.5, 0.5), s * rng.uniform(1.5, 2.5)])
        if w.grid.clearance(tp) < 0.45:
            continue
        direction = door_in - tp
        if not _place_robot(w, tp, direction, rng, (1.3, 2.0)):
            continue
        # after the door: turn immediately left or right along the corridor
        side = rng.choice([-1.0, 1.0])
        end_x = cor[0] + 0.8 if side < 0 else cor[2] - 0.8
        yc = (cor[1] + cor[3]) / 2
        leg = np.array([door[0] + side * min(5.0, abs(end_x - door[0])), yc])
        w.add_person(tp, target=True, heading=np.arctan2(*_unit(direction)[::-1]),
                     start_pause=float(rng.uniform(0.5, 1.0)),
                     route=[(door_in, 0.0), (np.array([door[0], yc]), 0.0), (leg, float(rng.uniform(1, 3)))])
        _add_others(w, int(rng.integers(0, 2)), rng, [tp, w.robot.pose[:2]])
        return w
    raise RuntimeError("T2 placement failed")


def build_T5(cfg, seed, attempt=0):
    w = _new_world(cfg, seed, "office", attempt=attempt)
    rng = w.rng
    yc = None
    for _ in range(30):
        room, door, door_in, door_out, s, cor = _office_room(w, rng)
        yc = (cor[1] + cor[3]) / 2
        side = rng.choice([-1.0, 1.0])
        tp = np.array([door[0] - side * rng.uniform(3.0, 4.5), yc])      # walking along the corridor to the door
        if w.grid.clearance(tp) < 0.45:
            continue
        if not _place_robot(w, tp, np.array([side, 0.0]), rng):
            continue
        # inside the room: step sideways behind the wall next to the door, wait 3-6 s, then come back out
        hide = door_in + np.array([side * rng.uniform(1.4, 2.0), s * rng.uniform(0.3, 0.8)])
        if w.grid.clearance(hide) < 0.45:
            hide = door_in + np.array([-side * 1.6, s * 0.5])
            if w.grid.clearance(hide) < 0.45:
                continue
        back = np.array([door[0] + side * rng.uniform(2.0, 4.0), yc])
        w.add_person(tp, target=True, heading=0.0 if side > 0 else np.pi, start_pause=float(rng.uniform(0.5, 1.0)),
                     route=[(np.array([door[0], yc]), 0.0), (door_in, 0.0), (hide, float(rng.uniform(3.0, 6.0))),
                            (door_in, 0.0), (np.array([door[0], yc]), 0.0), (back, 2.0)])
        _add_others(w, int(rng.integers(0, 2)), rng, [tp, w.robot.pose[:2]])
        return w
    raise RuntimeError("T5 placement failed")


# ---------------------------------------------------------------------------- T3 (crossing occluder)
def _straight_hall_run(w, rng, length=9.0):
    pts = w.walkable_points(0.8)
    for _ in range(500):
        a = pts[rng.integers(len(pts))]
        th = rng.uniform(-np.pi, np.pi)
        u = np.array([np.cos(th), np.sin(th)])
        b = a + length * u
        back = a - 3.0 * u
        if w.grid.clearance(b) > 0.6 and w.grid.segment_free(back, b, 0.6):
            return a, b, u
    raise RuntimeError("no straight run found")


def build_T3(cfg, seed, attempt=0):
    w = _new_world(cfg, seed, "hall", attempt=attempt)
    rng = w.rng
    tp, end, u = _straight_hall_run(w, rng)
    if not _place_robot(w, tp, u, rng):
        raise RuntimeError("T3 placement failed")
    speed = float(rng.uniform(*cfg["people"]["target_speed"]))
    w.add_person(tp, target=True, heading=np.arctan2(*u[::-1]), speed=speed, start_pause=0.5,
                 route=[(end, 2.0)])
    # crosser: passes 1-2 m in front of the robot at time tc, perpendicular to the walking direction
    rp = w.robot.pose[:2]
    tc = float(rng.uniform(4.0, 8.0))
    ahead = float(rng.uniform(1.0, 2.0))
    robot_speed_guess = speed              # the robot roughly keeps pace with the target
    xc = rp + u * (robot_speed_guess * tc + ahead)
    n = np.array([-u[1], u[0]]) * rng.choice([-1.0, 1.0])
    vc = float(rng.uniform(0.6, 1.0))
    start, stop = xc - 3.0 * n, xc + 3.0 * n
    for _ in range(4):                     # shorten the crossing if a side is blocked
        if w.grid.clearance(start) > 0.4 and w.grid.clearance(stop) > 0.4:
            break
        start, stop = xc - 0.8 * (xc - start), xc + 0.8 * (stop - xc)
    delay = max(0.0, tc - np.hypot(*(xc - start)) / vc)
    w.add_person(start, heading=np.arctan2(*n[::-1]), speed=vc, start_pause=delay, route=[(stop, 1.0)])
    _add_others(w, int(rng.integers(0, 2)), rng, [tp, rp, start])
    return w


# ---------------------------------------------------------------------------- T4 (look-alike)
def build_T4(cfg, seed, attempt=0):
    from pf.memory.evaluate import lookalike_world
    w = lookalike_world(cfg, seed, n_extra=int(np.random.default_rng([seed, 3]).integers(0, 2)))
    w.lookalike_ids = [1]          # the look-alike is the person added right after the target
    return w


# ---------------------------------------------------------------------------- T6 (crowded corridor)
def build_T6(cfg, seed, attempt=0):
    """Crowded corridor (Section 19.1: 6-10 others). Target walks down a 1.6-2.0 m corridor toward a corner.
    About half the others come toward the robot from around the corner (opposite direction), the rest start
    behind the robot and overtake (same direction); start delays are staggered so the stream is continuous."""
    over = {"loop": {"corridor_w": {"train": [1.6, 2.0], "test": [1.6, 2.0]}, "figure8_p": 0.0}}
    w, rng, tp, d_in, cur, d_out, nxt = _loop_turn_world(cfg, seed, over, attempt)
    if not _place_robot(w, tp, d_in, rng):
        raise RuntimeError("T6 placement failed")
    rp = w.robot.pose[:2]
    w.add_person(tp, target=True, heading=np.arctan2(*d_in[::-1]), start_pause=float(rng.uniform(0.5, 1.0)),
                 route=[(cur, 0.0), (nxt, 0.0)])
    lat = np.array([-d_in[1], d_in[0]])
    lat_out = np.array([-d_out[1], d_out[0]])
    n_target = int(rng.integers(6, 11))
    placed = 0
    for k in range(3 * n_target):
        if placed >= n_target:
            break
        same = k % 2 == 0
        if same:
            p = rp - d_in * rng.uniform(1.8, 9.0) + lat * rng.uniform(-0.35, 0.35)
        else:
            p = cur + d_out * rng.uniform(1.0, 8.0) + lat_out * rng.uniform(-0.35, 0.35)
        if not (w.grid.clearance(p) > 0.4 and all(np.hypot(*(p - q)) > 0.8 for q in w.people.pos)
                and np.hypot(*(p - rp)) > 1.5):
            continue
        route = [(cur, 0.0), (nxt, 0.0)] if same else [(cur, 0.0), (rp - d_in * 6.0, 0.0)]
        w.add_person(p, speed=float(rng.uniform(0.45, 0.9)), start_pause=float(rng.uniform(0.0, 6.0)), route=route)
        placed += 1
    if placed < 6:
        raise RuntimeError("T6 crowd placement failed")
    return w


# ---------------------------------------------------------------------------- T8 (dense open space)
def build_T8(cfg, seed, attempt=0):
    """Dense open space (Section 19.1): open hall, target walks a long straight run; 10-15 others walk
    purposefully across the target's path from several directions (both sides, both ends), staggered."""
    w = _new_world(cfg, seed, "hall", attempt=attempt)
    rng = w.rng
    tp, end, u = _straight_hall_run(w, rng, length=10.0)
    if not _place_robot(w, tp, u, rng):
        raise RuntimeError("T8 placement failed")
    w.add_person(tp, target=True, heading=np.arctan2(*u[::-1]), start_pause=0.5, route=[(end, 2.0)])
    rp = w.robot.pose[:2]
    nrm = np.array([-u[1], u[0]])
    n_target = int(rng.integers(10, 16))
    placed = 0
    for _ in range(40 * n_target):
        if placed >= n_target:
            break
        xc = tp + u * rng.uniform(0.5, 10.0)                     # crossing point on the target's path
        kind = rng.random()
        if kind < 0.7:                                            # crossing from the left or right
            side = rng.choice([-1.0, 1.0])
            start = xc + side * nrm * rng.uniform(3.0, 7.0)
            stop = xc - side * nrm * rng.uniform(3.0, 7.0)
        else:                                                     # walking against the target (head-on lane)
            start = xc + u * rng.uniform(4.0, 8.0) + nrm * rng.uniform(-1.5, 1.5)
            stop = tp - u * 2.0 + nrm * rng.uniform(-1.5, 1.5)
        if w.grid.clearance(start) < 0.45 or w.grid.clearance(stop) < 0.45:
            continue
        b = np.arctan2(*(start - rp)[::-1]) - w.robot.pose[2]
        b = abs((b + np.pi) % (2 * np.pi) - np.pi)
        if np.hypot(*(start - rp)) < 2.5 or (b < np.deg2rad(25) and np.hypot(*(start - rp)) < 6.0):
            continue                                              # keep the t = 0 target choice unambiguous
        if not all(np.hypot(*(start - q)) > 0.8 for q in w.people.pos):
            continue
        v = float(rng.uniform(0.5, 1.0))
        w.add_person(start, heading=np.arctan2(*(stop - start)[::-1]), speed=v,
                     start_pause=float(rng.uniform(0.0, 12.0)), route=[(stop, float(rng.uniform(1, 3)))])
        placed += 1
    if placed < 10:
        raise RuntimeError("T8 crowd placement failed")
    return w


# ---------------------------------------------------------------------------- T7
def build_T7(cfg, seed, attempt=0):
    w = World(cfg)
    rng = np.random.default_rng([seed, 77, attempt])
    fam = FAMILIES[int(rng.integers(len(FAMILIES)))]
    return w.reset(seed, fam, n_others=int(rng.integers(0, 9)))     # Section 19.1: 0-8 others


BUILDERS = {"T1": build_T1, "T2": build_T2, "T3": build_T3, "T4": build_T4, "T5": build_T5, "T6": build_T6,
            "T7": build_T7, "T8": build_T8}


def build(cfg, scenario, seed, max_tries=10):
    """Build a scenario on the layout of `seed`; placement is retried with new randomness if needed."""
    last = None
    for k in range(max_tries):
        try:
            w = BUILDERS[scenario](cfg, seed, attempt=k)
            w.scenario, w.scenario_attempt = scenario, k
            return w
        except RuntimeError as e:
            last = e
    raise RuntimeError(f"{scenario} seed {seed}: {last}")
