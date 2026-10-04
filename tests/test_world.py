"""M3 acceptance / sanity tests for the Stage C world (Section 4 and the M3-M4 early checks)."""
import numpy as np
import pytest

from pf.config import load_config
from pf.world.appearance import AppearanceModel
from pf.world.grid import Grid, decompose_rects, rasterize
from pf.world.maps import FAMILIES, generate_map
from pf.world.robot import ResponseModel, Robot, wrap
from pf.world.sim import World

CFG = load_config()


@pytest.mark.parametrize("family", FAMILIES)
def test_maps_deterministic_and_connected(family):
    a = generate_map(family, 5, CFG)
    b = generate_map(family, 5, CFG)
    assert np.array_equal(a.grid.occ, b.grid.occ)
    assert np.allclose(a.pois, b.pois)
    assert len(a.pois) >= 4
    # POIs are walkable with the 0.3 m planning inflation
    assert np.all(a.grid.clearance(a.pois) >= 0.3 - 1e-6)


def test_held_out_layout_ranges():
    for seed in range(1000, 1020):
        m = generate_map("office", seed, CFG)
        lo, hi = CFG["maps"]["office"]["corridor_w"]["test"]
        assert m.split == "test" and lo <= m.params["corridor_w"] <= hi
        m = generate_map("loop", seed, CFG)
        lo, hi = CFG["maps"]["loop"]["corridor_w"]["test"]
        assert lo <= m.params["corridor_w"] <= hi
    assert generate_map("hall", 3, CFG).split == "train"


def test_rect_decomposition_exact():
    m = generate_map("office", 2, CFG)
    occ2 = rasterize(m.width, m.height, m.grid.res, rects_occ=m.obstacles, solid_background=False)
    assert np.array_equal(occ2, m.grid.occ)


def test_raycast_simple_room():
    occ = rasterize(10, 10, 0.05, rects_free=[(1, 1, 9, 9)])
    g = Grid(occ, 0.05)
    d = g.raycast(np.array([5.0, 5.0]), np.array([0.0, np.pi / 2, np.pi]), 8.0)
    assert np.allclose(d, [4.0, 4.0, 4.0], atol=0.05)
    assert g.raycast(np.array([2.0, 5.0]), np.array([0.0]), 3.0)[0] == pytest.approx(3.0)


def test_people_never_enter_walls_and_target_speed():
    fast, total = 0, 0
    for seed in range(12):
        w = World(CFG).reset(seed, n_others=4)
        for _ in range(600):
            w.step(np.zeros(3))
        pos = w.history_arrays()["pos"].reshape(-1, 2)
        assert w.grid.clearance(pos).min() > 0.2, "a person overlapped a wall"
        v = np.linalg.norm(w.history_arrays()["vel"][:, w.target_index], axis=1)
        fast += (v > CFG["robot"]["vx_range"][1]).sum()
        total += len(v)
    assert fast / total < 0.05, "target is faster than the robot's max speed too often"


def test_people_walk_purposefully():
    w = World(CFG).reset(0, "office", n_others=3)
    for _ in range(600):
        w.step(np.zeros(3))
    pos = w.history_arrays()["pos"]
    path_len = np.linalg.norm(np.diff(pos, axis=0), axis=2).sum(0)
    assert np.all(path_len > 8.0)  # everyone visited several goals in 60 s


def test_head_leads_body_before_turns():
    leads = []
    for seed in range(10):
        w = World(CFG).reset(seed, n_others=4)
        modes, hd = [], []
        for _ in range(600):
            w.step(np.zeros(3))
            modes.append([a.head_mode for a in w.people.agents])
            hd.append(w.people.heading.copy())
        modes, hd = np.array(modes), np.array(hd)
        for i in range(modes.shape[1]):
            m = modes[:, i] == "lead"
            rate = np.abs(wrap(np.diff(hd[:, i])))
            for s in np.flatnonzero(m[1:] & ~m[:-1]) + 1:
                after = np.flatnonzero(rate[s:] > 0.05)
                if len(after):
                    leads.append(after[0] * w.dt)
    leads = np.array(leads)
    assert len(leads) > 100
    med = np.median(leads)
    assert 0.3 <= med <= 0.8, f"median head lead {med:.2f}s outside 0.3-0.8 s"


def test_robot_response_limits_and_latency():
    rc = CFG["robot"]
    rob = Robot(rc, 0.1, np.random.default_rng(0))
    occ = rasterize(40, 40, 0.05, rects_free=[(0.5, 0.5, 39.5, 39.5)])
    g = Grid(occ, 0.05)
    rob.reset(np.array([5.0, 20.0, 0.0]))
    vs = []
    for _ in range(40):
        rob.step(np.array([0.4, 0.0, 0.0]), g)
        vs.append(rob.vel[0])
    vs = np.array(vs)
    assert vs[0] == 0.0                     # 0.1 s latency
    assert np.all(np.diff(vs) <= rc["acc_lin"] * 0.1 + 1e-9)
    assert vs[-1] == pytest.approx(0.4, abs=0.01)
    m = ResponseModel(rc, 0.1)
    assert np.allclose(m.clamp(np.array([9, 9, 9])), [0.5, 0.2, 1.0])


def test_robot_blocked_by_wall():
    occ = rasterize(5, 5, 0.05, rects_free=[(0.5, 0.5, 4.5, 4.5)])
    g = Grid(occ, 0.05)
    rob = Robot(CFG["robot"], 0.1, np.random.default_rng(0))
    rob.reset(np.array([3.5, 2.5, 0.0]))
    for _ in range(100):
        rob.step(np.array([0.5, 0, 0]), g)
    assert g.clearance(rob.pose[:2]) >= CFG["robot"]["radius"] - 1e-9


def test_odometry_drift_magnitude():
    occ = rasterize(60, 10, 0.05, rects_free=[(0.5, 0.5, 59.5, 9.5)])
    g = Grid(occ, 0.05)
    errs = []
    for s in range(50):
        rob = Robot(CFG["robot"], 0.1, np.random.default_rng(s))
        rob.reset(np.array([2.0, 5.0, 0.0]))
        for _ in range(600):
            rob.step(np.array([0.5, 0, 0]), g)
        dist = rob.pose[0] - 2.0
        errs.append(np.hypot(*(rob.odom[:2] - rob.pose[:2])) / dist)
    # ~1% of distance (scale) plus yaw-drift-induced lateral error; must be small but nonzero
    assert 0.002 < np.median(errs) < 0.05


def test_appearance_views_and_lookalikes():
    app = AppearanceModel(CFG["appearance"], np.random.default_rng(0))
    i = app.add_person()
    look = app.lookalike_base(app.base[i])
    c = float(look @ app.base[i])
    assert 0.85 <= c <= 0.95
    front = app.observe([i] * 200, np.zeros(200))
    back = app.observe([i] * 200, np.full(200, np.pi))
    same = (front[:100] * front[100:]).sum(1).mean()
    cross = (front * back).sum(1).mean()
    assert same > cross + 0.05, "front/back views should look different"
