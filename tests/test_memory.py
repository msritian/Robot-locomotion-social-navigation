"""M5 acceptance: look-alike unit test (M2 keeps the target, M3 drifts) + memory mechanics.

Constructed track stream (no world): the target T and a look-alike L (cos 0.85-0.95) walk side by side.
Phase 1 (both visible, apart): normal following.
Phase 2 (crossing): T is hidden behind L for 3 s and L stands exactly where T is expected. The appearance
  check still accepts L at times (it is a look-alike), so every variant may select it briefly.
  M3 adds L's embeddings to its memory on every such step; M2's gate rejects them (other track within
  1 m before/after, s_app below tau_high).
Phase 3 (both visible again, splitting symmetrically to either side of the predicted path, so the motion
  term cannot tell them apart and appearance memory decides): the variant must pick T.
"""
import numpy as np

from pf.config import load_config
from pf.memory.target import TargetMemory
from pf.world.appearance import AppearanceModel

CFG = load_config()


def track(tid, pos, vel, emb):
    pos = np.asarray(pos, float)
    return {"track_id": tid, "pos_robot": pos - [0.0, 0.0], "pos_world_est": pos, "vel_world_est": np.asarray(vel, float),
            "body_heading_world_est": 0.0, "head_yaw_world_est": 0.0, "appearance": emb, "age_s": 1.0,
            "time_since_seen_s": 0.0, "_gt_id": -1}


def run(variant, seed):
    rng = np.random.default_rng(seed)
    app = AppearanceModel(CFG["appearance"], rng)
    T = app.add_person()
    L = app.add_person(app.lookalike_base(app.base[T]))
    mem = TargetMemory(CFG["memory"], variant)
    dt = 0.1
    v = np.array([0.35, 0.0])
    picks = []
    # t = 0: target straight ahead
    mem.init({"tracks": [track(0, [2.0, 0.0], v, app.observe([T], [np.pi])[0])]})
    x = 2.0
    for k in range(200):
        x += v[0] * dt
        view_T, view_L = rng.uniform(-np.pi, np.pi, 2)          # walking around: varied viewpoints
        tT = track(0, [x, 0.0], v, app.observe([T], [view_T], 0.05)[0])
        if k < 60:                                               # phase 1: both visible, 1.5 m apart
            tL = track(1, [x, 1.5], v, app.observe([L], [view_L], 0.05)[0])
            tracks, phase = [tT, tL], 1
        elif k < 90:                                             # phase 2: T hidden, L where T should be
            tL = track(1, [x, 0.05], v, app.observe([L], [view_L], 0.05)[0])
            tracks, phase = [tL], 2
        else:                                                    # phase 3: symmetric split
            off = min(0.3 + 0.02 * (k - 90), 1.0)
            tT = track(0, [x, +off], v, app.observe([T], [view_T], 0.05)[0])
            tL = track(1, [x, -off], v, app.observe([L], [view_L], 0.05)[0])
            tracks, phase = [tT, tL], 3
        t = mem.step({"tracks": tracks}, dt)
        picks.append((phase, None if t is None else (0 if t is tT else 1)))
    p3 = [p for ph, p in picks if ph == 3 and p is not None]
    return np.mean([p == 0 for p in p3]) if p3 else 0.0, mem


def test_m2_keeps_target_m3_drifts():
    m2 = np.mean([run("M2", s)[0] for s in range(30)])
    m3 = np.mean([run("M3", s)[0] for s in range(30)])
    assert m2 > 0.9, f"M2 picked the target only {m2:.2f} of the time after the crossing"
    assert m3 < m2 - 0.1, f"M3 did not drift (M3 {m3:.2f} vs M2 {m2:.2f})"


def test_gated_memory_grows_with_views_and_is_capped():
    _, mem = run("M2", 0)
    assert 1 < mem.memory.shape[0] <= CFG["memory"]["max_size"]
    _, mem3 = run("M3", 0)
    assert mem3.memory.shape[0] == CFG["memory"]["max_size"]


def test_fixed_memory_never_updates():
    _, mem = run("M1", 0)
    assert mem.memory.shape[0] == 1


def test_reacquire_is_stricter():
    rng = np.random.default_rng(3)
    app = AppearanceModel(CFG["appearance"], rng)
    T = app.add_person()
    L = app.add_person(app.lookalike_base(app.base[T]))
    mem = TargetMemory(CFG["memory"], "M2")
    mem.init({"tracks": [track(0, [2.0, 0.0], [0, 0], app.observe([T], [0.0])[0])]})
    for _ in range(10):                                          # target unseen for 1 s -> lost
        mem.step({"tracks": []}, 0.1)
    assert mem.lost
    accepted = 0
    for k in range(200):
        e = app.observe([L], [rng.uniform(-np.pi, np.pi)], 0.05)[0]
        if mem.step({"tracks": [track(5, [2.0, 0.0], [0, 0], e)]}, 0.1) is not None:
            accepted += 1
            mem.time_since_seen = 10.0                           # stay in the lost state
    assert accepted / 200 < 0.15, "look-alike re-acquired too easily while lost"
