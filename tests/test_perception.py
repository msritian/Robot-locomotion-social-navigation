"""M4 acceptance tests (Section 5.5): FOV/occlusion geometry, noise statistics within 10% of the
configured values over 10k samples, ID-swap rate near the configured value, noise_scale = 0 exact."""
import numpy as np
import pytest

from pf.config import load_config
from pf.perception import GroundTruth, Perception, PerceptionSensor, visibility
from pf.world.appearance import AppearanceModel
from pf.world.grid import Grid, rasterize
from pf.world.robot import wrap

CFG = load_config()
FOV = np.deg2rad(CFG["robot"]["fov_deg"])

# 20 x 20 m room with one 1 x 4 m wall segment at x in [10, 11], y in [8, 12]
GRID = Grid(rasterize(20, 20, 0.05, rects_free=[(0.5, 0.5, 19.5, 19.5)], rects_occ=[(10, 8, 11, 12)]), 0.05)


def los(origin, pts):
    return GRID.segment_free(np.broadcast_to(origin, pts.shape), pts, 0.0)


def vis(cam, pos):
    return visibility(np.asarray(cam, float), np.asarray(pos, float), 0.25, FOV, 0.3, 8.0, los)


def make_gt(pos, heading=None, head=None, cam=(5.0, 10.0, 0.0), app=None):
    pos = np.asarray(pos, float).reshape(-1, 2)
    n = len(pos)
    app = app or AppearanceModel(CFG["appearance"], np.random.default_rng(0))
    while len(app.base) < n:
        app.add_person()
    cam = np.asarray(cam, float)
    return GroundTruth(
        cam_pose=cam, robot_pose=cam.copy(), odom=cam.copy(), pos=pos,
        heading=np.zeros(n) if heading is None else np.asarray(heading, float),
        head=np.zeros(n) if head is None else np.asarray(head, float), person_radius=0.25,
        los_fn=los, ray_fn=GRID.raycast, app_fn=lambda i, v, e, s: app.observe(i, v, e, s))


# ---------------------------------------------------------------------------- geometry
def test_fov_and_range():
    # one person at a time (so nobody occludes anybody)
    pts = [[7, 10], [5, 12.5], [4, 10], [5.1, 10], [13.5, 10.5], [7, 11.9]]
    v = [bool(vis([5, 10, 0], [p])[0][0]) for p in pts]
    # ahead: yes; 90 deg to the side: no; behind: no; too close (<0.3 m): no; >8 m: no; 43.5 deg: yes
    assert v == [True, False, False, False, False, True]


def test_fov_70_deg():
    v, *_ = visibility(np.array([5, 10, 0.0]), np.array([[7, 11.7]]), 0.25, np.deg2rad(70), 0.3, 8.0, los)
    assert not v[0]   # 40.4 deg off-axis: inside 90 deg FOV, outside 70 deg


def test_wall_occlusion():
    # (12, 10) is straight behind the wall; the LOS to (12, 14.5) passes above it (y = 12.25 at x = 10)
    v, *_ = vis([8, 10, 0.3], [[12, 10], [12, 14.5]])
    assert v.tolist() == [False, True]


def test_person_occlusion_and_partial():
    # occluder exactly on the line of sight -> fully blocked
    v, p, *_ = vis([2, 10, 0], [[6, 10], [4, 10]])
    assert v.tolist() == [False, True]
    # occluder 0.3 m off the LOS (between 0.25 and 0.35) -> partial
    v, p, *_ = vis([2, 10, 0], [[6, 10], [4, 10.3]])
    assert v.tolist() == [True, True] and p.tolist() == [True, False]
    # occluder far off the LOS -> clear
    v, p, *_ = vis([2, 10, 0], [[6, 10], [4, 11.0]])
    assert v.tolist() == [True, True] and p.tolist() == [False, False]
    # an occluder BEHIND the person does not block it
    v, *_ = vis([2, 10, 0], [[4, 10], [6, 10]])
    assert v[0]


# ---------------------------------------------------------------------------- noise statistics
@pytest.mark.parametrize("dist", [2.0, 5.0])
def test_position_heading_head_noise_stats(dist):
    pc = CFG["perception"]
    sensor = PerceptionSensor(dict(pc, fp_rate=0.0), CFG["robot"], np.random.default_rng(1))
    gt = make_gt([[2 + dist, 4.0]], heading=[0.7], head=[-0.4], cam=(2.0, 4.0, 0.0))  # clear of the wall
    errs, bh, hd, nan = [], [], [], 0
    n = 0
    while n < 10_000:
        det, _ = sensor.detect(gt)
        k = det.gt_id >= 0
        if not k.any():
            continue
        n += 1
        errs.append(det.pos_robot[k][0] - [dist, 0.0])
        bh.append(wrap(det.body_heading_rel[k][0] - 0.7))
        h = det.head_rel[k][0]
        if np.isnan(h):
            nan += 1
        else:
            hd.append(wrap(h + 0.4))
    errs = np.array(errs)
    target = pc["pos_sigma0"] + pc["pos_sigma_k"] * dist
    assert np.allclose(errs.std(0), target, rtol=0.10)
    assert np.abs(errs.mean(0)).max() < 0.1 * target
    assert np.std(bh) == pytest.approx(np.deg2rad(pc["body_heading_sigma_deg"]), rel=0.10)
    assert np.std(hd) == pytest.approx(np.deg2rad(pc["head_yaw_sigma_deg"]), rel=0.10)
    assert nan / n == pytest.approx(pc["head_nan_p"], rel=0.15)   # 5% of 10k: binomial sd ~4.4% rel


@pytest.mark.parametrize("dist,p_expected", [(2.0, 0.95), (5.5, 0.825), (8.0, 0.70)])
def test_detection_probability(dist, p_expected):
    sensor = PerceptionSensor(dict(CFG["perception"], fp_rate=0.0), CFG["robot"], np.random.default_rng(2))
    gt = make_gt([[2 + dist - 1e-6, 4.0]], cam=(2.0, 4.0, 0.0))
    hits = sum(len(sensor.detect(gt)[0].gt_id) for _ in range(10_000))
    assert hits / 10_000 == pytest.approx(p_expected, abs=0.012)


def test_false_positive_rate():
    sensor = PerceptionSensor(CFG["perception"], CFG["robot"], np.random.default_rng(3))
    gt = make_gt(np.zeros((0, 2)))
    fps = sum(len(sensor.detect(gt)[0].gt_id) for _ in range(20_000))
    assert fps / 20_000 == pytest.approx(CFG["perception"]["fp_rate"], rel=0.10)


def test_depth_ray_noise():
    pc = CFG["perception"]
    sensor = PerceptionSensor(pc, CFG["robot"], np.random.default_rng(4))
    gt = make_gt(np.zeros((0, 2)), cam=(5.0, 10.0, np.pi))   # facing the x = 0.5 wall, 4.5 m away
    true = GRID.raycast(np.array([5.0, 10.0]), np.pi + sensor.ray_angles, 8.0)
    d = np.array([sensor.depth_rays(gt) for _ in range(400)])
    mid = len(true) // 2
    sd = d[:, mid].std()
    assert sd == pytest.approx(pc["depth_sigma0"] + pc["depth_sigma_k"] * true[mid], rel=0.15)


# ---------------------------------------------------------------------------- tracker
def test_id_swap_rate_when_close():
    """Two people 0.4 m apart, both visible: injected swaps happen at ~swap_p per step."""
    cfg = CFG
    per = Perception(cfg, seed=5)
    per.sensor.pc = dict(per.sensor.pc, fp_rate=0.0)
    gt = make_gt([[8.0, 10.2], [8.0, 9.8]])
    steps = 0
    for _ in range(3000):
        per.step_gt(gt)
        steps += 1
    rate = per.tracker.n_swaps / steps
    assert rate == pytest.approx(cfg["tracker"]["swap_p"], abs=0.04)


def test_no_swaps_when_far_apart():
    per = Perception(CFG, seed=6)
    gt = make_gt([[8.0, 11.5], [8.0, 8.5]])
    for _ in range(500):
        per.step_gt(gt)
    assert per.tracker.n_swaps == 0


def test_track_lifecycle():
    per = Perception(CFG, seed=7, noise_scale=0.0)
    gt = make_gt([[8.0, 10.0]])
    assert per.step_gt(gt)["tracks"] == []            # 1 detection: tentative only
    tr = per.step_gt(gt)["tracks"]
    assert len(tr) == 1                                # confirmed after 2 consecutive detections
    gone = make_gt([[8.0, 10.0]], cam=(5.0, 10.0, np.pi))  # turn away: person no longer visible
    alive = [len(per.step_gt(gone)["tracks"]) for _ in range(12)]
    assert alive[:10] == [1] * 10 and alive[-1] == 0  # deleted after 1.0 s unseen


def test_noise_scale_zero_exact():
    per = Perception(CFG, seed=8, noise_scale=0.0)
    gt = make_gt([[8.0, 10.5], [10.0, 7.0]], heading=[0.3, -2.0], head=[0.5, -1.0])
    for _ in range(5):
        obs = per.step_gt(gt)
    det = per.last_det
    assert np.allclose(np.sort(det.pos_robot[:, 0]), [3.0, 5.0])
    order = np.argsort(det.gt_id)
    assert np.allclose(det.pos_robot[order], [[3.0, 0.5], [5.0, -3.0]], atol=1e-12)
    assert np.allclose(det.body_heading_rel[order], [0.3, -2.0], atol=1e-12)
    assert np.allclose(det.head_rel[order], [0.5, -1.0], atol=1e-12)
    for t in obs["tracks"]:
        true = gt.pos[t["_gt_id"]]
        assert np.allclose(t["pos_world_est"], true, atol=1e-3)
        assert np.allclose(t["vel_world_est"], 0.0, atol=1e-3)
    assert np.allclose(obs["depth_rays"], GRID.raycast(gt.cam_pose[:2], gt.cam_pose[2] + per.sensor.ray_angles, 8.0))
    assert per.tracker.n_swaps == 0


def test_still_person_stable_while_robot_moves():
    """Early sanity check (Section 14): a still person is reported at a stable world position while
    the robot moves (world estimate uses the robot's odometry; with perfect odom it should not drift)."""
    per = Perception(CFG, seed=9)
    app = AppearanceModel(CFG["appearance"], np.random.default_rng(0))
    est = []
    for k in range(100):
        cam = np.array([2.0 + 0.04 * k, 10.0, -0.6 + 0.05 * np.sin(0.2 * k)])
        obs = per.step_gt(make_gt([[8.5, 6.0]], cam=cam, app=app))
        for t in obs["tracks"]:
            if t["_gt_id"] == 0 and t["time_since_seen_s"] == 0:
                est.append(t["pos_world_est"])
    est = np.array(est)
    assert len(est) > 50
    assert np.linalg.norm(est[20:].mean(0) - [8.5, 6.0]) < 0.1
    assert est[20:].std(0).max() < 0.15
