"""Our following brain (tracker + gated target memory + predictor + DWA + search) as an EVT-Bench agent.

EVT-Bench robot: a Spot base on the navmesh, driven by [vx, vy, wz] in [-1, 1] (scaled by the benchmark's
speeds), one RGB "jaw" camera (384x384, 90 deg) plus the panoptic image used by the benchmark's own sensors.

Perception (what the brain sees each step):
  - people: one detection per human instance in the panoptic image (instance masks, like a person
    segmenter). The instance ids are used ONLY to split pixels into people and for metrics; the brain gets
    no identity. Position = median depth over the mask + mask column -> bearing.
  - appearance: Hellinger-normalised HSV histograms of the upper and lower half of the mask (no learned model).
  - obstacles: 32 depth rays across the FOV from an added jaw depth camera (people and floor removed).
  - target: given once, by its mask on the first frames (first-frame-box protocol of single-object
    tracking). The VLAs get a language description instead.
The episode loop below is OmTrackVLA's trained_agent.evaluate_agent loop with only the agent swapped,
so termination rules and metrics are identical.
"""
from __future__ import annotations

import json
import os
import os.path as osp

import imageio
import magnum as mn
import numpy as np
from habitat_sim.gfx import LightInfo, LightPositionModel
from tqdm import trange

import habitat
from pf.config import deep_merge, load_config
from pf.control.brain import FollowerBrain
from pf.perception.sensor import Detections
from pf.perception.tracker import Tracker
from pf.prediction.base import ConstantVelocity

from evt.evt_cfg import ANG, CTRL_FREQ, EVT_OVERRIDES, HUMAN_IDS, LAT, LIN, STEP_S, VARIANTS, VX_MAX, VY_MAX, WZ_MAX  # noqa: F401
RGB, DEPTH, PAN = ("agent_1_articulated_agent_jaw_rgb", "agent_1_articulated_agent_jaw_depth",
                   "agent_1_articulated_agent_jaw_panoptic")

LIGHTS = [LightInfo(vector=v, color=[1.0, 1.0, 1.0], model=LightPositionModel.Global)
          for v in ([10.0, -2.0, 0.0, 0.0], [-10.0, -2.0, 0.0, 0.0], [0.0, -2.0, 10.0, 0.0], [0.0, -2.0, -10.0, 0.0])]


def depth_sensor_opts(h=384, w=384, hfov=90):
    k = "habitat.simulator.agents.agent_1.sim_sensors.jaw_depth_sensor"
    return [f"+habitat/simulator/sim_sensors@{k}=jaw_depth_sensor", f"{k}.height={h}", f"{k}.width={w}", f"{k}.hfov={hfov}"]


def align_depth_camera(sim):
    """Spot's stock jaw depth camera sits on the arm (link 6, rotated); the jaw RGB / panoptic cameras sit on the base.
    Mount depth exactly like panoptic so masks and depth line up pixel for pixel."""
    robot = sim.agents_mgr[1].articulated_agent
    cams = robot.params.cameras
    depth = [k for k in cams if "jaw_depth" in k]
    pan = [k for k in cams if "jaw_panoptic" in k] or [k for k in cams if "jaw_rgb" in k]
    if not getattr(align_depth_camera, "said", False):
        print(f"[align] {type(robot).__name__} cameras {list(cams)} -> depth {depth} := {pan[:1]}", flush=True)
        align_depth_camera.said = True
    assert depth and pan, list(cams)
    for k in depth:
        cams[k] = cams[pan[0]]
    robot.update()


# ---------------------------------------------------------------------------------------------- perception
SELF_DEPTH = 0.35     # m: closer depth is the robot's own body (jaw camera sees it at ~0.18 m in the lower middle)

def _hsv(rgb):
    x = rgb.astype(np.float32) / 255.0
    mx, mn_ = x.max(-1), x.min(-1)
    d = mx - mn_ + 1e-6
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
    s = np.where(mx > 0, (mx - mn_) / (mx + 1e-6), 0)
    return np.stack([h % 1.0, s, mx], -1)


def appearance(rgb, mask):
    """256-D: HSV histograms (8x4x4) of the upper and lower half of the mask, sqrt + L2 (cosine = Bhattacharyya)."""
    rows = np.where(mask.any(1))[0]
    mid = (rows[0] + rows[-1]) // 2
    out = []
    for part in (slice(0, mid + 1), slice(mid + 1, None)):
        m = np.zeros_like(mask)
        m[part] = mask[part]
        px = _hsv(rgb[m])
        if len(px) == 0:
            out.append(np.zeros(128))
            continue
        idx = (np.minimum(px[:, 0] * 8, 7).astype(int) * 16 + np.minimum(px[:, 1] * 4, 3).astype(int) * 4
               + np.minimum(px[:, 2] * 4, 3).astype(int))
        out.append(np.bincount(idx, minlength=128) / len(px))
    v = np.sqrt(np.concatenate(out))
    return v / (np.linalg.norm(v) + 1e-9)


class EVTPerception:
    def __init__(self, cfg, h, w, hfov_deg=90.0, min_px=150, max_range=12.0, n_rays=32):
        self.f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
        self.cx, self.cy = (w - 1) / 2, (h - 1) / 2
        self.min_px, self.max_range = min_px, max_range
        self.ray_angles = np.linspace(-np.radians(hfov_deg) / 2, np.radians(hfov_deg) / 2, n_rays)
        self.ray_cols = np.clip(np.round(self.cx - self.f * np.tan(self.ray_angles)).astype(int), 0, w - 1)
        self.cfg = cfg

    def people(self, rgb, depth, pan, main_id, only_id=None):
        ids = np.unique(pan)
        ids = [int(i) for i in ids if i == main_id or HUMAN_IDS[0] <= i <= HUMAN_IDS[1]]
        if only_id is not None:
            ids = [i for i in ids if i == only_id]
        pos, app, gid, sig, box = [], [], [], [], []
        person = (pan == main_id) | ((pan >= HUMAN_IDS[0]) & (pan <= HUMAN_IDS[1]))
        for i in ids:
            m = visible_part(pan == i, depth, person, self.min_px)
            if m is None:
                continue
            z = float(np.median(depth[m]))
            if z > self.max_range:
                continue
            u = float(np.median(np.where(m)[1]))
            lat = z * (self.cx - u) / self.f
            r = np.hypot(z, lat)
            z, lat = z * (r + 0.15) / r, lat * (r + 0.15) / r        # depth hits the body surface; centre is ~0.15 m behind
            pos.append([z, lat]); app.append(appearance(rgb, m)); gid.append(i)
            rr, cc = np.where(m)
            box.append((int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1))
            sig.append(0.05 + 0.03 * r)
        n = len(pos)
        self.last_boxes = box                 # pixel box of each detection's visible mask (hybrid: text grounding)
        return Detections(pos_robot=np.array(pos).reshape(n, 2), body_heading_rel=np.full(n, np.nan),
                          head_rel=np.full(n, np.nan), appearance=np.array(app).reshape(n, 256),
                          gt_id=np.array(gid, dtype=int), meas_sigma=np.array(sig))

    def rays(self, depth, pan, cam_h_above_floor, main_id):
        """Per ray column: nearest non-person point between 0.15 and 1.8 m above the floor (horizontal range)."""
        h, _ = depth.shape
        v = np.arange(h)[:, None]
        out = np.full(len(self.ray_cols), self.max_range)
        for k, c in enumerate(self.ray_cols):
            cols = slice(max(c - 2, 0), c + 3)
            d = depth[:, cols]
            p = pan[:, cols]
            person = (p == main_id) | ((p >= HUMAN_IDS[0]) & (p <= HUMAN_IDS[1]))
            height = cam_h_above_floor + (self.cy - v) / self.f * d
            ok = (d > SELF_DEPTH) & ~person & (height > 0.15) & (height < 1.8)
            if ok.any():
                out[k] = min(self.max_range, float(d[ok].min()) / np.cos(self.ray_angles[k]))
        return out


def visible_part(m, depth, person, min_px, ring=4):
    """The panoptic render ignores the scene mesh, so a person behind a wall still gets a mask (painted over the wall)
    and the depth there is the wall's. Keep only the farthest depth cluster of the mask (hidden pixels read the closer
    occluder), and accept it only if, on both its left and right, the depth just outside is clearly farther (the person
    stands in front of the background) or clearly closer (an occluder edge). Same depth on a side = painted on a surface.
    Returns the visible pixel mask or None."""
    valid = m & (depth > SELF_DEPTH) & np.isfinite(depth)
    if valid.sum() < min_px:
        return None
    dv = depth[valid]
    vis = valid & (depth > np.percentile(dv, 90) - 0.3)
    if vis.sum() < min_px:
        return None
    z = float(np.median(depth[vis]))
    rows, cols = np.nonzero(vis)
    h, w = depth.shape
    for side in (-1, 1):
        ds = []
        for r in np.unique(rows)[::2]:
            c = cols[rows == r]
            edge = c.min() if side < 0 else c.max()
            cc = edge + side * np.arange(1, ring + 1)
            cc = cc[(cc >= 0) & (cc < w)]
            ok = ~person[r, cc] & (depth[r, cc] > SELF_DEPTH)
            ds.extend(depth[r, cc[ok]].tolist())
        if len(ds) < 10:                    # image border or other people on this side: no evidence either way
            continue
        dm = float(np.median(ds))
        if not (dm > z + 0.1 or dm < z - 0.3):
            return None
    return vis


# ---------------------------------------------------------------------------------------------- agent
def pose2d(m):
    """Habitat (y up) transform -> planar (x, y, yaw) with x = X, y = -Z; the camera looks along local -Z."""
    t = m.translation
    f = m.transform_vector(mn.Vector3(0, 0, -1))
    return np.array([t[0], -t[2], np.arctan2(-f[2], f[0])]), float(t[1])


class OursAgent:
    def __init__(self, result_path, controller="C1", memory="M2", search="S1", save_video_n=3, cfg_over=None):
        self.result_path = result_path
        os.makedirs(result_path, exist_ok=True)
        self.cfg = load_config(overrides=deep_merge(EVT_OVERRIDES, cfg_over or {}))
        self.kw = dict(controller=controller, memory=memory, search=search)
        self.save_video_n, self.n_eps = save_video_n, 0
        self.per = None
        self.rgb_list, self.log = [], []

    def start(self, episode, main_id=None):
        self.brain = FollowerBrain(self.cfg, self.kw["controller"], predictor=ConstantVelocity(),
                                   memory=self.kw["memory"], search=self.kw["search"])
        self.tracker = Tracker(self.cfg["tracker"], STEP_S, 1.0, np.random.default_rng(0))
        # the id the benchmark's own detector sensor scores with (it keeps the first episode's id; see run log)
        self.main_id = int(episode.info["main_human_semantic_id"] if main_id is None else main_id)
        self.rgb_list, self.log, self.raw = [], [], []
        self.ep_tag, self.n_dump = f"{episode.scene_id.split('/')[-2]}_{episode.episode_id}", 0

    def act(self, sim, obs):
        rgb, depth, pan = obs[RGB][..., :3], obs[DEPTH], obs[PAN]
        depth = depth[..., 0] if depth.ndim == 3 else depth
        pan = pan[..., 0] if pan.ndim == 3 else pan
        if self.per is None:
            self.per = EVTPerception(self.cfg, *depth.shape)
        cam = sim._sensors[DEPTH]._sensor_object.node.absolute_transformation()
        odom, cam_y = pose2d(cam)
        robot = sim.agents_mgr[1].articulated_agent
        floor_y = float(sim.pathfinder.snap_point(robot.base_pos)[1])
        init = not self.brain.memory.initialized
        # EVT_ORACLE_ID=1 (ablation only): the target's identity is given, other people are never detected
        oracle = os.environ.get("EVT_ORACLE_ID") == "1"
        det = self.per.people(rgb, depth, pan, self.main_id, only_id=self.main_id if init or oracle else None)
        if os.environ.get("EVT_DEBUG") and len(self.log) < 3:
            u, c = np.unique(pan, return_counts=True)
            print(f"[debug] main={self.main_id} pan ids={dict(zip(u.tolist()[-6:], c.tolist()[-6:]))} depth {depth.min():.2f}-{depth.max():.2f} "
                  f"dtype {pan.dtype} n_det={len(det.gt_id)} init={init}", flush=True)
        tracks = self.tracker.update(det, odom)
        self.raw.append({"odom": odom.copy(), "pos": det.pos_robot.copy(), "gid": det.gt_id.copy(), "app": det.appearance.astype(np.float16),
                         "init": init, "rays": None})
        bobs = {"tracks": tracks, "robot_odom": odom.copy(), "ray_angles": self.per.ray_angles,
                "depth_rays": self.per.rays(depth, pan, cam_y - floor_y, self.main_id)}
        self.raw[-1]["rays"] = bobs["depth_rays"].astype(np.float32)
        cmd = self.brain.step(bobs)
        if cmd is None or not np.all(np.isfinite(cmd)):
            cmd = np.zeros(3)
        sel = self.brain.selected
        hp = sim.agents_mgr[0].articulated_agent.base_pos       # METRICS ONLY: true target, camera frame
        d = np.array([hp[0], -hp[2]]) - odom[:2]
        c, s_ = np.cos(odom[2]), np.sin(odom[2])
        rel = [round(float(c * d[0] + s_ * d[1]), 2), round(float(-s_ * d[0] + c * d[1]), 2)]
        mem = self.brain.memory
        sa_t = sa_o = None                                       # METRICS ONLY: appearance score of true target / best other
        if mem.initialized and tracks:
            sa = mem.s_app(np.stack([t["appearance"] for t in tracks]))
            g = np.array([t["_gt_id"] for t in tracks])
            sa_t = round(float(sa[g == self.main_id].max()), 3) if (g == self.main_id).any() else None
            sa_o = round(float(sa[g != self.main_id].max()), 3) if (g != self.main_id).any() else None
        if os.environ.get("EVT_DUMP") and self.main_id in det.gt_id and self.n_dump < 6:   # DEBUG: frames where range is off
            q = det.pos_robot[list(det.gt_id).index(self.main_id)]
            if abs(np.hypot(*q) - np.hypot(*rel)) > 0.8:
                self.n_dump += 1
                np.savez_compressed(os.path.join(self.result_path, f"dump_{self.ep_tag}_{len(self.log)}.npz"),
                                    rgb=rgb, depth=depth, pan=pan, det=q, true=np.array(rel), main=self.main_id)
        self.log.append({"sa_target": sa_t, "sa_other": sa_o, "scores": {k: round(v, 3) for k, v in mem.last_scores.items()},
                         "tgt_rel": rel, "state": self.brain.state, "sel_gt": -2 if sel is None else int(sel["_gt_id"]),
                         "n_det": int(len(det.gt_id)), "cmd": [round(float(c), 3) for c in cmd]})
        if self.n_eps < self.save_video_n:
            self.rgb_list.append(rgb.copy())
        return [float(np.clip(cmd[0] / VX_MAX, -1, 1)), float(np.clip(cmd[1] / VY_MAX, -1, 1)),
                float(np.clip(cmd[2] / WZ_MAX, -1, 1))]

    def finish(self, episode, save_dir):
        if self.rgb_list:
            imageio.mimsave(os.path.join(save_dir, f"{episode.episode_id}.mp4"), self.rgb_list, fps=8)
        np.save(os.path.join(save_dir, f"{episode.episode_id}_raw.npy"), np.array(self.raw, dtype=object), allow_pickle=True)
        sel = [r["sel_gt"] for r in self.log if r["sel_gt"] != -2]
        wrong = sum(1 for a, b in zip(sel[:-1], sel[1:]) if a != b and b != self.main_id)
        self.n_eps += 1
        return {"brain_wrong_switches": wrong, "brain_steps_selected_target": sum(s == self.main_id for s in sel),
                "brain_steps_selected_other": sum(s not in (self.main_id, -1) for s in sel)}


def evaluate_agent(config, dataset_split, save_path, agent: OursAgent) -> None:
    """OmTrackVLA trained_agent.evaluate_agent, with the agent swapped and brain diagnostics added to the json."""
    with habitat.TrackEnv(config=config, dataset=dataset_split) as env:
        sim = env.sim
        for _ in trange(len(env.episodes)):
            env.reset()
            sim.set_light_setup(LIGHTS)
            align_depth_camera(sim)
            result, record_infos = {}, []
            try:
                instruction = env.current_episode.info.get('instruction', None)
            except Exception:
                instruction = None
            env.task._get_observations(env.current_episode)
            sensor = env.task.sensor_suite.sensors.get("agent_1_main_humanoid_detector_sensor")
            agent.start(env.current_episode, getattr(sensor, "_human_id", None))
            finished = False
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent
            iter_step = followed_step = too_far_count = 0
            status = 'Normal'
            info = env.get_metrics()
            while not env.episode_over:
                record_info = {}
                obs = sim.get_sensor_observations()
                env.task._get_observations(env.current_episode)
                action = agent.act(sim, obs)
                action_dict = {
                    "action": ("agent_0_humanoid_navigate_action", "agent_1_base_velocity",
                               "agent_2_oracle_nav_randcoord_action_obstacle", "agent_3_oracle_nav_randcoord_action_obstacle",
                               "agent_4_oracle_nav_randcoord_action_obstacle", "agent_5_oracle_nav_randcoord_action_obstacle"),
                    "action_args": {"agent_1_base_vel": action},
                }
                iter_step += 1
                env.step(action_dict)
                info = env.get_metrics()
                if info['human_following'] == 1.0:
                    followed_step += 1
                    too_far_count = 0
                if np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos) > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        status = 'Lost'
                        finished = False
                        break
                record_info["step"] = iter_step
                record_info["dis_to_human"] = float(np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos))
                record_info["facing"] = info['human_following']
                record_info["base_velocity"] = action
                record_info.update(agent.log[-1])
                record_infos.append(record_info)
                if info['human_collision'] == 1.0:
                    status = 'Collision'
                    finished = False
                    break
            info = env.get_metrics()
            if env.episode_over:
                finished = True
            scene_key = osp.splitext(osp.basename(env.current_episode.scene_id))[0].split('.')[0]
            save_dir = os.path.join(save_path, scene_key)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(record_infos, f, indent=2)
            result['finish'] = finished
            result['status'] = status
            if iter_step < 300:
                result['success'] = info['human_following_success'] and info['human_following']
            else:
                result['success'] = info['human_following']
            result['following_rate'] = followed_step / iter_step
            result['following_step'] = followed_step
            result['total_step'] = iter_step
            result['collision'] = info['human_collision']
            if instruction is not None:
                result['instruction'] = instruction
            result.update(agent.finish(env.current_episode, save_dir))
            with open(os.path.join(save_dir, "{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result, f, indent=2, default=float)
            print(f"[ours] ep {env.current_episode.episode_id} {status} success={result['success']} "
                  f"TR={result['following_rate']:.2f} steps={iter_step}", flush=True)
