"""Part D closed loop: the trained K1 walker + the Stage C following brain in Isaac Sim (PROJECT_SPEC 13).

    every 0.1 s : perception (D1: noisy GT through the Stage C perception simulator, camera = K1 head pose)
                  -> target memory -> predictor -> controller (+DWA) -> (vx, vy, wz), clamped + rate limited
    every 0.02 s: K1 walker policy (TorchScript) on the Isaac Lab observation -> leg joint targets
    every 0.005 s: physics

People move with the Stage C people simulator (same scenario definitions); in Isaac they are kinematic mannequins.
Writes metrics.json (Section 10 metrics, same code as Stage C) and, with --video, three MP4 views + a composite.

    python isaac_follow/run_follow.py --scenario T6 --seed 1000 --method full --policy k1_walker.pt --video --headless
"""
from __future__ import annotations

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="T7")
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--method", default="full", choices=["full", "C0"])
ap.add_argument("--policy", required=True, help="exported TorchScript walker (policy.pt from play.py)")
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--crowd", type=int, default=0, help="extra walking people (showcase crowds of 8-15)")
ap.add_argument("--video", action="store_true")
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--out", default="follow_out")
ap.add_argument("--limits", default="", help="walker_response.yaml with measured command limits")
ap.add_argument("--interiorgs", default="", help="SAGE-3D scene id (e.g. 839962): realistic scene instead of boxes")
ap.add_argument("--scene_dir", default="interiorgs", help="dir with <id>.usdz and <id>_collision.usd")
ap.add_argument("--debug", action="store_true", help="print obs/actions/trunk height for the first policy steps")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

import k1_walker  # noqa: E402,F401
from k1_walker.k1_cfg import HEAD_CAMERA_OFFSET, HEAD_LINK  # noqa: E402
from pf.config import load_config  # noqa: E402
from pf.control.brain import FollowerBrain  # noqa: E402
from pf.eval.metrics import episode_metrics  # noqa: E402
from pf.eval.scenarios import build  # noqa: E402
from pf.perception import Perception  # noqa: E402
from pf.perception.sensor import GroundTruth  # noqa: E402
from pf.world.robot import wrap  # noqa: E402

from isaac_follow.scene_cfg import LOOKALIKE_SHIRTS, OTHER_SHIRTS, TARGET_SHIRT, make_env_cfg  # noqa: E402

os.makedirs(args.out, exist_ok=True)
cfg = load_config()
DT_BRAIN, DT_POLICY = cfg["sim"]["dt"], 0.02
SUB = int(round(DT_BRAIN / DT_POLICY))

# ------------------------------------------------------------------ 2D world (map, people, scenario)
interior_usda = None
if args.interiorgs:
    from isaac_follow.interiorgs_scene import write_scene_usda
    from pf.eval.scenarios import build_on_map
    from pf.world.interiorgs import scene_to_map
    coll = os.path.join(args.scene_dir, f"{args.interiorgs}_collision.usd")
    imap = scene_to_map(coll, args.interiorgs, cfg)
    interior_usda = write_scene_usda(os.path.join(args.out, f"scene_{args.interiorgs}.usda"),
                                     os.path.join(args.scene_dir, f"{args.interiorgs}.usdz"), coll, imap.params["frame"])
    w = build_on_map(cfg, imap, args.scenario, args.seed)
else:
    w = build(cfg, args.scenario, args.seed)
if args.crowd:
    pts = w.walkable_points(0.45)
    rng = np.random.default_rng([args.seed, 4242])
    for _ in range(args.crowd):
        for _t in range(500):
            p = pts[rng.integers(len(pts))]
            if np.hypot(*(p - w.robot.pose[:2])) > 2.5 and all(np.hypot(*(p - q)) > 1.0 for q in w.people.pos):
                break
        w.add_person(p, heading=rng.uniform(-np.pi, np.pi), start_pause=float(rng.uniform(0, 2)))
ti = w.target_index
look = set(getattr(w, "lookalike_ids", []))
colors = []
k_other = 0
for i in range(w.people.n):
    if i == ti:
        colors.append(TARGET_SHIRT)
    elif i in look:
        colors.append(LOOKALIKE_SHIRTS[len(colors) % 2])
    else:
        colors.append(OTHER_SHIRTS[k_other % len(OTHER_SHIRTS)])
        k_other += 1

# walker command limits: measured (B.6) if available, else the Stage C robot ranges
lim = {"vx": cfg["robot"]["vx_range"], "vy": cfg["robot"]["vy_range"], "wz": cfg["robot"]["wz_range"]}
if args.limits and os.path.exists(args.limits):
    lim.update(yaml.safe_load(open(args.limits)).get("limits", {}))
LO = np.array([lim["vx"][0], lim["vy"][0], lim["wz"][0]])
HI = np.array([lim["vx"][1], lim["vy"][1], lim["wz"][1]])
RATE = np.array([cfg["robot"]["acc_lin"], cfg["robot"]["acc_lin"], cfg["robot"]["acc_ang"]]) * DT_BRAIN * 2.0

# ------------------------------------------------------------------ Isaac env
env_cfg = make_env_cfg(w.map, w.people.pos, colors, cfg["robot"]["fov_deg"], video=args.video,
                       interior_usda=interior_usda)
x0, y0, yaw0 = w.robot.pose
env_cfg.scene.robot.init_state.pos = (float(x0), float(y0), 0.57)
env_cfg.scene.robot.init_state.rot = (float(np.cos(yaw0 / 2)), 0.0, 0.0, float(np.sin(yaw0 / 2)))
# mannequin details (head + shirt) as extra kinematic visual objects
for i in range(w.people.n):
    for part, shape, z in (("head", sim_utils.SphereCfg(radius=0.12), 1.55),
                           ("shirt", sim_utils.CylinderCfg(radius=0.215, height=0.45, axis="Z"), 0.95)):
        color = (0.85, 0.70, 0.58) if part == "head" else colors[i]
        shape.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        shape.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        shape.mass_props = sim_utils.MassPropertiesCfg(mass=1.0)
        shape.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6)
        setattr(env_cfg.scene, f"person_{i:02d}_{part}", RigidObjectCfg(
            prim_path=f"/World/Scene/Person_{i:02d}_{part}", spawn=shape,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(w.people.pos[i, 0], w.people.pos[i, 1], z))))
print(f"[run_follow] building Isaac env: {w.people.n} people, video={args.video}", flush=True)
env = ManagerBasedRLEnv(cfg=env_cfg)
print("[run_follow] env ready", flush=True)
if interior_usda is not None:
    # Gaussian-splat (NuRec) rendering needs isaacsim.replicator.nurec_utils, which the Isaac Lab container does not
    # ship -> Section 19.6 fallback: render the InteriorGS collision meshes (real walls, floor, furniture) with
    # simple materials instead of the splat.
    from isaac_follow.interiorgs_scene import style_collision_meshes
    import omni.usd
    n_vis = style_collision_meshes(omni.usd.get_context().get_stage(), "/World/Interior/scene_collision")
    print(f"[run_follow] InteriorGS: splat not renderable in this container; showing {n_vis} collision meshes", flush=True)
if args.policy == "zero":      # plumbing tests only: legs hold the default pose (not a walker)
    n_act = env.action_manager.total_action_dim
    def policy(o):
        return torch.zeros(o.shape[0], n_act, device=env.device)
else:
    policy = torch.jit.load(args.policy, map_location=env.device).eval()
robot = env.scene["robot"]
head_id = robot.find_bodies(HEAD_LINK)[0][0]
cam_off = torch.tensor([HEAD_CAMERA_OFFSET], device=env.device)
cmd_term = env.command_manager.get_term("base_velocity")
people_objs = [(env.scene[f"person_{i:02d}"], env.scene[f"person_{i:02d}_head"], env.scene[f"person_{i:02d}_shirt"])
               for i in range(w.people.n)]


def set_people(pos, heading):
    for i, (body, head, shirt) in enumerate(people_objs):
        q = torch.tensor([[np.cos(heading[i] / 2), 0, 0, np.sin(heading[i] / 2)]], device=env.device, dtype=torch.float32)
        for obj, z in ((body, 0.7), (head, 1.55), (shirt, 0.95)):
            pose = torch.cat([torch.tensor([[pos[i, 0], pos[i, 1], z]], device=env.device, dtype=torch.float32), q], 1)
            obj.write_root_pose_to_sim(pose)


def robot_pose2d():
    p = robot.data.root_pos_w[0].cpu().numpy()
    q = robot.data.root_quat_w[0].cpu().numpy()
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([p[0], p[1], yaw]), p[2]


def head_cam_pose2d():
    hp, hq = robot.data.body_pos_w[:, head_id], robot.data.body_quat_w[:, head_id]
    c = (hp + quat_apply(hq, cam_off))[0].cpu().numpy()
    q = hq[0].cpu().numpy()
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([c[0], c[1], yaw]), c[2]


grid = w.grid


def los_fn(origin, pts):   # walls are vertical extrusions of the 2D map: 2D LOS == 3D LOS through walls
    return grid.segment_free(np.broadcast_to(origin, pts.shape), pts, 0.0)


def app_fn(idx, view, extra_sigma, noise_scale):
    return w.app.observe(np.asarray([w.app_idx[i] for i in idx], dtype=int), view, extra_sigma, noise_scale)


per = Perception(cfg, args.seed)
if args.method == "full":
    from pf.eval.experiments import get_predictor
    brain = FollowerBrain(cfg, "C1", predictor=get_predictor("P2"), memory="M2", search="S1")
else:
    from pf.prediction.base import ConstantVelocity
    brain = FollowerBrain(cfg, "C0", predictor=ConstantVelocity(), memory="M2", search="S1")

obs, _ = env.reset()
set_people(w.people.pos, w.people.heading)
pose, _ = robot_pose2d()
w.robot.reset(pose)
# ------------------------------------------------------------------ video helpers
if args.video:
    from isaac_follow.video import HeadOverlay, TopView, chase_eye_target
    chase_cam = env.scene["chase_cam"]
    head_cam = env.scene["head_cam"]
    top = TopView(w.map, cfg)
    overlay = HeadOverlay(cfg["robot"]["fov_deg"])
    eye_s = None
    from isaac_follow.video import StreamingVideo
    where = f"InteriorGS {args.interiorgs}" if args.interiorgs else "procedural scene"
    title = (f"{args.scenario} | {where} | {w.people.n - 1} other people | "
             f"{'FULL system (C1+P2+M2+S1)' if args.method == 'full' else 'C0 reactive follower'} | walking K1 (our policy)")
    sv = StreamingVideo(os.path.join(args.out, f"showcase_{'interiorgs_' if args.interiorgs else ''}"
                                               f"{args.scenario}_{args.seed}_{args.method}"), args.fps, title)

L = {k: [] for k in ("robot", "cmd", "target", "people", "sel_gt", "in_fov", "visible", "state", "collided",
                     "robot_person_d", "trunk_z", "head_ang_vel", "heading_target")}
cmd = np.zeros(3)
last_cmd_t = 0.0
n_brain = int(args.seconds / DT_BRAIN)
prev_pos, prev_head = w.people.pos.copy(), w.people.heading.copy()
t0 = time.time()
fell = False
for kb in range(n_brain):
    # ---------------- brain tick (10 Hz)
    pose, z = robot_pose2d()
    w.robot._update_odom(w.robot.pose, pose)
    w.robot.pose = pose
    w.robot.t = kb * DT_BRAIN
    cam, cam_z = head_cam_pose2d()
    gt = GroundTruth(cam_pose=cam, robot_pose=pose.copy(), odom=w.robot.odom.copy(), pos=w.people.pos,
                     heading=w.people.heading, head=w.people.head, person_radius=w.people.radius,
                     los_fn=los_fn, ray_fn=grid.raycast, app_fn=app_fn, app_dim=w.app.dim)
    bobs = per.step_gt(gt)
    raw = brain.step(bobs)
    if raw is None or not np.all(np.isfinite(raw)):
        raw = np.zeros(3)
    target_cmd = np.clip(raw, LO, HI)
    cmd = cmd + np.clip(target_cmd - cmd, -RATE, RATE)        # rate limit between brain updates
    # metrics log (state at the brain tick)
    tp = w.people.pos[ti]
    d = tp - cam[:2]
    bearing = wrap(np.arctan2(d[1], d[0]) - cam[2])
    L["in_fov"].append(bool(abs(bearing) <= np.deg2rad(cfg["robot"]["fov_deg"]) / 2 and
                            cfg["robot"]["cam_range"][0] <= np.hypot(*d) <= cfg["robot"]["cam_range"][1]))
    L["visible"].append(bool(per.last_info["visible"][ti]))
    L["sel_gt"].append(-2 if brain.selected is None else int(brain.selected["_gt_id"]))
    L["state"].append(brain.state)
    # ---------------- people advance 0.1 s (2D sim, robot pose from Isaac)
    prev_pos, prev_head = w.people.pos.copy(), w.people.heading.copy()
    L["robot_person_d"].append(np.hypot(*(w.people.pos - pose[:2]).T))
    w.people.step(pose[:2], w.robot.radius)
    # ---------------- 5 policy steps (50 Hz)
    for s in range(SUB):
        a = (s + 1) / SUB
        hp = prev_pos + a * (w.people.pos - prev_pos)
        hh = prev_head + a * wrap(w.people.heading - prev_head)
        set_people(hp, hh)
        cmd_term.set(cmd)
        with torch.no_grad():
            act = policy(obs["policy"])
        if args.debug and kb * SUB + s < 30:
            o = obs["policy"][0]
            # history layout: each term's 5-step history is contiguous (term-major); take the newest entry of each
            dims = [3, 3, 3, 12, 12, 12, 2]
            last, off = [], 0
            for d in dims:
                last.append(o[off + 4 * d: off + 5 * d])
                off += 5 * d
            last = torch.cat(last)
            print(f"[debug] step {kb * SUB + s:2d} z={float(robot.data.root_pos_w[0, 2]):.3f} |a|max={float(act.abs().max()):.2f} "
                  f"angvel={last[0:3].tolist()} grav={last[3:6].tolist()} cmd={last[6:9].tolist()} "
                  f"qrel_max={float(last[9:21].abs().max()):.3f} qd_max={float(last[21:33].abs().max()):.3f} "
                  f"phase={last[45:47].tolist()} obs_dim={o.numel()}", flush=True)
            if kb * SUB + s == 0:
                print("[debug] joint_names", robot.joint_names, flush=True)
                print("[debug] joint_pos", [round(x, 3) for x in robot.data.joint_pos[0].tolist()], flush=True)
                print("[debug] default ", [round(x, 3) for x in robot.data.default_joint_pos[0].tolist()], flush=True)
                print("[debug] root_pos", robot.data.root_pos_w[0].tolist(), "root_quat", robot.data.root_quat_w[0].tolist(), flush=True)
                print("[debug] obs term order", env.observation_manager.active_terms["policy"], flush=True)
        obs, _, _, _, _ = env.step(act)
        if args.video and (kb * SUB + s) % max(1, round(50 / args.fps)) == 0:
            rp, _ = robot_pose2d()
            eye, tgt, eye_s = chase_eye_target(rp, eye_s)
            chase_cam.set_world_poses_from_view(torch.tensor([eye], device=env.device, dtype=torch.float32),
                                                torch.tensor([tgt], device=env.device, dtype=torch.float32))
            rp_now, _ = robot_pose2d()
            sel = brain.selected["_gt_id"] if brain.selected is not None else None
            status = ("following TARGET" if sel == ti else "WRONG PERSON" if sel is not None and sel >= 0
                      else brain.state)
            sv.add(chase_cam.data.output["rgb"][0, ..., :3].cpu().numpy(),
                   overlay.draw(head_cam.data.output["rgb"][0, ..., :3].cpu().numpy(), bobs, brain, w.robot.odom, ti, cam_z),
                   top.draw(w, brain), kb * DT_BRAIN + s * DT_POLICY,
                   float(np.hypot(*(w.people.pos[ti] - rp_now[:2]))), status)
    pose_after, z_after = robot_pose2d()
    L["robot"].append(pose_after)
    L["cmd"].append(cmd.copy())
    L["target"].append(w.people.pos[ti].copy())
    L["heading_target"].append(float(w.people.heading[ti]))
    L["people"].append(w.people.pos.copy())
    L["collided"].append(bool(grid.clearance(pose_after[:2]) < 0.12))
    L["trunk_z"].append(float(z_after))
    L["head_ang_vel"].append(float(robot.data.body_ang_vel_w[0, head_id].norm().cpu()))
    if z_after < 0.30:
        fell = True
        print(f"[run_follow] K1 FELL at t={kb * DT_BRAIN:.1f}s")
        break
    if kb % 10 == 0:
        print(f"[run_follow] t={kb * DT_BRAIN:5.1f}s state={brain.state:18s} cmd={np.round(cmd, 2)} "
              f"dist={np.hypot(*(w.people.pos[ti] - pose_after[:2])):.2f} wall={(time.time() - t0):.0f}s", flush=True)

log = {k: np.asarray(v) for k, v in L.items()}
log["target_index"] = ti
log["dt"] = DT_BRAIN
log["n_swaps"] = per.tracker.n_swaps
m = episode_metrics(log)
m.update(scenario=args.scenario, seed=args.seed, method=args.method, fell=fell, crowd=w.people.n - 1,
         sim_seconds=len(log["robot"]) * DT_BRAIN, wall_seconds=time.time() - t0,
         min_wall_clearance=float(min(grid.clearance(np.asarray(L["robot"])[:, :2]))),
         head_ang_vel_rms=float(np.sqrt(np.mean(np.square(L["head_ang_vel"])))))
print(json.dumps(m, indent=1, default=float))
with open(os.path.join(args.out, f"metrics_{args.scenario}_{args.seed}_{args.method}.json"), "w") as f:
    json.dump(m, f, indent=1, default=float)
np.savez_compressed(os.path.join(args.out, f"log_{args.scenario}_{args.seed}_{args.method}.npz"),
                    **{k: v for k, v in log.items() if isinstance(v, np.ndarray) and v.dtype != object})
if args.video:
    sv.close()
import sys as _sys
_sys.stdout.flush()
os._exit(0)   # Kit shutdown can hang for hours on CHTC nodes
