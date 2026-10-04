"""M1: spawn the K1 (22 DoF, official URDF) in Isaac Lab and hold the walking default pose with PD control
(zero leg actions) for 10 s. Pass = no env falls (trunk height stays > 0.35 m, |roll|, |pitch| < 0.5 rad).
Writes stand_test.json (+ an mp4 if --video and cameras are available).

    python k1_walker/scripts/stand_test.py --headless [--video]
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--video", action="store_true")
parser.add_argument("--out", default="stand_out")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import k1_walker  # noqa: E402,F401

os.makedirs(args.out, exist_ok=True)
cfg = parse_env_cfg("K1-Stand-v0", device="cuda:0", num_envs=4)
cfg.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}
env = gym.make("K1-Stand-v0", cfg=cfg, render_mode="rgb_array" if args.video else None)
if args.video:
    env.unwrapped.sim.set_camera_view(eye=(2.0, 2.0, 1.2), target=(0.0, 0.0, 0.5))
    env = gym.wrappers.RecordVideo(env, video_folder=args.out, step_trigger=lambda s: s == 0,
                                   video_length=int(args.seconds / 0.02), disable_logger=True)
obs, _ = env.reset()
robot = env.unwrapped.scene["robot"]
n_steps = int(args.seconds / env.unwrapped.step_dt)
act = torch.zeros(env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim, device="cuda:0")
heights, rp = [], []
for k in range(n_steps):
    obs, rew, term, trunc, info = env.step(act)
    q = robot.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1, 1))
    heights.append(robot.data.root_pos_w[:, 2].cpu())
    rp.append(torch.stack([roll, pitch], 1).abs().max(1).values.cpu())
H = torch.stack(heights)
RP = torch.stack(rp)
from isaaclab.utils.math import quat_apply  # noqa: E402

from k1_walker.k1_cfg import HEAD_CAMERA_OFFSET, HEAD_LINK  # noqa: E402

head = robot.find_bodies(HEAD_LINK)[0][0]
off = torch.tensor(HEAD_CAMERA_OFFSET, device="cuda:0").repeat(robot.num_instances, 1)
cam_pos = robot.data.body_pos_w[:, head] + quat_apply(robot.data.body_quat_w[:, head], off)
res = {
    "seconds": args.seconds, "num_envs": int(H.shape[1]),
    "trunk_height_start": H[0].tolist(), "trunk_height_end": H[-1].tolist(),
    "trunk_height_min": float(H.min()), "max_abs_roll_pitch": float(RP.max()),
    "head_camera_height_end": (cam_pos[:, 2] - env.unwrapped.scene.env_origins[:, 2]).cpu().tolist(),
    "fell": bool((H.min(0).values < 0.35).any() or (RP.max(0).values > 0.5).any()),
    "joint_names": robot.joint_names,
    "stiffness_legs": robot.actuators["legs"].stiffness[0].cpu().tolist(),
}
res["pass"] = not res["fell"]
print(json.dumps({k: v for k, v in res.items() if k != "joint_names"}, indent=1))
with open(os.path.join(args.out, "stand_test.json"), "w") as f:
    json.dump(res, f, indent=1)
env.close()
app.close()
