"""Walker evaluation (PROJECT_SPEC B.5) and response measurement (B.6) for an exported TorchScript policy.

B.5  (2) tracking: 5 min random command sequence (same sampler as training, no pushes) -> RMS error vx, vy, wz
     (3) robustness: 20 episodes x 60 s with random commands incl. stop-and-go / turn-in-place -> falls
     (4) head stability: RMS of the head-link angular velocity while walking
     (1) video of forward / sidestep / turn-in-place / stop (with --video, needs cameras)
B.6  step tests (vx 0->0.4, wz 0->0.8, stop from vx 0.4, vy 0->0.2): rise time, overshoot, delay, max speed,
     acceleration; first-order-plus-delay fit per axis; head-camera sway (lateral + yaw amplitude / frequency)
     -> walker_response.yaml (replaces the assumed Stage C robot parameters, M10)

    python k1_walker/scripts/eval_walker.py --policy k1_walker.pt --headless [--video]
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--out", default="walker_eval")
ap.add_argument("--video", action="store_true")
ap.add_argument("--robust_envs", type=int, default=20)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import k1_walker  # noqa: E402,F401
from k1_walker.k1_cfg import HEAD_CAMERA_OFFSET, HEAD_LINK  # noqa: E402

os.makedirs(args.out, exist_ok=True)
DEV = "cuda:0"
policy = torch.jit.load(args.policy, map_location=DEV).eval()


def make_env(n, external=False, video=False, episode_s=60.0):
    cfg = parse_env_cfg("K1-Velocity-Flat-Play-v0", device=DEV, num_envs=n)
    cfg.episode_length_s = episode_s
    cfg.events.push_robot = None
    cfg.scene.env_spacing = 3.0
    if external:
        from isaac_follow.external_command import ExternalVelocityCommandCfg
        cfg.commands.base_velocity = ExternalVelocityCommandCfg()
    env = gym.make("K1-Velocity-Flat-Play-v0", cfg=cfg, render_mode="rgb_array" if video else None)
    return env


def yaw_of(q):
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def run_random(n_envs, seconds):
    """Random command sequences from the training sampler; returns per-step (cmd, vel_b) and fall flags."""
    env = make_env(n_envs, episode_s=seconds + 5)
    u = env.unwrapped
    robot = u.scene["robot"]
    head = robot.find_bodies(HEAD_LINK)[0][0]
    obs, _ = env.reset()
    fell = torch.zeros(n_envs, dtype=torch.bool, device=DEV)
    cmds, vels, head_w = [], [], []
    for _ in range(int(seconds / u.step_dt)):
        with torch.no_grad():
            obs, _, term, trunc, _ = env.step(policy(obs["policy"]))
        fell |= term
        c = u.command_manager.get_command("base_velocity")
        v = torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], 1)
        cmds.append(c.cpu().numpy().copy())
        vels.append(v.cpu().numpy())
        head_w.append(robot.data.body_ang_vel_w[:, head].norm(dim=1).cpu().numpy())
    env.close()
    return np.array(cmds), np.array(vels), np.array(head_w), fell.cpu().numpy()


res = {}
# ---------------- B.5 (2) tracking over 5 minutes, 4 independent sequences
C, V, HW, F = run_random(4, 300.0)
lag = 10   # compare against the command 0.2 s earlier (the walker's response lag is part of B.6, not error)
err = V[lag:] - C[:-lag]
alive = ~F
res["tracking_rms"] = {"vx": float(np.sqrt(np.mean(err[:, alive, 0] ** 2))),
                       "vy": float(np.sqrt(np.mean(err[:, alive, 1] ** 2))),
                       "wz": float(np.sqrt(np.mean(err[:, alive, 2] ** 2))),
                       "lag_compensation_s": lag * 0.02, "envs_alive": int(alive.sum())}
err0 = V - C
res["tracking_rms_no_lag_comp"] = {k: float(np.sqrt(np.mean(err0[:, alive, i] ** 2))) for i, k in enumerate(("vx", "vy", "wz"))}
res["tracking_pass"] = bool(res["tracking_rms"]["vx"] < 0.1 and res["tracking_rms"]["vy"] < 0.1 and res["tracking_rms"]["wz"] < 0.2)
res["head_ang_vel_rms_walking"] = float(np.sqrt(np.mean(HW[:, alive] ** 2)))
# ---------------- B.5 (3) robustness: 20 x 60 s
_, _, _, F20 = run_random(args.robust_envs, 60.0)
res["robustness"] = {"episodes": args.robust_envs, "falls": int(F20.sum()), "pass": bool(F20.sum() == 0)}
print(json.dumps(res, indent=1), flush=True)

# ---------------- B.6 step tests with an external command
env = make_env(1, external=True, video=args.video, episode_s=200.0)
u = env.unwrapped
robot = u.scene["robot"]
head = robot.find_bodies(HEAD_LINK)[0][0]
term = u.command_manager.get_term("base_velocity")
off = torch.tensor([HEAD_CAMERA_OFFSET], device=DEV)
if args.video:
    env = gym.wrappers.RecordVideo(env, video_folder=args.out, step_trigger=lambda s: s == 0, video_length=100000,
                                   name_prefix="k1_walker_demo", disable_logger=True)


def segment(obs, cmd, seconds, rec):
    term.set(cmd)
    for _ in range(int(seconds / 0.02)):
        with torch.no_grad():
            obs, *_ = env.step(policy(obs["policy"]))
        v = torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], 1)[0].cpu().numpy()
        hp = robot.data.body_pos_w[:, head] + quat_apply(robot.data.body_quat_w[:, head], off)
        rel = quat_apply_inverse(robot.data.root_quat_w, hp - robot.data.root_pos_w)[0].cpu().numpy()
        rec.append({"cmd": list(cmd), "v": v.tolist(), "cam_rel": rel.tolist(),
                    "head_yaw_rel": float((yaw_of(robot.data.body_quat_w[:, head]) - yaw_of(robot.data.root_quat_w))[0]),
                    "z": float(robot.data.root_pos_w[0, 2])})
    return obs


obs, _ = env.reset()
rec = []
script = [((0, 0, 0), 3), ((0.4, 0, 0), 6), ((0, 0, 0), 4), ((0, 0, 0.8), 5), ((0, 0, 0), 4),
          ((0, 0.2, 0), 5), ((0, 0, 0), 4), ((0.6, 0, 0), 5), ((-0.3, 0, 0), 5), ((0, 0, -1.0), 4), ((0, 0, 0), 3)]
seg_idx = []
for cmd, sec in script:
    seg_idx.append((len(rec), cmd, sec))
    obs = segment(obs, cmd, sec, rec)
env.close()
V = np.array([r["v"] for r in rec])
t = np.arange(len(V)) * 0.02


def fit_fopd(y, u_step, dt=0.02):
    """First-order plus delay fit to a step response of size u_step starting at index 0."""
    best = None
    y_ss = np.mean(y[-int(1.0 / dt):])
    for delay in np.arange(0.0, 0.5, 0.02):
        for tau in np.arange(0.05, 2.0, 0.025):
            tt = np.arange(len(y)) * dt
            m = np.where(tt < delay, 0.0, y_ss * (1 - np.exp(-(tt - delay) / tau)))
            e = np.mean((m - y) ** 2)
            if best is None or e < best[0]:
                best = (e, delay, tau)
    k10 = np.argmax(y >= 0.1 * y_ss) * dt
    k90 = np.argmax(y >= 0.9 * y_ss) * dt
    return {"delay_s": round(float(best[1]), 3), "tau_s": round(float(best[2]), 3), "gain": float(y_ss / u_step),
            "steady_state": float(y_ss), "rise_time_10_90_s": float(k90 - k10),
            "overshoot_frac": float(max(0.0, (np.max(y) - y_ss) / max(abs(y_ss), 1e-6))),
            "max_accel": float(np.max(np.abs(np.diff(y))) / dt)}


steps = {}
for i0, cmd, sec in seg_idx:
    i1 = i0 + int(sec / 0.02)
    if cmd == (0.4, 0, 0):
        steps["vx_0_to_0.4"] = fit_fopd(V[i0:i1, 0], 0.4)
    if cmd == (0, 0, 0.8):
        steps["wz_0_to_0.8"] = fit_fopd(V[i0:i1, 2], 0.8)
    if cmd == (0, 0.2, 0):
        steps["vy_0_to_0.2"] = fit_fopd(V[i0:i1, 1], 0.2)
# stop from 0.4: segment right after the vx step
i_stop = seg_idx[2][0]
y = V[i_stop:i_stop + int(4 / 0.02), 0]
steps["stop_from_0.4"] = {"time_to_below_0.05_s": float(np.argmax(np.abs(y) < 0.05) * 0.02),
                          "max_decel": float(np.max(np.abs(np.diff(y))) / 0.02)}
max_speeds = {"vx_fwd_at_0.6_cmd": float(np.mean(V[seg_idx[7][0] + 150:seg_idx[7][0] + 250, 0])),
              "vx_back_at_-0.3_cmd": float(np.mean(V[seg_idx[8][0] + 150:seg_idx[8][0] + 250, 0])),
              "wz_at_-1.0_cmd": float(np.mean(V[seg_idx[9][0] + 100:seg_idx[9][0] + 200, 2]))}
# head-camera sway while walking forward at 0.4 m/s (steady part)
i0 = seg_idx[1][0] + 100
cam = np.array([r["cam_rel"] for r in rec[i0:i0 + 200]])
hy = np.array([r["head_yaw_rel"] for r in rec[i0:i0 + 200]])
lat = cam[:, 1] - cam[:, 1].mean()
spec = np.abs(np.fft.rfft(lat))
freq = np.fft.rfftfreq(len(lat), 0.02)
sway = {"lateral_amp_m": float((lat.max() - lat.min()) / 2), "yaw_amp_deg": float(np.rad2deg((hy.max() - hy.min()) / 2)),
        "freq_hz": float(freq[1:][np.argmax(spec[1:])]), "camera_height_m": float(cam[:, 2].mean() + np.mean([r["z"] for r in rec[i0:i0 + 200]]))}
res.update({"steps": steps, "max_speeds": max_speeds, "head_sway": sway})
# ---------------- walker_response.yaml (measured robot model for Stage C, M10)
tau = float(np.mean([steps[k]["tau_s"] for k in ("vx_0_to_0.4", "vy_0_to_0.2")]))
resp = {
    "source": "measured by k1_walker/scripts/eval_walker.py (B.6) on " + os.path.basename(args.policy),
    "robot": {
        "tau": [tau, tau, steps["wz_0_to_0.8"]["tau_s"]],
        "latency": float(np.mean([s["delay_s"] for k, s in steps.items() if "delay_s" in s])),
        "vx_range": [max(-0.3, round(max_speeds["vx_back_at_-0.3_cmd"], 3)), min(0.6, round(max_speeds["vx_fwd_at_0.6_cmd"], 3))],
        "vy_range": [-0.2, 0.2], "wz_range": [-min(1.0, abs(max_speeds["wz_at_-1.0_cmd"])), min(1.0, abs(max_speeds["wz_at_-1.0_cmd"]))],
        "acc_lin": float(max(steps["vx_0_to_0.4"]["max_accel"], 0.1)), "acc_ang": float(max(steps["wz_0_to_0.8"]["max_accel"], 0.5)),
        "sway_lat": sway["lateral_amp_m"], "sway_yaw_deg": sway["yaw_amp_deg"], "sway_hz": sway["freq_hz"],
        "camera_height": sway["camera_height_m"],
    },
    "limits": {"vx": None, "vy": None, "wz": None},
}
resp["limits"] = {"vx": resp["robot"]["vx_range"], "vy": resp["robot"]["vy_range"], "wz": resp["robot"]["wz_range"]}
with open(os.path.join(args.out, "walker_response.yaml"), "w") as f:
    yaml.safe_dump(resp, f, sort_keys=False)
with open(os.path.join(args.out, "walker_eval.json"), "w") as f:
    json.dump(res, f, indent=1)
np.savez_compressed(os.path.join(args.out, "step_tests.npz"), t=t, v=V, cmd=np.array([r["cmd"] for r in rec]))
print(json.dumps(res, indent=1))
app.close()
