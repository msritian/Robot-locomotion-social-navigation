"""Bisect why the walker falls in the closed-loop env. Zero velocity command for 5 s; prints min trunk height.
  A: training play env + ExternalVelocityCommand (1 env)
  B: closed-loop FollowEnvCfg, empty map (no walls, no people)
  C: closed-loop FollowEnvCfg, T7 seed 1000 scene (walls + people)
    python debug_bisect.py --variant A --policy k1_walker.pt --headless
"""
import argparse
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=["A", "B", "C"])
ap.add_argument("--policy", required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import k1_walker  # noqa: E402,F401
from isaac_follow.external_command import ExternalVelocityCommandCfg  # noqa: E402

DEV = "cuda:0"
if args.variant == "A":
    cfg = parse_env_cfg("K1-Velocity-Flat-Play-v0", device=DEV, num_envs=1)
    cfg.events.push_robot = None
    cfg.commands.base_velocity = ExternalVelocityCommandCfg()
else:
    from isaac_follow.scene_cfg import make_env_cfg
    from pf.config import load_config
    from pf.eval.scenarios import build
    pcfg = load_config()
    w = build(pcfg, "T7", 1000)
    if args.variant == "B":
        w.map.params["_rects"] = np.zeros((0, 4))
        pos, cols = np.zeros((0, 2)), []
    else:
        pos, cols = w.people.pos, [(0.5, 0.5, 0.5)] * w.people.n
    cfg = make_env_cfg(w.map, pos, cols, video=False)
    x0, y0, yaw0 = w.robot.pose
    cfg.scene.robot.init_state.pos = (float(x0), float(y0), 0.57)
    cfg.scene.robot.init_state.rot = (float(np.cos(yaw0 / 2)), 0.0, 0.0, float(np.sin(yaw0 / 2)))
    cfg.sim.device = DEV
env = ManagerBasedRLEnv(cfg=cfg)
pol = torch.jit.load(args.policy, map_location=env.device).eval()
robot = env.scene["robot"]
obs, _ = env.reset()
print(f"[{args.variant}] events:", [n for n in env.event_manager.active_terms.get("reset", [])],
      "startup:", env.event_manager.active_terms.get("startup", []), flush=True)
zmin, amax = 9.0, 0.0
for k in range(250):
    with torch.no_grad():
        a = pol(obs["policy"])
    obs, *_ = env.step(a)
    z = float(robot.data.root_pos_w[0, 2])
    zmin, amax = min(zmin, z), max(amax, float(a.abs().max()))
    if k in (0, 5, 25, 100, 249):
        print(f"[{args.variant}] step {k} z={z:.3f} |a|={float(a.abs().max()):.2f}", flush=True)
print(f"[{args.variant}] RESULT min_z={zmin:.3f} max|a|={amax:.2f} -> {'FELL' if zmin < 0.3 else 'STOOD'}", flush=True)
print(f"[{args.variant}] stiffness legs", robot.actuators["legs"].stiffness[0].tolist(),
      "effort", robot.actuators["legs"].effort_limit[0].tolist(), flush=True)
import sys as _sys
_sys.stdout.flush()
os._exit(0)   # Kit shutdown can hang for hours on CHTC nodes
