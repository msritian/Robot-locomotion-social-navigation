"""Debug: does the exported TorchScript walker match RSL-RL's own policy, and does it stand/walk in the
training env (play config) and in the closed-loop follow env?

    python debug_policy.py --checkpoint model_3998.pt --policy k1_walker.pt --headless
"""
import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--policy", required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import k1_walker  # noqa: E402,F401
from k1_walker.agents.rsl_rl_ppo_cfg import K1FlatPPORunnerCfg  # noqa: E402

DEV = "cuda:0"
cfg = parse_env_cfg("K1-Velocity-Flat-Play-v0", device=DEV, num_envs=4)
cfg.events.push_robot = None
env = gym.make("K1-Velocity-Flat-Play-v0", cfg=cfg)
venv = RslRlVecEnvWrapper(env)
agent = K1FlatPPORunnerCfg()
runner = OnPolicyRunner(venv, agent.to_dict(), log_dir=None, device=DEV)
runner.load(args.checkpoint)
ref = runner.get_inference_policy(device=DEV)
ts = torch.jit.load(args.policy, map_location=DEV).eval()
obs = venv.get_observations()
o = obs["policy"] if isinstance(obs, dict) or hasattr(obs, "keys") else obs
print("obs type", type(obs), "policy obs shape", tuple(o.shape), flush=True)
with torch.no_grad():
    a_ref = ref(obs)
    a_ts = ts(o)
print("max |a_ref - a_ts| at reset:", float((a_ref - a_ts).abs().max()), flush=True)
robot = env.unwrapped.scene["robot"]
for name, pol in (("rsl_rl", lambda ob: ref(ob)), ("torchscript", lambda ob: ts(ob["policy"] if hasattr(ob, "keys") else ob))):
    obs, _ = venv.reset()
    zmin = 10.0
    for k in range(250):          # 5 s
        with torch.no_grad():
            act = pol(obs)
        obs, _, done, _ = venv.step(act)
        zmin = min(zmin, float(robot.data.root_pos_w[:, 2].min()))
    cmd = env.unwrapped.command_manager.get_command("base_velocity")[0].tolist()
    print(f"[{name}] 5 s rollout: min trunk z {zmin:.3f}, last cmd {cmd}, "
          f"vel_b {robot.data.root_lin_vel_b[0, :2].tolist()}", flush=True)
app.close()
