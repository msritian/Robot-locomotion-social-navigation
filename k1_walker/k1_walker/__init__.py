"""K1 velocity-tracking walker tasks for Isaac Lab (PROJECT_SPEC Part B). Import after the app is launched."""
import gymnasium as gym

from . import agents  # noqa: F401

for _id, _cfg in (("K1-Velocity-Flat-v0", "K1FlatEnvCfg"), ("K1-Velocity-Flat-Play-v0", "K1FlatEnvCfg_PLAY"),
                  ("K1-Stand-v0", "K1StandEnvCfg")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:K1FlatPPORunnerCfg",
        },
    )
