"""Velocity command term whose value is set from outside (by the following brain), so the trained walker
policy sees exactly the observation layout it was trained with."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class ExternalVelocityCommand(CommandTerm):
    cfg: "ExternalVelocityCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def set(self, cmd):
        self.vel_command_b[:] = torch.as_tensor(cmd, dtype=torch.float32, device=self.device).reshape(-1, 3)

    def _update_metrics(self):
        self.metrics["error_vel_xy"] += torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1)
        self.metrics["error_vel_yaw"] += torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass


@configclass
class ExternalVelocityCommandCfg(CommandTermCfg):
    class_type: type = ExternalVelocityCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
