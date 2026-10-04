"""Custom MDP terms for the K1 walker (PROJECT_SPEC B.3).

- FollowVelocityCommand: commands shaped like the following brain's output: resampled every 2-6 s with smooth
  ramps AND sudden changes, stop-and-go, turn-in-place, ~15% zero commands.
- gait_phase: sin/cos gait clock (zeros while the command is ~zero, so standing is a separate mode).
- randomize_motor_strength: scales actuator effort limits per env (+-10%).
- stand_still_feet_contact: reward both feet on the ground when the command is zero.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------- commands
class FollowVelocityCommand(UniformVelocityCommand):
    """Target command is resampled every 2-6 s; the issued command either ramps toward it with acceleration
    limits (smooth) or jumps to it (sudden). Modes: stand (zero), turn-in-place, walk (uniform in ranges)."""

    cfg: "FollowVelocityCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.target = torch.zeros(self.num_envs, 3, device=self.device)
        self.sudden = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.acc = torch.tensor([cfg.acc_lin, cfg.acc_lin, cfg.acc_ang], device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return
        r = torch.empty(n, device=self.device)
        tgt = torch.zeros(n, 3, device=self.device)
        tgt[:, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        tgt[:, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        tgt[:, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # small lateral commands are rarer than forward ones when following: zero vy half the time
        tgt[:, 1] *= (torch.rand(n, device=self.device) < 0.5).float()
        u = torch.rand(n, device=self.device)
        stand = u < self.cfg.p_stand
        turn = (u >= self.cfg.p_stand) & (u < self.cfg.p_stand + self.cfg.p_turn_in_place)
        tgt[stand] = 0.0
        tgt[turn, :2] = 0.0
        wz = torch.empty(int(turn.sum()), device=self.device).uniform_(0.3, self.cfg.ranges.ang_vel_z[1])
        tgt[turn, 2] = wz * (torch.randint(0, 2, (int(turn.sum()),), device=self.device) * 2 - 1)
        self.target[env_ids] = tgt
        self.sudden[env_ids] = torch.rand(n, device=self.device) < self.cfg.p_sudden
        self.is_standing_env[env_ids] = stand
        self.is_heading_env[env_ids] = False

    def _update_command(self):
        dt = self._env.step_dt
        step = (self.target - self.vel_command_b).clamp(-self.acc * dt, self.acc * dt)
        self.vel_command_b[:] = torch.where(self.sudden[:, None], self.target, self.vel_command_b + step)
        # tiny commands are exactly zero (the brain also sends exact zeros when stopping)
        small = self.vel_command_b.abs() < 0.02
        self.vel_command_b[small] = 0.0

    def reset(self, env_ids: Sequence[int] | None = None):
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.vel_command_b[ids] = 0.0           # every episode starts from standing
        return extras


@configclass
class FollowVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = FollowVelocityCommand
    p_stand: float = 0.15
    p_turn_in_place: float = 0.10
    p_sudden: float = 0.35
    acc_lin: float = 0.6        # m/s^2 for smooth ramps
    acc_ang: float = 1.5        # rad/s^2
    heading_command: bool = False
    rel_heading_envs: float = 0.0
    rel_standing_envs: float = 0.0


# ---------------------------------------------------------------------------- observations
def gait_phase(env: "ManagerBasedRLEnv", period: float = 0.7, command_name: str = "base_velocity") -> torch.Tensor:
    t = env.episode_length_buf.float() * env.step_dt
    ph = 2 * math.pi * (t / period)
    cmd = env.command_manager.get_command(command_name)
    moving = (cmd.abs().sum(dim=1) > 0.0).float()[:, None]
    return torch.stack([torch.sin(ph), torch.cos(ph)], dim=1) * moving


def feet_contact(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Binary contact state per foot (privileged critic input)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1)[0]
    return (f > threshold).float()


# ---------------------------------------------------------------------------- events
def hold_default_targets(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """PD targets of joints the policy does not control (head, arms) = the default (walking) pose.
    Isaac Lab initializes all joint position targets to ZERO and only action terms update them, so without
    this the arms are driven to the URDF zero pose (T-pose)."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = asset_cfg.joint_ids
    tgt = asset.data.default_joint_pos[env_ids][:, ids]
    asset.set_joint_position_target(tgt, joint_ids=ids, env_ids=env_ids)

def randomize_motor_strength(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None, scale_range=(0.9, 1.1),
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    for name, act in asset.actuators.items():
        if not hasattr(act, "_k1_base_effort"):
            act._k1_base_effort = act.effort_limit.clone()
        s = torch.empty(len(env_ids), act.effort_limit.shape[1], device=env.device).uniform_(*scale_range)
        act.effort_limit[env_ids] = act._k1_base_effort[env_ids] * s


# ---------------------------------------------------------------------------- rewards
def stand_still_feet_contact(env: "ManagerBasedRLEnv", command_name: str, sensor_cfg: SceneEntityCfg,
                             threshold: float = 1.0) -> torch.Tensor:
    """1 when the command is zero and every foot is in contact, else 0."""
    sensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1)[0]
    both = (f > threshold).all(dim=1).float()
    cmd = env.command_manager.get_command(command_name)
    return both * (cmd.abs().sum(dim=1) == 0.0).float()


def head_ang_vel_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Angular velocity of the head (camera) link: keeps the camera steady (B.5 check 4)."""
    asset = env.scene[asset_cfg.name]
    w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0]]
    return (w ** 2).sum(dim=1)

