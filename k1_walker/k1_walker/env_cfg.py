"""K1 velocity-tracking walker (PROJECT_SPEC B.3), Isaac Lab manager-based task.

Template: Isaac Lab's G1/H1 flat velocity tasks. Policy controls the 12 leg joints (head and arms held at the
walking default pose by their PD actuators). Physics 200 Hz, policy 50 Hz (decimation 4); Booster's K1 walking
controller also runs its policy at 50 Hz with action scale 0.25 (booster_deploy).
Offline-friendly: plane terrain, no Nucleus materials or HDR sky textures.
"""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vmdp
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from . import mdp as k1mdp
from .k1_cfg import FOOT_BODIES, HEAD_LINK, K1_CFG, LEG_JOINT_NAMES

LEGS = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)
NON_FOOT_BODIES = ["trunk", ".*_hip_pitch_link", ".*_hip_roll_link", ".*_hip_yaw_link", ".*_knee_pitch_link",
                   ".*shoulder.*", ".*elbow.*", ".*head.*"]
# Section B.3 command ranges
VX, VY, WZ = (-0.3, 0.6), (-0.3, 0.3), (-1.0, 1.0)


@configclass
class K1ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=vmdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=vmdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=vmdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=vmdp.joint_pos_rel, params={"asset_cfg": LEGS}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=vmdp.joint_vel_rel, params={"asset_cfg": LEGS}, scale=0.1,
                            noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=vmdp.last_action)
        gait_phase = ObsTerm(func=k1mdp.gait_phase, params={"period": 0.7})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5          # short history (B.3)

    @configclass
    class CriticCfg(PolicyCfg):
        """Privileged critic: noise-free policy terms + base linear velocity, base height, foot contacts."""
        base_lin_vel = ObsTerm(func=vmdp.base_lin_vel)
        base_height = ObsTerm(func=vmdp.base_pos_z)
        feet_contact = ObsTerm(func=k1mdp.feet_contact,
                               params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES)})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class K1Rewards(RewardsCfg):
    termination_penalty = RewTerm(func=vmdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(func=vmdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5,
                                   params={"command_name": "base_velocity", "std": 0.25})
    track_ang_vel_z_exp = RewTerm(func=vmdp.track_ang_vel_z_world_exp, weight=1.0,
                                  params={"command_name": "base_velocity", "std": 0.4})
    feet_air_time = RewTerm(
        func=vmdp.feet_air_time_positive_biped, weight=0.5,
        params={"command_name": "base_velocity", "threshold": 0.35,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES)},
    )
    feet_slide = RewTerm(func=vmdp.feet_slide, weight=-0.2,
                         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
                                 "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES)})
    dof_pos_limits = RewTerm(func=vmdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": LEGS})
    joint_deviation_hip = RewTerm(func=vmdp.joint_deviation_l1, weight=-0.1,
                                  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint",
                                                                                            ".*_hip_roll_joint"])})
    base_height = RewTerm(func=vmdp.base_height_l2, weight=-5.0, params={"target_height": 0.53})
    stand_still = RewTerm(func=vmdp.stand_still_joint_deviation_l1, weight=-0.5,
                          params={"command_name": "base_velocity", "asset_cfg": LEGS})
    stand_feet_contact = RewTerm(func=k1mdp.stand_still_feet_contact, weight=0.5,
                                 params={"command_name": "base_velocity",
                                         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES)})
    head_ang_vel = RewTerm(func=k1mdp.head_ang_vel_l2, weight=-0.02,
                           params={"asset_cfg": SceneEntityCfg("robot", body_names=[HEAD_LINK])})


@configclass
class K1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: K1ObservationsCfg = K1ObservationsCfg()
    rewards: K1Rewards = K1Rewards()

    def __post_init__(self):
        super().__post_init__()
        # --- scene: flat floor, offline lighting, K1
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.visual_material = None
        self.scene.height_scanner = None
        self.scene.sky_light = AssetBaseCfg(prim_path="/World/skyLight",
                                            spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.9, 0.9, 0.92)))
        self.scene.robot = K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 2.0
        self.curriculum.terrain_levels = None
        # --- rates (B.3): physics 200 Hz, policy 50 Hz
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        # --- action: 12 leg joints, offsets x 0.25 around the walking default pose
        self.actions.joint_pos = vmdp.JointPositionActionCfg(asset_name="robot", joint_names=LEG_JOINT_NAMES,
                                                             scale=0.25, use_default_offset=True,
                                                             preserve_order=True)
        # --- commands (B.3)
        self.commands.base_velocity = k1mdp.FollowVelocityCommandCfg(
            asset_name="robot", resampling_time_range=(2.0, 6.0), debug_vis=False,
            ranges=k1mdp.FollowVelocityCommandCfg.Ranges(lin_vel_x=VX, lin_vel_y=VY, ang_vel_z=WZ,
                                                         heading=(-math.pi, math.pi)),
        )
        # --- domain randomization (B.3)
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.2)
        self.events.add_base_mass = EventTerm(
            func=vmdp.randomize_rigid_body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                    "mass_distribution_params": (0.9, 1.1), "operation": "scale"})
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = EventTerm(func=vmdp.push_by_setting_velocity, mode="interval",
                                           interval_range_s=(5.0, 10.0),
                                           params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}})
        self.events.actuator_gains = EventTerm(
            func=vmdp.randomize_actuator_gains, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "stiffness_distribution_params": (0.9, 1.1), "damping_distribution_params": (0.9, 1.1),
                    "operation": "scale", "distribution": "uniform"})
        self.events.motor_strength = EventTerm(func=k1mdp.randomize_motor_strength, mode="reset",
                                               params={"scale_range": (0.9, 1.1)})
        self.events.hold_head_arms = EventTerm(
            func=k1mdp.hold_default_targets, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*head.*", ".*shoulder.*", ".*elbow.*"])})
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        }
        # --- reward weights (standard set, B.3)
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.dof_torques_l2.params["asset_cfg"] = LEGS
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = LEGS
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_l2 = RewTerm(func=vmdp.action_l2, weight=-0.002)   # keep actions small (v2)
        self.rewards.undesired_contacts = RewTerm(
            func=vmdp.undesired_contacts, weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=NON_FOOT_BODIES), "threshold": 1.0})
        # --- terminations (B.3)
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["trunk", ".*head.*"]
        self.terminations.base_too_low = DoneTerm(func=vmdp.root_height_below_minimum,
                                                  params={"minimum_height": 0.30})


@configclass
class K1FlatEnvCfg_PLAY(K1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.base_velocity.debug_vis = True


@configclass
class K1StandEnvCfg(K1FlatEnvCfg_PLAY):
    """M1 stand test: zero commands, no pushes, no DR."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.commands.base_velocity.p_stand = 1.0
        self.events.actuator_gains = None
        self.events.motor_strength = None
