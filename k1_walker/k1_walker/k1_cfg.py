"""Booster K1 (22 DoF) articulation config for the walking task.

Sources (see NOTICE; never invented here):
- Joint motor models (armature, velocity limits, torque-speed knee points), actuator grouping, 0.57 m spawn
  height, solver iterations: booster_train/assets/robots/booster_k1.py @ 8bb5b53.
- WALKING default joint pose, PD stiffness/damping and effort limits: Booster's K1 walking controller
  (booster_deploy @ 7bb1462, tasks/locomotion/robots/k1/__init__.py, K1WalkControllerCfg).
- Actuation delay: 0-4 physics steps (0-20 ms at 200 Hz), the domain-randomization range of PROJECT_SPEC B.3.
  (booster_train uses 2-8 steps for its motion-tracking tasks.)
"""
from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from .booster_actuator import (
    BoosterDelayedPDActuatorCfg,
    BoosterJointE4310,
    BoosterJointE4315,
    BoosterJointE6408,
    BoosterJointE6416,
    BoosterJointHT4438,
    BoosterJointR14,
    BoosterK1AnkleParaWrapperCfg,
)

try:
    from booster_assets import BOOSTER_ASSETS_DIR
except ImportError:  # the container ships booster_assets at /opt/booster_assets
    BOOSTER_ASSETS_DIR = os.environ.get("BOOSTER_ASSETS", "/opt/booster_assets")

K1_URDF = f"{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.urdf"

LEG_JOINTS = [
    ".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint",
    ".*_knee_pitch_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
]
# policy action order (explicit, so exported policies have a fixed joint order)
LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
FOOT_BODIES = ".*_ankle_roll_link"
# The URDF importer merges fixed joints, so the camera link becomes part of the head pitch link.
# Camera = HEAD_LINK + HEAD_CAMERA_OFFSET (head_realsense_rgb_joint origin in the URDF).
HEAD_LINK = "aahead_pitch_link"
HEAD_CAMERA_OFFSET = (0.053617, -0.0115, 0.10231)

# booster_deploy K1WalkControllerCfg (order: head 2, L arm 4, R arm 4, L leg 6, R leg 6)
DEFAULT_JOINT_POS = {
    "aahead_yaw_joint": 0.0, "aahead_pitch_joint": 0.0,
    "aaleft_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": -1.25,
    "left_elbow_pitch_joint": 0.0, "left_elbow_yaw_joint": -0.5,
    "aaright_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": 1.25,
    "right_elbow_pitch_joint": 0.0, "right_elbow_yaw_joint": 0.5,
    ".*_hip_pitch_joint": -0.15, ".*_hip_roll_joint": 0.0, ".*_hip_yaw_joint": 0.0,
    ".*_knee_pitch_joint": 0.3, ".*_ankle_pitch_joint": -0.15, ".*_ankle_roll_joint": 0.0,
}

MIN_DELAY, MAX_DELAY = 0, 4

K1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=K1_URDF,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": BoosterDelayedPDActuatorCfg(
            min_delay=MIN_DELAY, max_delay=MAX_DELAY,
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_pitch_joint"],
            booster_joint_cfgs={
                ".*_hip_pitch_joint": BoosterJointE6408(effort_limit=30.0, stiffness=100.0, damping=2.0),
                ".*_hip_roll_joint": BoosterJointE4315(effort_limit=20.0, stiffness=100.0, damping=2.0),
                ".*_hip_yaw_joint": BoosterJointE4310(effort_limit=15.0, stiffness=100.0, damping=2.0),
                ".*_knee_pitch_joint": BoosterJointE6416(effort_limit=35.0, stiffness=100.0, damping=2.0),
            },
        ),
        "feet": BoosterDelayedPDActuatorCfg(
            min_delay=MIN_DELAY, max_delay=MAX_DELAY,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            booster_joint_cfgs={
                ".*_ankle_pitch_joint": BoosterK1AnkleParaWrapperCfg(
                    base_joint_cfg=BoosterJointE4310(effort_limit=24.0), serial_index=0,
                    armature_ratio=(1.4, 0.4), stiffness=65.0, damping=1.0),
                ".*_ankle_roll_joint": BoosterK1AnkleParaWrapperCfg(
                    base_joint_cfg=BoosterJointE4310(effort_limit=15.0), serial_index=1,
                    armature_ratio=(1.4, 0.4), stiffness=65.0, damping=1.0),
            },
        ),
        "arms": BoosterDelayedPDActuatorCfg(
            min_delay=MIN_DELAY, max_delay=MAX_DELAY,
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_elbow_pitch_joint",
                              ".*_elbow_yaw_joint"],
            booster_joint_cfgs=BoosterJointR14(effort_limit=14.0, stiffness=20.0, damping=2.0),
        ),
        "head": BoosterDelayedPDActuatorCfg(
            min_delay=MIN_DELAY, max_delay=MAX_DELAY,
            joint_names_expr=[".*head.*"],
            booster_joint_cfgs=BoosterJointHT4438(effort_limit=6.0, stiffness=4.0, damping=1.0),
        ),
    },
)
