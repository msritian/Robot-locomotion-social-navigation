"""3D Isaac scene built from a Stage C 2D map (PROJECT_SPEC 13.3/13.4).

- Every obstacle rectangle of the map (exact decomposition of the occupancy grid) becomes a static box collider:
  walls 2.5 m high, furniture (rects not touching the outer wall mass) 0.75 m (desks/tables) - so the walker
  physically cannot pass where the 2D planner cannot.
- People: kinematic mannequins (capsule body here; sphere head + colored shirt band added by run_follow.py),
  posed every step from the Stage C people simulator. No collisions (robot-person contact is checked by distance, Section 10).
- Cameras: K1 head camera (same HFOV as Stage C), third-person chase camera.
"""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from k1_walker.env_cfg import K1FlatEnvCfg_PLAY
from k1_walker.k1_cfg import HEAD_CAMERA_OFFSET, HEAD_LINK

from .external_command import ExternalVelocityCommandCfg

WALL_H = 2.5
FURNITURE_H = 0.75
FLOOR_C = (0.62, 0.60, 0.56)
WALL_C = (0.86, 0.85, 0.82)
FURN_C = (0.45, 0.33, 0.22)
TARGET_SHIRT = (0.85, 0.10, 0.10)
LOOKALIKE_SHIRTS = [(0.80, 0.18, 0.14), (0.75, 0.12, 0.20)]
OTHER_SHIRTS = [(0.15, 0.35, 0.75), (0.20, 0.60, 0.30), (0.90, 0.75, 0.15), (0.45, 0.25, 0.60), (0.95, 0.55, 0.20),
                (0.30, 0.65, 0.70), (0.55, 0.55, 0.55), (0.10, 0.20, 0.35)]


def classify_rects(world_map, furniture_max_m2=3.0):
    """Walls vs furniture: a rect belongs to a wall if its connected obstacle component touches the map border OR is
    larger than furniture_max_m2 (e.g. the inner block of a corridor loop); small free-standing blobs are furniture."""
    from scipy import ndimage
    occ = world_map.grid.occ
    res = world_map.grid.res
    lab, n = ndimage.label(occ)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    area = ndimage.sum(occ, lab, index=np.arange(1, n + 1)) * res * res
    walls, furn = [], []
    for x0, y0, x1, y1 in world_map.obstacles:
        cy, cx = int(((y0 + y1) / 2) / res), int(((x0 + x1) / 2) / res)
        cy, cx = min(cy, occ.shape[0] - 1), min(cx, occ.shape[1] - 1)
        k = lab[cy, cx]
        is_wall = k in border or (k > 0 and area[k - 1] > furniture_max_m2)
        (walls if is_wall else furn).append((x0, y0, x1, y1))
    return walls, furn


def _box(i, rect, h, color, kind):
    x0, y0, x1, y1 = rect
    return AssetBaseCfg(
        prim_path=f"/World/Scene/{kind}_{i:03d}",
        spawn=sim_utils.CuboidCfg(size=(x1 - x0, y1 - y0, h), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=((x0 + x1) / 2, (y0 + y1) / 2, h / 2)),
    )


def _person(i, color, pos):
    return RigidObjectCfg(
        prim_path=f"/World/Scene/Person_{i:02d}",
        spawn=sim_utils.CapsuleCfg(
            radius=0.2, height=1.0, axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=60.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28), roughness=0.7),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(pos[0], pos[1], 0.7)),
    ), color


def make_env_cfg(world_map, people_pos, people_colors, fov_deg=90.0, video=False, width=1280, height=720,
                 interior_usda=None, env_usd=None, env_offset=(0.0, 0.0, 0.0)):
    """people_pos: (N, 2) initial positions; people_colors: N shirt colors. Robot spawn is set by the caller.
    interior_usda: an assembled InteriorGS scene (splat + collision, isaac_follow.interiorgs_scene) used INSTEAD
    of the extruded box walls/furniture (Section 19.2)."""
    walls, furn = classify_rects(world_map) if (interior_usda is None and env_usd is None) else ([], [])
    hidden = []
    if env_usd is not None:
        # photoreal scene is visual only -> invisible collision boxes from the sliced map (0.1 m, conservative)
        from pf.world.grid import decompose_rects
        occ, res, f = world_map.grid.occ, world_map.grid.res, 2
        Hh, Ww = occ.shape[0] // f * f, occ.shape[1] // f * f
        coarse = occ[:Hh, :Ww].reshape(Hh // f, f, Ww // f, f).any(axis=(1, 3))
        hidden = [tuple(r) for r in decompose_rects(coarse, res * f)]

    @configclass
    class FollowEnvCfg(K1FlatEnvCfg_PLAY):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 1
            self.scene.env_spacing = 0.0
            self.episode_length_s = 10_000.0
            self.commands.base_velocity = ExternalVelocityCommandCfg()
            self.observations.policy.enable_corruption = False
            for name in ("push_robot", "actuator_gains", "motor_strength", "add_base_mass"):
                if hasattr(self.events, name):
                    setattr(self.events, name, None)
            # keep the reset events but without randomization: joints exactly at the walking default pose (without
            # this they start at 0 = a pose never seen in training -> runaway actions), root exactly at the spawn
            self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
            self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
            self.events.reset_base.params = {
                "pose_range": {k: (0.0, 0.0) for k in ("x", "y", "yaw")},
                "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}}
            self.terminations.time_out = None
            self.terminations.base_too_low = None
            self.terminations.base_contact = None
            self.sim.physics_material.static_friction = 0.8
            self.sim.physics_material.dynamic_friction = 0.8
            self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=FLOOR_C, roughness=0.9)
            self.scene.sky_light = AssetBaseCfg(prim_path="/World/skyLight",
                                                spawn=sim_utils.DomeLightCfg(intensity=900.0, color=(1.0, 0.98, 0.95)))
            self.scene.ceiling_light = AssetBaseCfg(
                prim_path="/World/sunLight",
                spawn=sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0, color=(1.0, 0.97, 0.9)),
                init_state=AssetBaseCfg.InitialStateCfg(rot=(0.92, 0.2, 0.2, 0.2)))
            if interior_usda is not None:
                self.scene.interior = AssetBaseCfg(prim_path="/World/Interior",
                                                   spawn=sim_utils.UsdFileCfg(usd_path=interior_usda))
            for i, r in enumerate(walls):
                setattr(self.scene, f"wall_{i:03d}", _box(i, r, WALL_H, WALL_C, "Wall"))
            for i, (x0, y0, x1, y1) in enumerate(hidden):
                setattr(self.scene, f"collider_{i:04d}", AssetBaseCfg(
                    prim_path=f"/World/Colliders/C_{i:04d}",
                    spawn=sim_utils.CuboidCfg(size=(x1 - x0, y1 - y0, 2.0), visible=False,
                                              collision_props=sim_utils.CollisionPropertiesCfg()),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=((x0 + x1) / 2, (y0 + y1) / 2, 1.0))))
            for i, r in enumerate(furn):
                setattr(self.scene, f"furn_{i:03d}", _box(i, r, FURNITURE_H, FURN_C, "Furniture"))
            for i, p in enumerate(people_pos):
                cfg, _ = _person(i, people_colors[i], p)
                setattr(self.scene, f"person_{i:02d}", cfg)
            if video:
                # K1 head camera: same horizontal FOV as Stage C; looks along the head link's +x
                f = 10.0
                ap = 2 * f * math.tan(math.radians(fov_deg) / 2)
                self.scene.head_cam = CameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + HEAD_LINK + "/head_cam", update_period=0.0,
                    height=height, width=width, data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=f, horizontal_aperture=ap,
                                                     clipping_range=(0.05, 40.0)),
                    offset=CameraCfg.OffsetCfg(pos=HEAD_CAMERA_OFFSET, rot=(1.0, 0.0, 0.0, 0.0), convention="world"))
                self.scene.chase_cam = CameraCfg(
                    prim_path="/World/ChaseCam", update_period=0.0, height=height, width=width, data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955,
                                                     clipping_range=(0.05, 60.0)))

    return FollowEnvCfg()
