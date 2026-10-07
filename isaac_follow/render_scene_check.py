"""Render check for InteriorGS scenes: where do the splat and the collision meshes end up relative to our map?
Saves top-down and eye-level PNGs (a) as loaded and (b) with omni:nurec:useProxyTransform = True on the splat.

    python isaac_follow/render_scene_check.py --interiorgs 840025 --scene_dir interiorgs --out scene_check --headless
"""
import argparse
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--interiorgs", required=True)
ap.add_argument("--scene_dir", default="interiorgs")
ap.add_argument("--out", default="scene_check")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
# NuRec settings from the Isaac Sim NuRec docs (single GPU, keep tonemapping for gaussians)
args.kit_args = "--/renderer/multiGpu/enabled=false --/rtx/rtpt/gaussian/skipTonemapping/enabled=false"
app = AppLauncher(args).app

import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

import k1_walker  # noqa: E402,F401
from isaac_follow.interiorgs_scene import style_collision_meshes, write_scene_usda  # noqa: E402
from isaac_follow.scene_cfg import make_env_cfg  # noqa: E402
from pf.config import load_config  # noqa: E402
from pf.world.interiorgs import scene_to_map  # noqa: E402

os.makedirs(args.out, exist_ok=True)
cfg = load_config()
coll = os.path.join(args.scene_dir, f"{args.interiorgs}_collision.usd")
m = scene_to_map(coll, args.interiorgs, cfg)
usda = write_scene_usda(os.path.join(args.out, "scene.usda"), os.path.join(args.scene_dir, f"{args.interiorgs}.usdz"),
                        coll, m.params["frame"])
env_cfg = make_env_cfg(m, np.zeros((0, 2)), [], video=True, interior_usda=usda)
p0 = m.pois[0]
env_cfg.scene.robot.init_state.pos = (float(p0[0]), float(p0[1]), 0.57)
env_cfg.scene.top_cam = CameraCfg(prim_path="/World/TopCam", update_period=0.0, height=720, width=1280, data_types=["rgb"],
                                  spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, horizontal_aperture=20.0,
                                                                   clipping_range=(0.1, 200.0)))
env_cfg.sim.device = "cuda:0"
env = ManagerBasedRLEnv(cfg=env_cfg)
stage = omni.usd.get_context().get_stage()
print("[check] map size", m.width, m.height, "frame", m.params["frame"], "poi0", p0.tolist(), flush=True)
print("[check] collision meshes styled:", style_collision_meshes(stage, "/World/Interior/scene_collision"), flush=True)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
for path in ("/World/Interior/scene_collision", "/World/Interior/gauss"):
    pr = stage.GetPrimAtPath(path)
    if pr.IsValid():
        b = cache.ComputeWorldBound(pr).ComputeAlignedRange()
        print(f"[check] world bbox {path}: {list(b.GetMin())} .. {list(b.GetMax())}", flush=True)
vols = [p for p in stage.Traverse() if p.GetTypeName() == "Volume"]
print("[check] volumes:", [str(p.GetPath()) for p in vols], flush=True)
try:
    import isaacsim.replicator.nurec_utils as nu  # noqa: F401
    print("[check] nurec_utils available", flush=True)
except Exception as e:  # noqa: BLE001
    print("[check] nurec_utils NOT available:", repr(e)[:120], flush=True)
top, head = env.scene["top_cam"], env.scene["head_cam"]
cx, cy = m.width / 2, m.height / 2
top.set_world_poses_from_view(torch.tensor([[cx, cy, 35.0]], device=env.device),
                              torch.tensor([[cx + 0.01, cy, 0.0]], device=env.device))
act = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
obs, _ = env.reset()


def snap(tag):
    for _ in range(30):
        env.step(act)
    for name, cam in (("top", top), ("head", head)):
        img = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        imageio.imwrite(os.path.join(args.out, f"{tag}_{name}.png"), img)
        print(f"[check] {tag}_{name}: mean {img.mean():.1f} std {img.std():.1f}", flush=True)


snap("a_asloaded")
for v in vols:
    a = v.GetAttribute("omni:nurec:useProxyTransform")
    if a:
        a.Set(True)
snap("b_proxytransform")
import sys as _sys  # noqa: E402
_sys.stdout.flush()
os._exit(0)
