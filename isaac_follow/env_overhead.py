"""Alignment check: render an Isaac environment from straight above (narrow FOV, ~orthographic) placed exactly as in
run_follow (translate = -origin, -floor_z), so the image can be overlaid on our sliced 2D map.

    python isaac_follow/env_overhead.py --name hospital --env_maps isaac_follow/env_maps --out overhead --headless
"""
import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--env_maps", default="isaac_follow/env_maps")
ap.add_argument("--out", default="overhead")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from isaac_follow.characters import add_visual_environment  # noqa: E402

os.makedirs(args.out, exist_ok=True)
d = np.load(os.path.join(args.env_maps, f"{args.name}_map.npz"), allow_pickle=True)
occ, res, origin, floor_z, url = d["occ"], float(d["res"]), d["origin"], float(d["floor_z"]), str(d["url"])
W, H = occ.shape[1] * res, occ.shape[0] * res
sim = SimulationContext(SimulationCfg(dt=0.01, device="cuda:0"))
stage = omni.usd.get_context().get_stage()
add_visual_environment(stage, url, (-float(origin[0]), -float(origin[1]), -floor_z))
sim_utils.spawn_light("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
height = 200.0
hfov = 2 * math.degrees(math.atan(0.55 * max(W, H) / height))
f = 50.0
ap_mm = 2 * f * math.tan(math.radians(hfov) / 2)
cam = Camera(CameraCfg(prim_path="/World/Overhead", update_period=0, height=1600, width=1600, data_types=["rgb"],
                       spawn=sim_utils.PinholeCameraCfg(focal_length=f, horizontal_aperture=ap_mm,
                                                        clipping_range=(1.0, 1000.0))))
sim.reset()
cx, cy = W / 2, H / 2
cam.set_world_poses_from_view(torch.tensor([[cx, cy, height]], device="cuda:0"),
                              torch.tensor([[cx, cy + 0.001, 0.0]], device="cuda:0"))
for _ in range(60):
    sim.step()
    cam.update(0.01)
img = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
imageio.imwrite(os.path.join(args.out, f"{args.name}_overhead.png"), img)
json.dump({"center": [cx, cy], "height": height, "hfov_deg": hfov, "map_w": W, "map_h": H, "px": 1600},
          open(os.path.join(args.out, f"{args.name}_overhead.json"), "w"))
print(f"[overhead] {args.name}: map {W:.1f}x{H:.1f} m, hfov {hfov:.1f} deg, mean {img.mean():.1f}", flush=True)
import sys  # noqa: E402
sys.stdout.flush()
os._exit(0)
