"""Assemble an InteriorGS / SAGE-3D scene for Isaac Sim (PROJECT_SPEC 19.2 "Use 2"):
3D Gaussian splat (USDZ, appearance) + collision mesh (USD, physics), aligned to our Stage C map frame.

Alignment (same as SAGE-3D's Data/template.usda): the splat reference gets rotateX = -90 (y-up -> z-up), the
collision payload gets a 180 deg rotation about z. pf.world.interiorgs builds the 2D map from the collision mesh
AFTER that 180 deg rotation and shifts it by `frame = (lo_x, lo_y)`, so here both are translated by -frame
to make Isaac world coordinates equal to Stage C map coordinates.
"""
from __future__ import annotations

from pathlib import Path

TEMPLATE = """#usda 1.0
(
    defaultPrim = "Interior"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Interior"
{{
    double3 xformOp:translate = ({tx}, {ty}, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Xform "gauss" (
        prepend references = @{usdz}[gauss.usda]@
    )
    {{
        double3 xformOp:rotateXYZ = (-90, 0, 0)
        double3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }}

    def Xform "scene_collision" (
        prepend payload = @{collision}@
    )
    {{
        quatf xformOp:orient = (6.123234e-17, 0, 0, 1)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}
}}
"""


def write_scene_usda(out_path, usdz_path, collision_path, frame):
    """frame: (lo_x, lo_y) from Map.params['frame']."""
    txt = TEMPLATE.format(tx=-float(frame[0]), ty=-float(frame[1]), usdz=Path(usdz_path).resolve(),
                          collision=Path(collision_path).resolve())
    Path(out_path).write_text(txt)
    return str(out_path)
