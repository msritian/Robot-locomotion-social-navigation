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


FLOOR_RGB = (0.78, 0.76, 0.72)
WALL_RGB = (0.92, 0.91, 0.88)
FURNITURE_RGB = [(0.55, 0.40, 0.28), (0.35, 0.38, 0.42), (0.62, 0.55, 0.45), (0.28, 0.42, 0.55), (0.70, 0.66, 0.60),
                 (0.45, 0.30, 0.25)]


def _drop_ceiling(prim, z_max=2.2):
    """Remove faces whose vertices are all above z_max (ceiling): keeps the interior lit and the chase camera free."""
    import numpy as np
    from pxr import UsdGeom, Vt
    m = UsdGeom.Mesh(prim)
    pts = np.asarray(m.GetPointsAttr().Get())
    cnt = np.asarray(m.GetFaceVertexCountsAttr().Get())
    idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
    if pts.size == 0 or cnt.size == 0:
        return
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    keep_c, keep_i = [], []
    for st, c in zip(starts, cnt):
        f = idx[st:st + c]
        if np.all(pts[f, 2] > z_max):
            continue
        keep_c.append(c)
        keep_i.extend(f.tolist())
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray([int(c) for c in keep_c]))
    m.GetFaceVertexIndicesAttr().Set(Vt.IntArray([int(i) for i in keep_i]))


def style_collision_meshes(stage, root):
    """Make the (invisible) SAGE-3D collision meshes visible with simple materials: floor plan = walls/floor colour,
    every other object a muted furniture colour. Returns the number of meshes made visible."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    def material(name, rgb):
        path = Sdf.Path(f"/World/Looks/{name}")
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    wall = material("igs_wall", WALL_RGB)
    furn = [material(f"igs_furn_{i}", c) for i, c in enumerate(FURNITURE_RGB)]
    root_prim = stage.GetPrimAtPath(root)
    if not root_prim.IsValid():
        return 0
    stage.Load(root_prim.GetPath())                      # payloads may be unloaded
    n = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(root) or prim.GetTypeName() != "Mesh":
            continue
        img = UsdGeom.Imageable(prim)
        img.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        img.GetPurposeAttr().Set(UsdGeom.Tokens.default_)
        name = prim.GetName().lower() + prim.GetParent().GetName().lower()
        is_shell = "floorplan" in name or "wall" in name or "window" in name
        if is_shell:
            _drop_ceiling(prim)
        m = wall if is_shell else furn[sum(prim.GetName().encode()) % len(furn)]
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(m, UsdShade.Tokens.strongerThanDescendants)
        n += 1
    # parents may be invisible too
    p = root_prim
    while p.IsValid() and str(p.GetPath()) != "/":
        if p.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(p).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        p = p.GetParent()
    return n
