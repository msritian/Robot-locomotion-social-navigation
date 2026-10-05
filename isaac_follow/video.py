"""Video helpers for Part D (PROJECT_SPEC 13.6): chase-camera placement, head-camera overlays, top-down view,
and the side-by-side composite with a title bar (scenario, method, live distance / tracking state)."""
from __future__ import annotations

import os
import subprocess

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pf.perception.demo import odom_to_true
from pf.world.render import OTHER_C, TARGET_C, TopDownRenderer


def chase_eye_target(robot_pose, eye_s=None, back=3.0, up=1.5, alpha=0.08):
    """Third-person chase camera 3 m behind and 1.5 m above the K1, exponentially smoothed."""
    x, y, yaw = robot_pose
    eye = np.array([x - back * np.cos(yaw), y - back * np.sin(yaw), up])
    eye_s = eye if eye_s is None else (1 - alpha) * eye_s + alpha * eye
    tgt = [x + 1.0 * np.cos(yaw), y + 1.0 * np.sin(yaw), 0.6]
    return eye_s.tolist(), tgt, eye_s


def _font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


class HeadOverlay:
    """Draw tracked people (target box red, others grey) and the brain state on the K1 head-camera image.
    Projection uses the same pinhole model (HFOV) as the head camera; tracks come from the robot's own
    perception (robot frame), so overlay errors show perception noise."""

    def __init__(self, fov_deg):
        self.fov = np.deg2rad(fov_deg)
        self.font = _font(22)
        self.big = _font(40)

    def draw(self, rgb, obs, brain, odom, target_gt, cam_z):
        img = Image.fromarray(np.ascontiguousarray(rgb).astype(np.uint8))
        W, H = img.size
        f = (W / 2) / np.tan(self.fov / 2)
        d = ImageDraw.Draw(img)
        sel_id = brain.selected["track_id"] if brain.selected is not None else None
        for t in obs["tracks"]:
            x, y = t["pos_robot"]
            if x < 0.3:
                continue
            u = W / 2 - f * y / x
            v_top = H / 2 - f * (1.67 - cam_z) / x
            v_bot = H / 2 - f * (0.0 - cam_z) / x
            half = f * 0.25 / x
            is_sel = t["track_id"] == sel_id
            col = (230, 30, 30) if is_sel else (200, 200, 200)
            d.rectangle([u - half, v_top, u + half, v_bot], outline=col, width=4 if is_sel else 2)
            d.text((u - half, v_top - 26), ("TARGET " if is_sel else "") + f"id {t['track_id']}", fill=col, font=self.font)
        if brain.state.startswith("search") or brain.state == "coast":
            msg = "TARGET LOST - SEARCHING" if brain.state.startswith("search") else "TARGET OCCLUDED"
            d.rectangle([0, H - 70, W, H], fill=(0, 0, 0))
            d.text((20, H - 60), msg + f"  ({brain.state})", fill=(255, 210, 0), font=self.big)
        d.text((12, 10), "K1 head camera", fill=(255, 255, 255), font=self.font)
        return np.asarray(img)


class TopView:
    def __init__(self, world_map, cfg, size=(1280, 720)):
        self.r = TopDownRenderer(world_map, cfg["robot"]["fov_deg"], cfg["robot"]["cam_range"][1])
        self.size = size

    def draw(self, w, brain):
        colors = [TARGET_C if t else OTHER_C for t in w.people.is_target]
        for i in getattr(w, "lookalike_ids", []):
            colors[i] = "#ff9896"
        ov = []
        odom, pose = w.robot.odom, w.robot.pose
        if brain.forecast is not None and brain.state == "follow":
            m = brain.forecast.best
            pts = np.array([odom_to_true(p, odom, pose) for p in brain.forecast.modes[m]])
            ov.append(("line", pts, dict(color="#9467bd", lw=3)))
        if brain.goal is not None:
            ov.append(("points", odom_to_true(np.asarray(brain.goal), odom, pose)[None], dict(marker="*", c="#2ca02c", s=150)))
        fr = self.r.draw(pose, w.robot.camera_pose(), w.people.pos, w.people.heading, w.people.head, colors,
                         text="top-down: red = target, purple = predicted target path, green star = robot goal",
                         overlays=ov)
        return np.asarray(Image.fromarray(fr).resize(self.size))


def compose(frames, log, fps, out_prefix, title):
    """Write per-view MP4s and a side-by-side composite (chase | head | top) with a title bar, 30 fps."""
    import imageio_ffmpeg
    n = min(len(v) for v in frames.values())
    if n == 0:
        return
    font = _font(28)
    small = _font(22)
    ti = log["target_index"]
    paths = {}
    for name, fr in frames.items():
        p = f"{out_prefix}_{name}.mp4"
        imageio.mimsave(p, [np.asarray(x, dtype=np.uint8) for x in fr[:n]], fps=fps, codec="libx264", quality=8,
                        macro_block_size=8)
        paths[name] = p
    # composite: scale each view to 640x360, stack horizontally, add a 70 px title bar with live metrics
    comp = []
    n_brain = len(log["robot"])
    for k in range(n):
        tiles = [np.asarray(Image.fromarray(np.asarray(frames[v][k], dtype=np.uint8)).resize((640, 360)))
                 for v in ("chase", "head", "top")]
        row = np.concatenate(tiles, axis=1)
        bar = Image.new("RGB", (row.shape[1], 70), (18, 18, 20))
        d = ImageDraw.Draw(bar)
        kb = min(n_brain - 1, int(k * n_brain / n))
        dist = float(np.hypot(*(log["target"][kb] - log["robot"][kb][:2])))
        st = str(log["state"][kb])
        sel = int(log["sel_gt"][kb])
        track = "following TARGET" if sel == ti else ("WRONG PERSON" if sel >= 0 else st)
        d.text((14, 8), title, fill=(255, 255, 255), font=font)
        d.text((14, 42), f"t={kb * 0.1:5.1f}s   distance to target {dist:4.2f} m   {track}", fill=(255, 210, 0), font=small)
        comp.append(np.concatenate([np.asarray(bar), row], axis=0))
    tmp = f"{out_prefix}_composite_raw.mp4"
    imageio.mimsave(tmp, comp, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", tmp, "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    f"{out_prefix}.mp4"], check=False)
    if os.path.exists(f"{out_prefix}.mp4"):
        os.remove(tmp)
    paths["composite"] = f"{out_prefix}.mp4"
    print("videos:", paths)
    return paths


class StreamingVideo:
    """Writes the three views and the composite frame-by-frame (constant memory). Replaces buffering all frames,
    which exceeded 32 GB for 75 s x 3 views at 1280x720."""

    def __init__(self, out_prefix, fps, title):
        self.prefix, self.fps, self.title = out_prefix, fps, title
        kw = dict(fps=fps, codec="libx264", quality=8, macro_block_size=8)
        self.w = {v: imageio.get_writer(f"{out_prefix}_{v}.mp4", **kw) for v in ("chase", "head", "top")}
        self.comp = imageio.get_writer(f"{out_prefix}_composite_raw.mp4", **kw)
        self.font, self.small = _font(28), _font(22)
        self.n = 0

    def add(self, chase, head, top, t, dist, status):
        views = {"chase": chase, "head": head, "top": top}
        for v, img in views.items():
            self.w[v].append_data(np.asarray(img, dtype=np.uint8))
        tiles = [np.asarray(Image.fromarray(np.asarray(views[v], dtype=np.uint8)).resize((640, 360)))
                 for v in ("chase", "head", "top")]
        row = np.concatenate(tiles, axis=1)
        bar = Image.new("RGB", (row.shape[1], 70), (18, 18, 20))
        d = ImageDraw.Draw(bar)
        d.text((14, 8), self.title, fill=(255, 255, 255), font=self.font)
        d.text((14, 42), f"t={t:5.1f}s   distance to target {dist:4.2f} m   {status}", fill=(255, 210, 0), font=self.small)
        self.comp.append_data(np.concatenate([np.asarray(bar), row], axis=0))
        self.n += 1

    def close(self):
        import imageio_ffmpeg
        for wr in list(self.w.values()) + [self.comp]:
            wr.close()
        raw = f"{self.prefix}_composite_raw.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", raw, "-r", "30",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{self.prefix}.mp4"], check=False)
        if os.path.exists(f"{self.prefix}.mp4"):
            os.remove(raw)
        print("videos:", {v: f"{self.prefix}_{v}.mp4" for v in self.w} | {"composite": f"{self.prefix}.mp4"},
              f"frames={self.n}", flush=True)
