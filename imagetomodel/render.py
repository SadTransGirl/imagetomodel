"""Pictures of a model, drawn here with no 3D engine: an orthographic camera, the sprite as
the texture with no smoothing, a depth buffer. The GBA's own view (`gba`) draws a point at
(x, z - y), so a faithful model gives the sprite back; `faithful` scores that."""
import math

import numpy as np
from PIL import Image

from .geometry import normal

VIEWS = {
    # yaw (degrees round the up axis, 0 from the south), pitch (degrees looking down)
    "front": (0.0, 0.0),
    "three-quarter": (35.0, 30.0),
    "side": (90.0, 10.0),
    "back": (180.0, 25.0),
    "top": (0.0, 89.0),
}


def _camera(view):
    if view == "gba":
        return None
    yaw, pitch = VIEWS[view] if isinstance(view, str) else view
    a, b = math.radians(yaw), math.radians(pitch)
    # (the world turned so the camera looks along -z: yaw about y, then pitch about x)
    ry = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return rx @ ry


def _screen(points, cam):
    """Points to (u, v, depth): u right, v down, depth nearer the camera the bigger."""
    p = np.asarray(points, dtype=float)
    if cam is None:
        return np.stack([p[:, 0], p[:, 2] - p[:, 1], p[:, 1] + p[:, 2]], axis=1)
    q = p @ cam.T
    return np.stack([q[:, 0], -q[:, 1], q[:, 2]], axis=1)


def render(mesh, texture, view="three-quarter", scale=4, shade=None, frame=None, margin=2, background=(0, 0, 0, 0)):
    """The model from a view, `scale` pixels a pixel of the model. `texture`: the sprite's
    rgba array. `frame`: (u0, v0, u1, v1) of the model's pixels to draw (the sprite's own
    frame for the GBA view); else the model's extent plus a margin. `shade`: faces lit a
    little by their slant (on for every view but the GBA's)."""
    tex = np.asarray(texture)
    th, tw = tex.shape[:2]
    cam = _camera(view)
    if shade is None:
        shade = view != "gba"
    tris = list(mesh.triangles())
    if not tris:
        return Image.new("RGBA", (1, 1), background)
    all_pts = np.array([p for _, pts, _ in tris for p in pts])
    scr = _screen(all_pts, cam)
    if frame is None:
        u0, v0 = math.floor(scr[:, 0].min()) - margin, math.floor(scr[:, 1].min()) - margin
        u1, v1 = math.ceil(scr[:, 0].max()) + margin, math.ceil(scr[:, 1].max()) + margin
    else:
        u0, v0, u1, v1 = frame
    width, height = max(1, int(round((u1 - u0) * scale))), max(1, int(round((v1 - v0) * scale)))
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[:, :] = background
    depth = np.full((height, width), -np.inf)
    light = np.array([-0.4, 0.8, 0.45])
    light /= np.linalg.norm(light)
    for k, (name, pts, tex_uv) in enumerate(tris):
        s = scr[3 * k:3 * k + 3]
        px = (s[:, 0] - u0) * scale
        py = (s[:, 1] - v0) * scale
        x_min, x_max = int(math.floor(px.min())), int(math.ceil(px.max()))
        y_min, y_max = int(math.floor(py.min())), int(math.ceil(py.max()))
        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, width - 1), min(y_max, height - 1)
        if x_min > x_max or y_min > y_max:
            continue
        area = (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0])
        if abs(area) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x_min, x_max + 1) + 0.5, np.arange(y_min, y_max + 1) + 0.5)
        w0 = ((px[1] - gx) * (py[2] - gy) - (px[2] - gx) * (py[1] - gy)) / area
        w1 = ((px[2] - gx) * (py[0] - gy) - (px[0] - gx) * (py[2] - gy)) / area
        w2 = 1.0 - w0 - w1
        eps = -1e-6
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            continue
        d = w0 * s[0, 2] + w1 * s[1, 2] + w2 * s[2, 2]
        u = w0 * tex_uv[0][0] + w1 * tex_uv[1][0] + w2 * tex_uv[2][0]
        v = w0 * tex_uv[0][1] + w1 * tex_uv[1][1] + w2 * tex_uv[2][1]
        # (a face's corner on a pixel's edge samples the pixel inside the face)
        cu = (tex_uv[0][0] + tex_uv[1][0] + tex_uv[2][0]) / 3.0
        cv = (tex_uv[0][1] + tex_uv[1][1] + tex_uv[2][1]) / 3.0
        ui = np.clip(np.floor(u + (cu - u) * 1e-4), 0, tw - 1).astype(int)
        vi = np.clip(np.floor(v + (cv - v) * 1e-4), 0, th - 1).astype(int)
        colour = tex[vi, ui]
        opaque = colour[:, :, 3] > 127
        sub = depth[y_min:y_max + 1, x_min:x_max + 1]
        win = inside & opaque & (d > sub)
        if not win.any():
            continue
        rgb = colour[:, :, :3].astype(float)
        if shade:
            n = np.array(normal(pts))
            f = 0.72 + 0.28 * max(0.0, float(n @ light)) + 0.08 * max(0.0, float(-(n @ light)))
            rgb = np.clip(rgb * f, 0, 255)
        sub[win] = d[win]
        block = out[y_min:y_max + 1, x_min:x_max + 1]
        block[win, :3] = rgb[win].astype(np.uint8)
        block[win, 3] = 255
    return Image.fromarray(out, "RGBA")


def faithful(mesh, sprite):
    """How faithful the model is: its GBA view against the sprite. The share of the sprite's
    drawn pixels drawn back exactly, the share drawn at all, the mean colour distance (0 to
    1) where both are drawn, and a sheet: the sprite, the view, what differs (red)."""
    view = np.array(render(mesh, sprite.rgba, "gba", scale=1, frame=(0, 0, sprite.W, sprite.H), shade=False))
    want = sprite.drawn
    got = view[:, :, 3] > 127
    both = want & got
    same = both & (sprite.rgba[:, :, :3] == view[:, :, :3]).all(axis=2)
    n = int(want.sum())
    dist = np.abs(sprite.rgba[:, :, :3].astype(float) - view[:, :, :3].astype(float)).sum(axis=2) / (3 * 255.0)
    score = {
        "exact": float(same.sum()) / n if n else 1.0,
        "covered": float(both.sum()) / n if n else 1.0,
        "extra": int((got & ~want).sum()),
        "colour_distance": float(dist[both].mean()) if both.any() else 0.0,
        "pixels": n,
    }
    diff = np.zeros_like(view)
    diff[want | got] = (40, 40, 40, 255)
    diff[(want | got) & ~same] = (230, 40, 40, 255)
    sheet = Image.new("RGBA", (sprite.W * 3 + 8, sprite.H), (24, 24, 32, 255))
    sheet.paste(sprite.image(), (0, 0))
    sheet.paste(Image.fromarray(view, "RGBA"), (sprite.W + 4, 0), Image.fromarray(view, "RGBA"))
    sheet.paste(Image.fromarray(diff, "RGBA"), (2 * sprite.W + 8, 0))
    return score, sheet
