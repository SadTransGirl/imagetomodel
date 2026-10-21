"""A thing that does something, in frames of cubes: each frame a voxel.Grid and how long it
shows. Written as an animated GIF (to look at), an animated .vox (MagicaVoxel's frames)
and a .glb whose animation shows each frame's node in turn (Godot makes an AnimationPlayer
of it).

The healing machine's: Poke Balls set on its slots one by one, flashing while the jingle
plays, then taken away. Its fix says how ("animations" in fixes/rooms.json):

    "animations": {"heal": {"slots": {"colour": "6bc66b"}, "balls": 6,
                            "ball": ["f04030", "f8f8f8"], "glow": "fff8c8",
                            "place": 0.4, "flashes": 6, "flash": 0.15, "hold": 0.6}}
"""
import struct
from collections import Counter

from . import voxel


def hex_colour(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def slots(grid, colour):
    """The slots on a thing: the patches of cubes of `colour` at the top of where they
    stand (a tray's ball rests), each [(x, y, z)], from the back row to the front, left to
    right."""
    tops = {}
    for (x, y, z), c in grid.cubes.items():
        if c == colour and (x, y + 1, z) not in grid.cubes:
            tops[(x, z)] = max(tops.get((x, z), -1), y)
    left = set(tops)
    out = []
    while left:
        start = left.pop()
        patch = [start]
        todo = [start]
        while todo:
            x, z = todo.pop()
            for q in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
                if q in left:
                    left.discard(q)
                    patch.append(q)
                    todo.append(q)
        out.append([(x, tops[(x, z)], z) for x, z in patch])
    out.sort(key=lambda p: (min(z for _, _, z in p), min(x for x, _, _ in p)))
    return out


def copy(grid):
    out = voxel.Grid()
    out.cubes = dict(grid.cubes)
    out.owner = dict(grid.owner)
    return out


def heal(grid, spec):
    """The healing machine's frames, [(grid, seconds)]: idle, a ball set on each slot in
    turn, the balls flashing, held, then taken away."""
    owner = next(iter(grid.owner.values()), "")
    slot_colour = hex_colour(spec.get("slots", {}).get("colour", "6bc66b"))
    top, bottom = [hex_colour(h) for h in spec.get("ball", ["f04030", "f8f8f8"])]
    glow = hex_colour(spec.get("glow", "fff8c8"))
    place = float(spec.get("place", 0.4))
    flash = float(spec.get("flash", 0.15))
    found = slots(grid, slot_colour)[:int(spec.get("balls", 6))]
    frames = [(copy(grid), place)]
    now = copy(grid)
    balls = []
    for patch in found:
        # (a ball: the slot's cubes its white bottom, its red top a cube over them)
        ball = []
        for (x, y, z) in patch:
            now.set(x, y, z, bottom, owner)
            now.set(x, y + 1, z, top, owner)
            ball += [((x, y, z), bottom), ((x, y + 1, z), top)]
        balls.append(ball)
        frames.append((copy(now), place))
    lit = copy(now)
    for ball in balls:
        for key, _ in ball:
            lit.set(*key, glow, owner)
    for _ in range(int(spec.get("flashes", 6))):
        frames.append((copy(lit), flash))
        frames.append((copy(now), flash))
    frames.append((copy(now), float(spec.get("hold", 0.6))))
    frames.append((copy(grid), place))
    return frames


def repaint(grid, changed):
    """The thing with its picture changed where `changed` says ({pixel: colour}, in the
    thing's own pixels): each cube the GBA's view sees at a pixel wearing its new colour."""
    out = copy(grid)
    view = voxel.gba_view(set(grid.cubes))
    for p, c in changed.items():
        key = view.get(p)
        if key is not None:
            out.cubes[key] = tuple(c)
    return out


def _mid(m):
    return int(m, 16) if isinstance(m, str) else int(m)


def flicker(grid, spec, ctx):
    """A PC turning on, as the game does it (field_specials.c, AnimatePcTurnOn): its screen
    on, off, on, off, on, `flash` seconds each (6 of the GBA's frames), then on while it's
    used (`hold`), then off. Its metatiles `off` and `on` (the Building tileset's 0x062,
    0x063) from the game's data."""
    off, on = _mid(spec.get("off", "0x062")), _mid(spec.get("on", "0x063"))
    lit = repaint(grid, ctx["changed"]({cell: on for cell, mid in ctx["cells"].items() if mid == off}))
    flash = float(spec.get("flash", 0.1))
    frames = [(copy(grid), float(spec.get("idle", 0.6)))]
    for k in range(5):
        frames.append((copy(lit if k % 2 == 0 else grid), flash))
    frames[-1] = (frames[-1][0], float(spec.get("hold", 1.2)))
    return frames


def cycle(grid, spec, ctx):
    """An escalator running, as the game runs it (special_field_anim.c): each of its
    `sections`' metatiles (the game's tables: sEscalatorMetatiles_*) moved on a stage in turn,
    every `step` seconds (8 of the GBA's frames), up or down (`up`); from when it starts,
    looped from the first stage that comes round again."""
    tables = [[_mid(m) for m in t] for t in spec["sections"]]
    up, last = bool(spec.get("up", False)), len(tables[0]) - 1
    state = {cell: mid for cell, mid in ctx["cells"].items() if any(mid in t for t in tables)}
    stage, seen, frames = 0, [], []
    for _ in range(64):
        for t in tables:
            for cell, mid in list(state.items()):
                if not up and mid == t[stage]:
                    state[cell] = t[stage + 1] if stage != last else t[0]
                elif up and mid == t[last - stage]:
                    state[cell] = t[1 - stage] if stage != last else t[last]
        stage = (stage + 1) % (last + 1)
        key = (stage, tuple(sorted(state.items())))
        if key in seen:
            frames = frames[seen.index(key):]
            break
        seen.append(key)
        frames.append(dict(state))
    step = float(spec.get("step", 8 / 60.0))
    changes = [ctx["changed"](f) for f in frames]
    # (only what moves: its steps' colours trading places (`colours`, else the commonest
    # pair that does and what trades with them often); the rest of its picture's changes
    # (its side's panel sliding with them) are the GBA faking a slope a 3D one has)
    pairs = Counter((tuple(ctx["old"](p)), tuple(c)) for ch in changes for p, c in ch.items())
    if "colours" in spec:
        moving = {tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in spec["colours"]}
    elif pairs:
        (a, b), _ = pairs.most_common(1)[0]
        moving = {a, b}
        total = sum(pairs.values())
        for (x, y), n in pairs.items():
            if n >= 0.05 * total and (x in (a, b) or y in (a, b)):
                moving |= {x, y}
    else:
        moving = set()
    return [(repaint(grid, {p: c for p, c in ch.items() if tuple(c) in moving and tuple(ctx["old"](p)) in moving}), step)
            for ch in changes]


KINDS = {"heal": heal, "flicker": flicker, "cycle": cycle}
NEEDS_GAME = {"flicker", "cycle"}                       # (the game's data: its metatiles)


def frames_of(grid, kind, spec, ctx=None):
    """A thing's frames for an animation of its fix ("animations": {name: spec}), by its
    "kind" (or its name): heal; flicker and cycle (from the game's metatiles, `ctx`: its
    map's cells and how a picture of them with some changed differs)."""
    kind = spec.get("kind", kind)
    if kind in NEEDS_GAME:
        return KINDS[kind](grid, spec, ctx)
    return KINDS[kind](grid, spec)


def write_gif(path, frames, texture, view=(-30.0, 35.0), scale=10):
    """The frames rendered from `view`, each as long as it shows, looping."""
    from PIL import Image
    from .render import render
    meshes = [voxel.mesh_of(g, strip_v=texture.shape[0], strip_w=texture.shape[1]) for g, _ in frames]
    # (one frame for all: the extent of every frame, so the picture holds still)
    whole = voxel.Grid()
    for g, _ in frames:
        whole.cubes.update(g.cubes)
        whole.owner.update(g.owner)
    ref = render(voxel.mesh_of(whole, strip_v=texture.shape[0], strip_w=texture.shape[1]), texture, view, scale=scale)
    pics = []
    for m in meshes:
        im = render(m, texture, view, scale=scale, background=(255, 255, 255, 255))
        canvas = Image.new("RGB", ref.size, (255, 255, 255))
        canvas.paste(im.convert("RGB").crop((0, 0) + ref.size), (0, 0))
        pics.append(canvas)
    pics[0].save(path, save_all=True, append_images=pics[1:], duration=[int(d * 1000) for _, d in frames], loop=0)


def _vox_string(s):
    b = s.encode()
    return struct.pack("<i", len(b)) + b


def _vox_dict(d):
    out = struct.pack("<i", len(d))
    for k, v in d.items():
        out += _vox_string(k) + _vox_string(str(v))
    return out


def write_vox_frames(path, frames, fps=10):
    """An animated .vox: a model a frame (MagicaVoxel's animation, each shape's model keyed
    by its frame, "_f"), each frame held for as many of `fps` frames as it shows; all of
    one size, so the frames line up."""
    whole = voxel.Grid()
    for g, _ in frames:
        whole.cubes.update(g.cubes)
    (x0, y0, z0), (x1, y1, z1) = whole.bounds()
    sx, sy, sz = x1 - x0 + 1, z1 - z0 + 1, y1 - y0 + 1
    if max(sx, sy, sz) > 256:
        raise ValueError("%dx%dx%d cubes: MagicaVoxel holds 256 a side" % (sx, sy, sz))
    colours = []
    for g, _ in frames:
        for c in g.cubes.values():
            if c not in colours:
                colours.append(c)
    palette = colours[:255]
    index = {c: i + 1 for i, c in enumerate(palette)}

    def nearest(c):
        if c not in index:
            best = min(palette, key=lambda p: sum((p[k] - c[k]) ** 2 for k in range(3)))
            index[c] = index[best]
        return index[c]

    def chunk(tag, content, children=b""):
        return tag + struct.pack("<ii", len(content), len(children)) + content + children
    body = b""
    ticks = []
    t = 0
    for g, seconds in frames:
        data = bytearray()
        for (x, y, z), c in g.cubes.items():
            data += bytes((x - x0, z1 - z, y - y0, nearest(c)))
        body += chunk(b"SIZE", struct.pack("<iii", sx, sy, sz))
        body += chunk(b"XYZI", struct.pack("<i", len(g.cubes)) + bytes(data))
        ticks.append(t)
        t += max(1, int(round(seconds * fps)))
    # (the scene: a transform, a group, a transform, the shape whose models are the frames)
    body += chunk(b"nTRN", struct.pack("<i", 0) + _vox_dict({}) + struct.pack("<iiii", 1, -1, -1, 1) + _vox_dict({}))
    body += chunk(b"nGRP", struct.pack("<i", 1) + _vox_dict({}) + struct.pack("<ii", 1, 2))
    body += chunk(b"nTRN", struct.pack("<i", 2) + _vox_dict({}) + struct.pack("<iiii", 3, -1, 0, 1) + _vox_dict({}))
    shp = struct.pack("<i", 3) + _vox_dict({}) + struct.pack("<i", len(frames))
    for i, tick in enumerate(ticks):
        shp += struct.pack("<i", i) + _vox_dict({"_f": tick})
    body += chunk(b"nSHP", shp)
    rgba = bytearray()
    for i in range(256):
        c = palette[i] if i < len(palette) else (0, 0, 0)
        rgba += bytes((c[0], c[1], c[2], 255 if i < len(palette) else 0))
    body += chunk(b"RGBA", bytes(rgba))
    with open(path, "wb") as f:
        f.write(b"VOX " + struct.pack("<i", 200) + chunk(b"MAIN", b"", body))


def write_glb(path, frames, sprite, name="thing"):
    """A .glb of the frames: each different frame a node of its own (the same ones shared),
    and an animation, `name`, showing each in its turn (its scale 1 while it shows, else 0,
    stepped), looping in Godot when it's set to. At rest, the first frame shows."""
    from . import gltf
    states, order = [], []
    for g, _ in frames:
        key = frozenset(g.cubes.items())
        if key not in states:
            states.append(key)
        order.append(states.index(key))
    mesh = None
    seen = set()
    for (g, _), i in zip(frames, order):
        if i in seen:
            continue
        seen.add(i)
        mesh = voxel.mesh_of(g, mesh, strip_v=sprite.H, strip_w=sprite.W, name="%s-%d" % (name, i))
    keys = {"%s-%d" % (name, i): [] for i in range(len(states))}
    t = 0.0
    for (g, seconds), i in zip(frames, order):
        for node, k in keys.items():
            shown = 1.0 if node == "%s-%d" % (name, i) else 0.0
            if not k or k[-1][1] != shown:
                k.append((t, shown))
        t += seconds
    for k in keys.values():
        if k[-1][0] < t:
            k.append((t, k[-1][1]))                     # (the last frame shown to the end)
    gltf.write_glb(path, mesh, sprite, name=name, split=True,
                   animation={"name": name, "length": t, "keys": keys})
