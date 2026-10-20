"""The model as a glTF (binary) and as a Wavefront OBJ, in the modelling kit's own
conventions (LGFR3D's tools/kit.py): a tile a unit, x east, y up, z south, the block's
middle at the origin, texture coordinates from the top left of the kit's canvas (the
sprite at its left, the thing alone at its right, a strip to paint in below). The kit's
`import` brings either back as a model of the game."""
import io
import json
import os
import struct

from PIL import Image

from .geometry import strip_pixels

PX = 16
PAINT_MIN = 32               # (the kit's: the painting strip's least height)


def canvas_size(cols, rows):
    w, h = cols * PX, rows * PX
    return 2 * w, h + max(h, PAINT_MIN)


def kit_canvas(sprite, cut=None, painted=None):
    """The kit's texture: the sprite at the top left (the block's picture), again at the top
    right (the thing alone, with `cut` pixels (x, y pairs) cleared), the strip clear below."""
    cw, ch = canvas_size(sprite.cols, sprite.rows)
    out = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    im = sprite.image()
    out.paste(im, (0, 0))
    alone = im.copy()
    if cut:
        px = alone.load()
        for k in range(0, len(cut) - 1, 2):
            x, y = int(cut[k]), int(cut[k + 1])
            if 0 <= x < alone.width and 0 <= y < alone.height:
                px[x, y] = (0, 0, 0, 0)
    out.paste(alone, (sprite.cols * PX, 0))
    px = out.load()
    for (x, y), c in (painted or {}).items():
        if sprite.cols * PX + x < cw and sprite.H + y < ch:
            px[sprite.cols * PX + x, sprite.H + y] = (*c, 255)
    return out


def _object_name(name):
    """A polygon's object: its name up to any ":" (a part's tag)."""
    return (name or "building").split(":")[0]


def _vertices(mesh, sprite, split=False):
    """The mesh's corners as the kit keeps them, each once: positions in tiles from the
    block's middle, texture coordinates of the thing alone (the canvas's right half, 0 to 1).
    Polygons grouped by object: one group, or (`split`) one per polygon name."""
    W, H = sprite.cols * PX, sprite.rows * PX
    cw, ch = canvas_size(sprite.cols, sprite.rows)
    groups = {}
    order = []
    for name, pts, tex in mesh.polys:
        obj = _object_name(name) if split else None
        if obj not in groups:
            groups[obj] = ([], [], {}, [])
            order.append(obj)
        pos, uvs, seen, polys = groups[obj]
        ring = []
        for p, t in zip(pts, tex):
            key = (round(p[0], 3), round(p[1], 3), round(p[2], 3), round(t[0], 3), round(t[1], 3))
            if key not in seen:
                seen[key] = len(pos)
                pos.append(((p[0] - W / 2.0) / PX, p[1] / PX, (p[2] - H / 2.0) / PX))
                uvs.append(((W + t[0]) / cw, t[1] / ch))
            ring.append(seen[key])
        polys.append((name, ring))
    return [(obj, groups[obj][0], groups[obj][1], groups[obj][3]) for obj in order]


def write_glb(path, mesh, sprite, name="building", note=None, split=False, cut=None, animation=None):
    """A .glb: one object wearing the kit's canvas (or, `split`, one per polygon name: the
    kit's import makes a model of each), drawn with no smoothing, both sides. The kit's
    import takes it back when `note` names its block ({"kind": "thing", "key": ID}), or by
    its texture's top left matching the block's picture. `animation`: {"name", "length",
    "keys": {object: [(seconds, 1 shown or 0 not), ...]}}, each object's scale stepped so
    (a flipbook of frames; at rest each as its first key)."""
    blob = bytearray()
    views = []

    def add_view(data, target=None):
        while len(blob) % 4:
            blob.append(0)
        v = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target:
            v["target"] = target
        blob.extend(data)
        views.append(v)
        return len(views) - 1

    accessors, meshes, nodes = [], [], []
    for obj, pos, uvs, polys in _vertices(mesh, sprite, split):
        idx = []
        for _, ring in polys:
            for j in range(1, len(ring) - 1):
                idx += [ring[0], ring[j], ring[j + 1]]
        if not idx:
            continue
        pv = add_view(struct.pack("<%df" % (len(pos) * 3), *[c for p in pos for c in p]), 34962)
        tv = add_view(struct.pack("<%df" % (len(uvs) * 2), *[c for t in uvs for c in t]), 34962)
        iv = add_view(struct.pack("<%dI" % len(idx), *idx), 34963)
        accessors += [
            {"bufferView": pv, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": [min(p[i] for p in pos) for i in range(3)], "max": [max(p[i] for p in pos) for i in range(3)]},
            {"bufferView": tv, "componentType": 5126, "count": len(uvs), "type": "VEC2"},
            {"bufferView": iv, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
        ]
        obj_name = obj or name
        meshes.append({"name": obj_name, "primitives": [{"attributes": {"POSITION": len(accessors) - 3, "TEXCOORD_0": len(accessors) - 2},
                                                          "indices": len(accessors) - 1, "material": 0}]})
        nodes.append({"name": obj_name, "mesh": len(meshes) - 1})
    anims = []
    if animation:
        channels, samplers = [], []
        for i, node in enumerate(nodes):
            k = animation["keys"].get(node["name"])
            if not k:
                continue
            if k[0][1] == 0.0:
                node["scale"] = [0.0, 0.0, 0.0]
            times = [t for t, _ in k]
            tv = add_view(struct.pack("<%df" % len(times), *times))
            vv = add_view(struct.pack("<%df" % (3 * len(k)), *[v for _, s in k for v in (s, s, s)]))
            accessors += [
                {"bufferView": tv, "componentType": 5126, "count": len(times), "type": "SCALAR",
                 "min": [min(times)], "max": [max(times)]},
                {"bufferView": vv, "componentType": 5126, "count": len(k), "type": "VEC3"},
            ]
            samplers.append({"input": len(accessors) - 2, "output": len(accessors) - 1, "interpolation": "STEP"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": i, "path": "scale"}})
        anims.append({"name": animation["name"], "channels": channels, "samplers": samplers,
                      "extras": {"length": animation["length"]}})
    buf = io.BytesIO()
    kit_canvas(sprite, cut, strip_pixels(mesh, sprite.W)).save(buf, "PNG")
    img_view = add_view(buf.getvalue())
    while len(blob) % 4:
        blob.append(0)
    tex_name = "lgfr-%s-%s" % (note["kind"], note["key"]) if note else name
    asset = {"version": "2.0", "generator": "imagetomodel"}
    if note:
        asset["extras"] = {"lgfr": dict(note)}
    doc = {"asset": asset, "scene": 0, "scenes": [{"name": tex_name, "nodes": list(range(len(nodes)))}],
           "nodes": nodes, "meshes": meshes,
           "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
           "materials": [{"name": tex_name, "doubleSided": True, "alphaMode": "MASK", "alphaCutoff": 0.5,
                          "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
           "textures": [{"source": 0, "sampler": 0}],
           "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
           "images": [{"name": tex_name, "mimeType": "image/png", "bufferView": img_view}]}
    if anims:
        doc["animations"] = anims
    js = json.dumps(doc, separators=(",", ":")).encode()
    while len(js) % 4:
        js += b" "
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))


def write_obj(path, mesh, sprite, name="building", note=None, split=False, cut=None):
    """A .obj with its .mtl and the canvas as a .png beside it (Blender: File > Import >
    Wavefront, the defaults). Its polygons are kept whole; `split`: an object per name."""
    stem = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.dirname(path) or "."
    tex_name = "lgfr-%s-%s" % (note["kind"], note["key"]) if note else name
    png = stem + ".png"
    kit_canvas(sprite, cut, strip_pixels(mesh, sprite.W)).save(os.path.join(folder, png))
    with open(os.path.join(folder, stem + ".mtl"), "w") as f:
        f.write("# imagetomodel\nnewmtl %s\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n"
                "d 1.0\nillum 1\nmap_Kd %s\nmap_d %s\n" % (tex_name, png, png))
    lines = ["# imagetomodel: 1 unit a tile; x east, y up, z south"]
    if note:
        lines.append("# lgfr " + json.dumps(note, separators=(",", ":")))
    lines.append("mtllib %s.mtl" % stem)
    nv = 0
    for obj, pos, uvs, polys in _vertices(mesh, sprite, split):
        lines.append("o " + (obj or name))
        lines += ["v %g %g %g" % tuple(round(c, 5) for c in p) for p in pos]
        # (an OBJ's v runs up the picture, from its bottom)
        lines += ["vt %g %g" % (round(t[0], 6), round(1 - t[1], 6)) for t in uvs]
        lines += ["usemtl " + tex_name, "s off"]
        for _, ring in polys:
            lines.append("f " + " ".join("%d/%d" % (nv + i + 1, nv + i + 1) for i in ring))
        nv += len(pos)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
