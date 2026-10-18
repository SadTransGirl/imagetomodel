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

PX = 16
PAINT_MIN = 32               # (the kit's: the painting strip's least height)


def canvas_size(cols, rows):
    w, h = cols * PX, rows * PX
    return 2 * w, h + max(h, PAINT_MIN)


def kit_canvas(sprite):
    """The kit's texture: the sprite at the top left (the block's picture), again at the top
    right (the thing alone), the painting strip clear below."""
    cw, ch = canvas_size(sprite.cols, sprite.rows)
    out = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    im = sprite.image()
    out.paste(im, (0, 0))
    out.paste(im, (sprite.cols * PX, 0))
    return out


def _vertices(mesh, sprite):
    """The mesh's corners as the kit keeps them, each once: positions in tiles from the
    block's middle, texture coordinates of the thing alone (the canvas's right half, 0 to 1)."""
    W, H = sprite.cols * PX, sprite.rows * PX
    cw, ch = canvas_size(sprite.cols, sprite.rows)
    pos, uvs, seen, polys = [], [], {}, []
    for name, pts, tex in mesh.polys:
        ring = []
        for p, t in zip(pts, tex):
            key = (round(p[0], 3), round(p[1], 3), round(p[2], 3), round(t[0], 3), round(t[1], 3))
            if key not in seen:
                seen[key] = len(pos)
                pos.append(((p[0] - W / 2.0) / PX, p[1] / PX, (p[2] - H / 2.0) / PX))
                uvs.append(((W + t[0]) / cw, t[1] / ch))
            ring.append(seen[key])
        polys.append((name, ring))
    return pos, uvs, polys


def write_glb(path, mesh, sprite, name="building", note=None):
    """A .glb: one object wearing the kit's canvas, drawn with no smoothing, both sides. The
    kit's import takes it back when `note` names its block ({"kind": "thing", "key": ID}),
    or by its texture's top left matching the block's picture."""
    pos, uvs, polys = _vertices(mesh, sprite)
    idx = []
    for _, ring in polys:
        for j in range(1, len(ring) - 1):
            idx += [ring[0], ring[j], ring[j + 1]]
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

    pv = add_view(struct.pack("<%df" % (len(pos) * 3), *[c for p in pos for c in p]), 34962)
    tv = add_view(struct.pack("<%df" % (len(uvs) * 2), *[c for t in uvs for c in t]), 34962)
    iv = add_view(struct.pack("<%dI" % len(idx), *idx), 34963)
    buf = io.BytesIO()
    kit_canvas(sprite).save(buf, "PNG")
    img_view = add_view(buf.getvalue())
    while len(blob) % 4:
        blob.append(0)
    tex_name = "lgfr-%s-%s" % (note["kind"], note["key"]) if note else name
    accessors = [
        {"bufferView": pv, "componentType": 5126, "count": len(pos), "type": "VEC3",
         "min": [min(p[i] for p in pos) for i in range(3)], "max": [max(p[i] for p in pos) for i in range(3)]},
        {"bufferView": tv, "componentType": 5126, "count": len(uvs), "type": "VEC2"},
        {"bufferView": iv, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
    ]
    asset = {"version": "2.0", "generator": "imagetomodel"}
    if note:
        asset["extras"] = {"lgfr": dict(note)}
    doc = {"asset": asset, "scene": 0, "scenes": [{"name": tex_name, "nodes": [0]}],
           "nodes": [{"name": name, "mesh": 0}],
           "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
           "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
           "materials": [{"name": tex_name, "doubleSided": True, "alphaMode": "MASK", "alphaCutoff": 0.5,
                          "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
           "textures": [{"source": 0, "sampler": 0}],
           "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
           "images": [{"name": tex_name, "mimeType": "image/png", "bufferView": img_view}]}
    js = json.dumps(doc, separators=(",", ":")).encode()
    while len(js) % 4:
        js += b" "
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))


def write_obj(path, mesh, sprite, name="building", note=None):
    """A .obj with its .mtl and the canvas as a .png beside it (Blender: File > Import >
    Wavefront, the defaults). Its polygons are kept whole."""
    stem = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.dirname(path) or "."
    tex_name = "lgfr-%s-%s" % (note["kind"], note["key"]) if note else name
    png = stem + ".png"
    kit_canvas(sprite).save(os.path.join(folder, png))
    with open(os.path.join(folder, stem + ".mtl"), "w") as f:
        f.write("# imagetomodel\nnewmtl %s\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n"
                "d 1.0\nillum 1\nmap_Kd %s\nmap_d %s\n" % (tex_name, png, png))
    pos, uvs, polys = _vertices(mesh, sprite)
    lines = ["# imagetomodel: 1 unit a tile; x east, y up, z south"]
    if note:
        lines.append("# lgfr " + json.dumps(note, separators=(",", ":")))
    lines += ["mtllib %s.mtl" % stem, "o " + name]
    lines += ["v %g %g %g" % tuple(round(c, 5) for c in p) for p in pos]
    # (an OBJ's v runs up the picture, from its bottom)
    lines += ["vt %g %g" % (round(t[0], 6), round(1 - t[1], 6)) for t in uvs]
    lines += ["usemtl " + tex_name, "s off"]
    for _, ring in polys:
        lines.append("f " + " ".join("%d/%d" % (i + 1, i + 1) for i in ring))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
