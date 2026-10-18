"""The model as LGFR3D keeps one (godot/models/*.json, a `mesh` piece as the modelling kit
makes them): its triangles in pixels, wearing a canvas with the block's picture at its left
and the thing alone at its right. Nothing of the game's pictures need go into it: with
`paint_all`, every pixel of the sprite is painted on the canvas, so the model wears the
sprite whatever the game draws there (new art); without it, the model wears the game's own
picture of its tiles (the sprite is the game's art), as the port's building modeller does.

Where it stands is said by `prints`, the fingerprints of its tiles' pictures, which only
the game knows (its catalogue: main.gd --catalog). Given the catalogue and the thing's id,
they're copied in; else they're left as 1, a print no picture has, so the model stands
nowhere until they're filled in (0 would stand everywhere)."""
import json
import os

PX = 16
UNPLACED = 1


def catalog_thing(catalog_dir, thing_id):
    """A thing's entry from a catalogue folder (its tiles, prints and size)."""
    cat = json.load(open(os.path.join(catalog_dir, "catalog.json")))
    thing = cat["things"].get(thing_id)
    if thing is None:
        raise KeyError("%s isn't in the catalogue %s" % (thing_id, catalog_dir))
    return thing


def model(mesh, sprite, name, measured=None, thing=None, thing_id=None, paint_all=True, note=None):
    cols, rows = sprite.cols, sprite.rows
    W, H = cols * PX, rows * PX
    tris, uvs = [], []
    for _, pts, tex in mesh.triangles():
        for p in pts:
            tris.extend(round(v, 3) for v in p)
        for t in tex:
            uvs.extend((round(W + t[0], 3), round(t[1], 3)))
    cut = []
    paint = {}
    for y in range(sprite.H):
        for x in range(sprite.W):
            c = sprite.colour(x, y)
            if c is None:
                cut.extend((W + x, y))
            elif paint_all:
                paint.setdefault("%02x%02x%02x" % c, []).extend((W + x, y))
    if thing is not None:
        tiles = [[int(t) for t in r] for r in thing["tiles"]]
        prints = [[int(q) for q in r] for r in thing["prints"]]
    else:
        tiles = [[-1] * cols for _ in range(rows)]
        prints = [[UNPLACED] * cols for _ in range(rows)]
    out = {"name": name, "size": [cols, rows], "tiles": tiles, "prints": prints, "priority": 1,
           "pieces": [{"kind": "mesh", "canvas": [2 * W, H], "whole_at": [0, 0], "object_at": [W, 0],
                       "cut": cut, "tris": tris, "uvs": uvs, "paint": paint}],
           "imagetomodel": {"paint_all": bool(paint_all), "sprite": [sprite.W, sprite.H], "scale": sprite.scale}}
    if thing_id:
        out["thing"] = thing_id
    if measured is not None:
        out["measured"] = measured
    if thing is None:
        out["note"] = ("prints are %d (unplaced): fill in the tiles' fingerprints from the game's catalogue "
                       "(imagetomodel ... --catalog DIR --thing ID), or import the .glb with tools/kit.py" % UNPLACED)
    if note:
        out["note"] = (out.get("note", "") + " " + note).strip()
    return out


def write(path, model_dict):
    with open(path, "w") as f:
        json.dump(model_dict, f, separators=(",", ":"))
