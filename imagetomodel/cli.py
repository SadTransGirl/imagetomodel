"""imagetomodel building SPRITE.png -o DIR: a building's 3D model from its sprite.

Writes, in DIR, named after the sprite (or --name):
  NAME.lgfr.json       the model as LGFR3D keeps one (godot/models/)
  NAME.glb, NAME.obj   the model for Blender, Blockbench, Godot: the kit's conventions
  NAME-measure.json    what was read from the sprite (a spec to correct and pass back)
  NAME-faithful.png    the sprite, the model seen from the GBA's angle, what differs
  NAME-VIEW.png        the model from other angles
and prints how faithful it is: the share of the sprite's pixels the model gives back."""
import argparse
import json
import os
import sys

from .picture import Sprite
from .measure import measure
from .building import build
from .render import render, faithful, VIEWS
from . import gltf, lgfr


def hex_colour(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="imagetomodel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("building", help="a building from its sprite")
    b.add_argument("sprite")
    b.add_argument("-o", "--out", default="out")
    b.add_argument("--name", help="the model's name (the sprite's file name)")
    b.add_argument("--scale", type=int, help="how many times the sprite was scaled up (found from its pixel blocks)")
    b.add_argument("--ground", help="colours round the sprite that aren't it, to clear (hex, comma separated)")
    b.add_argument("--spec", help="a JSON file of measures that win over what's read (see NAME-measure.json)")
    b.add_argument("--inset", type=float, default=1.0, help="how far the windows are set back (pixels)")
    b.add_argument("--door-inset", type=float, default=1.0, help="how far the door is set back (pixels)")
    b.add_argument("--no-relief", action="store_true", help="the wall flat: no door or windows set back")
    b.add_argument("--depth", type=float, help="the building's depth in tiles (the roof's picture stretched over it); else as drawn")
    b.add_argument("--catalog", help="LGFR3D's catalogue folder (main.gd --catalog), for the tiles' prints")
    b.add_argument("--thing", help="the thing's id in the catalogue the model stands for (e.g. 5x5-1a2b3c4d)")
    b.add_argument("--game-art", action="store_true", help="the sprite is the game's own picture: wear that, paint nothing")
    b.add_argument("--views", default="three-quarter,side,front", help="which other views to draw (%s)" % ",".join(VIEWS))
    b.add_argument("--preview-scale", type=int, default=4)
    a = ap.parse_args(argv)

    ground = [hex_colour(c) for c in a.ground.split(",")] if a.ground else None
    sprite = Sprite.load(a.sprite, scale=a.scale, ground=ground)
    name = a.name or os.path.splitext(os.path.basename(a.sprite))[0]
    spec = json.load(open(a.spec)) if a.spec else None
    m = measure(sprite, spec)
    mesh = build(sprite, m, inset=a.inset, door_inset=a.door_inset, relief=not a.no_relief,
                 depth=a.depth * 16 if a.depth else None)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, name)
    with open(stem + "-measure.json", "w") as f:
        json.dump(m, f, indent=1)

    thing = None
    note = None
    if a.catalog and a.thing:
        thing = lgfr.catalog_thing(a.catalog, a.thing)
    if a.thing:
        note = {"kind": "thing", "key": a.thing}
    model = lgfr.model(mesh, sprite, "imagetomodel %s" % name, measured=m, thing=thing, thing_id=a.thing, paint_all=not a.game_art)
    lgfr.write(stem + ".lgfr.json", model)
    gltf.write_glb(stem + ".glb", mesh, sprite, name=name, note=note)
    gltf.write_obj(stem + ".obj", mesh, sprite, name=name, note=note)

    score, sheet = faithful(mesh, sprite)
    sheet = sheet.resize((sheet.width * a.preview_scale, sheet.height * a.preview_scale), 0)
    sheet.save(stem + "-faithful.png")
    for view in [v for v in a.views.split(",") if v]:
        render(mesh, sprite.rgba, view, scale=a.preview_scale).save("%s-%s.png" % (stem, view))

    print("%s: %dx%d (scale %d), wall from row %d, band %d, panel %s, left %s, right %s, door %s, %d windows; %d triangles"
          % (name, sprite.W, sprite.H, sprite.scale, m["wall_top"], m["band"], m["panel"],
             m["left"] and "edge %d rows %d-%d" % (m["left"]["edge"], m["left"]["top"], m["left"]["eave"]),
             m["right"] and "edge %d rows %d-%d" % (m["right"]["edge"], m["right"]["top"], m["right"]["eave"]),
             m["door"], len(m["windows"]), len(mesh)))
    print("faithful: %.1f%% of the sprite's pixels exact, %.1f%% covered, %d drawn past it, colour distance %.3f"
          % (score["exact"] * 100, score["covered"] * 100, score["extra"], score["colour_distance"]))
    if thing is None:
        print("note: the model's prints are unset (it stands nowhere yet): pass --catalog DIR --thing ID, or bring %s.glb back with LGFR3D's tools/kit.py import" % name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
