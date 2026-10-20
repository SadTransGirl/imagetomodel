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
    r = sub.add_parser("room", help="a room from a screenshot of it")
    r.add_argument("shot")
    r.add_argument("-o", "--out", default="out")
    r.add_argument("--name", help="the model's name (the file's name)")
    r.add_argument("--scale", type=int, help="how many times the shot was scaled up (found from its pixel blocks)")
    r.add_argument("--spec", help="a JSON file: {\"wall_rows\": N, \"tiles\": {\"x,y\": \"floor\"|\"wall\"|\"wall_side\"|\"thing\"|\"void\"}, \"front_height\": px}")
    r.add_argument("--front-height", default="auto", help="how tall a free-standing thing's front is (pixels, unsquashed): 'auto' reads it from the art's shading (the darker band along its bottom)")
    r.add_argument("--low-height", type=float, default=8.0, help="how tall a thing of one tile is (pixels)")
    r.add_argument("--wall-thick", type=float, default=4.0, help="how thick the walls down the sides are (pixels)")
    r.add_argument("--fixes", action="append", default=[], help="a JSON file of things set by hand, by fingerprint (see NAME-labels.json); the library shipped with the tool always applies")
    r.add_argument("--style", choices=("voxel", "faces"), default="voxel", help="voxel: a cube a pixel, as made in MagicaVoxel (with .vox files); faces: boxes with their sides painted, in relief")
    r.add_argument("--relief", type=int, default=3, help="how many pixels the lightest pixel of a thing's face stands out over the darkest, as voxel art (0: flat)")
    r.add_argument("--thick", type=float, default=2.0, help="how thick a cutout (a plant, a flower) is (pixels)")
    r.add_argument("--tops", choices=("painted", "picture"), default="painted", help="the top of a thing standing against the wall: painted from its art, or the picture the GBA draws there (the wall's band: exact from its angle)")
    r.add_argument("--views", default="three-quarter,front", help="which other views to draw (%s)" % ",".join(VIEWS))
    r.add_argument("--preview-scale", type=int, default=4)
    a = ap.parse_args(argv)
    if a.command == "room":
        return room_main(a)

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


def room_main(a):
    from . import room
    sprite = Sprite.load(a.shot, scale=a.scale)
    room.mark_void(sprite)
    name = a.name or os.path.splitext(os.path.basename(a.shot))[0]
    spec = json.load(open(a.spec)) if a.spec else {}
    g = room.grid(sprite)
    info = room.label(sprite, g, spec)
    things = room.runs(info, g)
    front = "auto" if str(a.front_height) == "auto" else float(a.front_height)
    if a.style == "voxel":
        return room_voxel_main(a, sprite, name, spec, g, info, things, front)
    mesh = room.build(sprite, g, info, things, front_height=front, low_height=a.low_height,
                      wall_thick=a.wall_thick, spec=spec, tops=a.tops, levels=a.relief, thick=a.thick)
    objs = room.objects(info, g, things)
    cut = room.cutouts(sprite, g, info, things)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, name)
    with open(stem + "-labels.json", "w") as f:
        json.dump({"grid": g, "wall_rows": info["wall_rows"], "wall_first": info["wall_first"], "labels": info["labels"],
                   "things": things, "objects": [{k: v for k, v in o.items() if k not in ("own", "tiles")} for o in objs],
                   "floor_colours": info["floor_colours"], "wall_colours": info["wall_colours"]}, f, indent=1)
    room.labels_picture(sprite, g, info, things, scale=a.preview_scale).save(stem + "-labels.png")
    # the whole room as one model; and each thing as a model of its own tiles
    whole = lgfr.model(mesh, sprite, "imagetomodel room %s" % name, extra_cut=cut)
    whole["room"] = {"grid": g, "wall_rows": info["wall_rows"]}
    lgfr.write(stem + ".lgfr.json", whole)
    # each thing as a model of its own, in a folder of its own: its tiles' picture, its
    # painted sides in the canvas's strip, for Blender, Blockbench and the kit's import
    things_dir = stem + "-things"
    os.makedirs(things_dir, exist_ok=True)
    made = 0
    for k, o in enumerate(objs):
        one = room.build_object(sprite, g, info, things, k, front_height=front, low_height=a.low_height, spec=spec, tops=a.tops, levels=a.relief, thick=a.thick)
        if one is None:
            continue
        crop, omesh, ocut, block = one
        folder = os.path.join(things_dir, "thing-%d" % k)
        os.makedirs(folder, exist_ok=True)
        ostem = os.path.join(folder, "thing-%d" % k)
        model = lgfr.model(omesh, crop, "imagetomodel %s thing %d" % (name, k), extra_cut=ocut)
        model["imagetomodel"]["tiles"] = [o["columns"][0], o["rows"][0], o["columns"][1], o["rows"][1]]
        model["imagetomodel"]["bbox"] = o["bbox"]
        model["imagetomodel"]["kind"] = o["kind"]
        lgfr.write(ostem + ".lgfr.json", model)
        gltf.write_glb(ostem + ".glb", omesh, crop, name="thing-%d" % k, cut=ocut)
        gltf.write_obj(ostem + ".obj", omesh, crop, name="thing-%d" % k, cut=ocut)
        render(omesh, crop.rgba, "three-quarter", scale=a.preview_scale).save(ostem + "-three-quarter.png")
        render(omesh, crop.rgba, "back", scale=a.preview_scale).save(ostem + "-back.png")
        made += 1
    gltf.write_glb(stem + ".glb", mesh, sprite, name=name, split=True, cut=cut)
    gltf.write_obj(stem + ".obj", mesh, sprite, name=name, split=True, cut=cut)
    # (the faithfulness: with the wall's pixels cut from the things at the wall's foot, the
    # wall behind shows there: the same pixels)
    score, sheet = faithful(mesh, sprite)
    sheet = sheet.resize((sheet.width * a.preview_scale, sheet.height * a.preview_scale), 0)
    sheet.save(stem + "-faithful.png")
    for view in [v for v in a.views.split(",") if v]:
        render(mesh, sprite.rgba, view, scale=a.preview_scale).save("%s-%s.png" % (stem, view))
    kinds = {}
    for t in things:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print("%s: %dx%d (scale %d), grid at %d,%d: %d x %d tiles, the wall %d rows from row %d, things: %s; %d triangles"
          % (name, sprite.W, sprite.H, sprite.scale, g["ox"], g["oy"], g["cols"], g["rows"], info["wall_rows"], info["wall_first"],
             ", ".join("%d %s" % (n, k) for k, n in sorted(kinds.items())), len(mesh)))
    print("faithful: %.1f%% of the picture's pixels exact, %.1f%% covered, %d drawn past it, colour distance %.3f"
          % (score["exact"] * 100, score["covered"] * 100, score["extra"], score["colour_distance"]))
    print("labels: %s-labels.png (W wall, S side wall, T thing: h hanging, A against the wall, b a box, B a box standing tall, c a cutout); correct them with --spec" % name)
    print("%d things as models of their own in %s/thing-N/ (a .glb, .obj and model JSON each, their sides painted from their art)" % (made, things_dir))
    return 0


def room_voxel_main(a, sprite, name, spec, g, info, things, front):
    """The room in true voxels: a .vox for MagicaVoxel, the port's model, a .glb and .obj
    meshed from the cubes; and each thing the same, in a folder of its own."""
    from . import room, voxel
    fixes = room.load_fixes(a.fixes, spec)
    grid, objs = room.build_voxels(sprite, g, info, things, front_height=front, low_height=a.low_height,
                                   wall_thick=int(a.wall_thick), spec=spec, thick=int(a.thick), fixes=fixes)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, name)
    with open(stem + "-labels.json", "w") as f:
        listed = []
        for k, o in enumerate(objs):
            entry = {key: v for key, v in o.items() if key not in ("own", "tiles")}
            entry["owner"] = "thing-%d" % k
            entry["fingerprint"] = room.fingerprint(sprite, o)
            fix = room.fix_for(sprite, o, fixes)
            if fix:
                entry["fixed"] = fix.get("name", True)
            listed.append(entry)
        json.dump({"grid": g, "wall_rows": info["wall_rows"], "wall_first": info["wall_first"], "labels": info["labels"],
                   "things": things, "objects": listed,
                   "floor_colours": info["floor_colours"], "wall_colours": info["wall_colours"]}, f, indent=1)
    room.labels_picture(sprite, g, info, things, scale=a.preview_scale).save(stem + "-labels.png")
    voxel.write_vox(stem + ".vox", grid)
    mesh = voxel.mesh_of(grid, strip_v=sprite.H, strip_w=sprite.W)
    whole = lgfr.model(mesh, sprite, "imagetomodel room %s" % name, paint_all=False)
    whole["room"] = {"grid": g, "wall_rows": info["wall_rows"], "style": "voxel"}
    lgfr.write(stem + ".lgfr.json", whole)
    gltf.write_glb(stem + ".glb", mesh, sprite, name=name, split=True)
    gltf.write_obj(stem + ".obj", mesh, sprite, name=name, split=True)
    score, sheet = faithful(mesh, sprite)
    sheet.resize((sheet.width * a.preview_scale, sheet.height * a.preview_scale), 0).save(stem + "-faithful.png")
    for view in [v for v in a.views.split(",") if v]:
        render(mesh, sprite.rgba, view, scale=a.preview_scale).save("%s-%s.png" % (stem, view))
    # each thing on its own: its cubes moved to its tiles' corner
    things_dir = stem + "-things"
    os.makedirs(things_dir, exist_ok=True)
    made = 0
    for k, o in enumerate(objs):
        owner = "thing-%d" % k
        mine = grid.only(owner)
        if not len(mine):
            continue
        (ta, tb), (ca, cb) = o["rows"], o["columns"]
        bx, by = g["ox"] + ca * 16, g["oy"] + ta * 16
        crop = sprite.crop(bx, by, (cb - ca + 1) * 16, (tb - ta + 1) * 16)
        moved = mine.moved(-bx, 0, -by)
        folder = os.path.join(things_dir, owner)
        os.makedirs(folder, exist_ok=True)
        ostem = os.path.join(folder, owner)
        voxel.write_vox(ostem + ".vox", moved)
        omesh = voxel.mesh_of(moved, strip_v=crop.H, strip_w=crop.W, name=owner)
        model = lgfr.model(omesh, crop, "imagetomodel %s %s" % (name, owner), paint_all=False)
        model["imagetomodel"].update({"tiles": [ca, ta, cb, tb], "bbox": o["bbox"], "kind": o["kind"], "cubes": len(moved)})
        lgfr.write(ostem + ".lgfr.json", model)
        gltf.write_glb(ostem + ".glb", omesh, crop, name=owner)
        gltf.write_obj(ostem + ".obj", omesh, crop, name=owner)
        render(omesh, crop.rgba, "three-quarter", scale=a.preview_scale).save(ostem + "-three-quarter.png")
        render(omesh, crop.rgba, "back", scale=a.preview_scale).save(ostem + "-back.png")
        made += 1
    kinds = {}
    for o in objs:
        kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
    print("%s: %dx%d (scale %d), grid at %d,%d: %d x %d tiles, the wall %d rows from row %d, things: %s; %d cubes, %d faces"
          % (name, sprite.W, sprite.H, sprite.scale, g["ox"], g["oy"], g["cols"], g["rows"], info["wall_rows"], info["wall_first"],
             ", ".join("%d %s" % (n, k) for k, n in sorted(kinds.items())), len(grid), len(mesh.polys)))
    print("faithful: %.1f%% of the picture's pixels exact, %.1f%% covered, %d drawn past it, colour distance %.3f"
          % (score["exact"] * 100, score["covered"] * 100, score["extra"], score["colour_distance"]))
    fixed = [room.fix_for(sprite, o, fixes).get("name") for o in objs]
    fixed = [f for f in fixed if f]
    print("%s.vox for MagicaVoxel; %d things on their own in %s/thing-N/ (a .vox, .glb, .obj and the port's model each)" % (name, made, things_dir))
    if fixed:
        print("set by hand (fixes): %s" % ", ".join(fixed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
