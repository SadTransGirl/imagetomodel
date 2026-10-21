import json
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from synthetic_room import room as draw_room                  # noqa: E402

from imagetomodel.picture import Sprite                        # noqa: E402
from imagetomodel import room, voxel                           # noqa: E402
from imagetomodel.render import faithful, render               # noqa: E402
from imagetomodel.cli import main                              # noqa: E402


def _box_sprite():
    """A box as the GBA draws one: a light top (6 rows), outlined, over a darker front
    (5 rows) with a dark foot, on nothing."""
    im = np.zeros((16, 16, 4), dtype=np.uint8)
    TOP, FRONT, LINE = (250, 220, 90), (200, 140, 40), (60, 60, 80)
    for y in range(2, 13):
        for x in range(2, 14):
            edge = x in (2, 13) or y in (2, 12)
            c = LINE if edge else (TOP if y < 8 else FRONT)
            im[y, x] = (*c, 255)
    return Sprite(Image.fromarray(im, "RGBA"))


def test_a_cube_shows_its_front_and_top_on_two_rows():
    g = voxel.Grid()
    g.set(3, 2, 7, (255, 0, 0))
    mesh = voxel.mesh_of(g, strip_v=16, strip_w=16)
    assert len(mesh.polys) == 6
    tex = np.zeros((16, 16, 4), dtype=np.uint8)
    view = np.array(render(mesh, tex, "gba", scale=1, frame=(0, 0, 16, 16), shade=False))
    drawn = [(x, y) for y in range(16) for x in range(16) if view[y, x, 3]]
    # (its front face on row z - y = 5, its top on row 4)
    assert sorted(drawn) == [(3, 4), (3, 5)]


def test_greedy_faces_merge_one_colour():
    g = voxel.Grid()
    for x in range(4):
        for z in range(3):
            g.set(x, 0, z, (10, 20, 30))
    mesh = voxel.mesh_of(g)
    assert len(mesh.polys) == 6                      # (a slab: one face a side)


def test_a_box_is_unprojected_faithfully():
    s = _box_sprite()
    own = {(x, y) for y in range(16) for x in range(16) if s.drawn[y, x]}
    g = voxel.Grid()
    voxel.unproject_box(g, s, own, lambda x, top, zf: 5)
    mesh = voxel.mesh_of(g, strip_v=s.H, strip_w=s.W)
    score, _ = faithful(mesh, s)
    # (exact but for the row along the front's top edge: a cube there shows two rows)
    assert score["covered"] == 1.0 and score["extra"] == 0
    assert score["exact"] >= 1 - 12.0 / len(own) - 1e-9, score
    ys = {k[1] for k in g.cubes}
    zs = {k[2] for k in g.cubes}
    assert ys == set(range(5)) and min(zs) == 2 + 5 and max(zs) == 12


def test_vox_round_trip():
    g = voxel.Grid()
    g.set(0, -1, 5, (1, 2, 3))
    g.set(2, 4, 9, (200, 100, 50))
    g.set(2, 4, 8, (200, 100, 50))
    with tempfile.TemporaryDirectory() as d:
        voxel.write_vox(os.path.join(d, "t.vox"), g)
        size, cubes, palette = voxel.read_vox(os.path.join(d, "t.vox"))
    assert size == (3, 5, 6)
    assert len(cubes) == 3
    colours = {tuple(palette[c[3] - 1][:3]) for c in cubes}
    assert colours == {(1, 2, 3), (200, 100, 50)}
    # (the front at the low end of the depth, up the third axis)
    assert (0, 4, 0) in {c[:3] for c in cubes}


def test_room_in_voxels_is_faithful_and_things_stand():
    s = Sprite(draw_room())
    room.mark_void(s)
    g = room.grid(s)
    info = room.label(s, g)
    things = room.runs(info, g)
    grid, objs = room.build_voxels(s, g, info, things)
    mesh = voxel.mesh_of(grid, strip_v=s.H, strip_w=s.W)
    score, _ = faithful(mesh, s)
    # (what's built past the picture, a closed hole, a thing proud of the wall, costs a little)
    assert score["exact"] >= 0.95, score
    assert score["covered"] >= 0.995, score
    owners = set(grid.owner.values())
    assert {"floor", "wall", "side-left", "side-right"} <= owners
    # (the counter stands as tall as its front is drawn, the stool a box, the plant a slab)
    counter = next(k for k, o in enumerate(objs) if o["columns"] == [2, 8])
    ys = [k[1] for k, c in grid.cubes.items() if grid.owner[k] == "thing-%d" % counter]
    assert max(ys) + 1 == room._front_rows(info, objs[counter])
    plant = next(k for k, o in enumerate(objs) if o["cutout"] and o["kind"] == "against-wall")
    assert len(grid.only("thing-%d" % plant)) > 0
    # (nothing stands behind the back wall's face)
    wall_z = (info["wall_first"] + info["wall_rows"]) * 16
    assert all(z >= wall_z for (x, y, z), o in grid.owner.items() if o.startswith("thing"))


def test_room_cli_writes_voxels():
    with tempfile.TemporaryDirectory() as d:
        draw_room(scale=2).save(os.path.join(d, "r.png"))
        assert main(["room", os.path.join(d, "r.png"), "-o", os.path.join(d, "out"), "--views", "three-quarter"]) == 0
        out = os.path.join(d, "out")
        for f in ("r.vox", "r.lgfr.json", "r.glb", "r.obj", "r-labels.json", "r-faithful.png", "r-three-quarter.png"):
            assert os.path.exists(os.path.join(out, f)), f
        size, cubes, _ = voxel.read_vox(os.path.join(out, "r.vox"))
        assert len(cubes) > 1000 and max(size) <= 256
        things = sorted(os.listdir(os.path.join(out, "r-things")))
        assert len(things) >= 3
        for name in things:
            folder = os.path.join(out, "r-things", name)
            assert os.path.exists(os.path.join(folder, name + ".vox"))
            model = json.load(open(os.path.join(folder, name + ".lgfr.json")))
            piece = model["pieces"][0]
            assert piece["kind"] == "mesh" and len(piece["tris"]) >= 9
            # (every face wears a plain colour painted in the strip under the picture)
            H = model["size"][1] * 16
            assert all(v >= H for v in piece["uvs"][1::2])


def test_the_floor_shadow_beside_a_thing_is_not_the_thing():
    s = _box_sprite()
    im = s.rgba.copy()
    SHADOW = (123, 123, 123, 255)
    for y in range(4, 14):
        im[y, 14] = SHADOW                               # (to its right)
    for x in range(4, 15):
        im[13, x] = SHADOW                               # (below it)
    s2 = Sprite(Image.fromarray(im, "RGBA"))
    own = {(x, y) for y in range(16) for x in range(16) if s2.drawn[y, x]}
    shadow = room._drop_shadow(s2, own)
    assert {(14, y) for y in range(4, 14)} <= shadow and {(x, 13) for x in range(4, 15)} <= shadow
    # (none of the box itself: its outline and its grey-free body stay)
    assert not shadow & {(x, y) for y in range(2, 13) for x in range(2, 14)}


def test_a_stepped_profile_stands_and_lies_in_turn():
    s = _box_sprite()
    own = {(x, y) for y in range(16) for x in range(16) if s.drawn[y, x]}
    g = voxel.Grid()
    kinds = voxel.segments_kinds([["up", 3], ["back", 4], ["up", "rest"]], 13)
    shown = voxel.unproject_profile(g, s, own, kinds)
    score, _ = faithful(voxel.mesh_of(g, strip_v=16, strip_w=16), s)
    # (the dark ends of the rows that lie wear the colour next in: a little less exact)
    assert score["exact"] >= 0.9, score
    # (a lip three tall, a tray four deep behind it, a panel standing behind that)
    col = sorted((y, z) for (x, y, z) in g.cubes if x == 6)
    assert max(y for y, z in col if z == 12) == 2
    assert max(y for y, _ in col) > 2 and min(z for _, z in col) < 9
    assert shown[(6, 12)] == (0, 12)


def test_recess_and_raise():
    s = _box_sprite()
    own = {(x, y) for y in range(16) for x in range(16) if s.drawn[y, x]}
    g = voxel.Grid()
    kinds = voxel.box_kinds(lambda x, t, zf: 5, own)
    voxel.unproject_profile(g, s, own, kinds, recess={(6, 10): 2}, raised={(6, 4)})
    assert (6, 2, 12) not in g.cubes and (6, 2, 10) in g.cubes        # (set two in)
    assert (6, 5, 9) in g.cubes                                       # (a cube on the top)


def test_lathe_is_round_and_skips_the_outline():
    im = np.zeros((16, 16, 4), dtype=np.uint8)
    for y in range(4, 14):
        for x in range(4, 12):
            edge = x in (4, 11) or y in (4, 13)
            im[y, x] = (40, 40, 60, 255) if edge else (80, 200, 90, 255)
    s = Sprite(Image.fromarray(im, "RGBA"))
    own = {(x, y) for y in range(16) for x in range(16) if s.drawn[y, x]}
    g = voxel.Grid()
    voxel.lathe(g, s, own, 0, 20, "pot")
    zs = {z for (x, y, z) in g.cubes if y == 4}
    assert max(zs) - min(zs) >= 4                                     # (deep as it is wide)
    assert (40, 40, 60) not in {c for (x, y, z), c in g.cubes.items() if 0 < y < 9}


def test_fixes_by_fingerprint_and_tile():
    s = Sprite(draw_room())
    room.mark_void(s)
    g = room.grid(s)
    info = room.label(s, g)
    things = room.runs(info, g)
    objs = room.objects(info, g, things)
    stool = next(o for o in objs if o["columns"] == [9, 9] and o["rows"] == [6, 6])
    fp = room.fingerprint(s, stool)
    assert len(fp) == 12 and fp == room.fingerprint(s, stool)
    fixes = room.load_fixes(spec={"objects": {fp: {"name": "stool", "front": 3}}})
    assert room.fix_for(s, stool, fixes)["name"] == "stool"
    fixes = room.load_fixes(spec={"objects": {"9,6": {"name": "by tile"}}})
    assert room.fix_for(s, stool, fixes)["name"] == "by tile"
    counter = next(o for o in objs if o["columns"] == [2, 8])
    cfp = room.fingerprint(s, counter)
    grid, objs2 = room.build_voxels(s, g, info, things, fixes=room.load_fixes(spec={"objects": {cfp: {"front": 3}}}))
    k = next(i for i, o in enumerate(objs2) if o["columns"] == [2, 8])
    assert max(y for (x, y, z), o in grid.owner.items() if o == "thing-%d" % k) == 2


def _escalator_sprite():
    """Two handrails (orange) climbing to the left over glass, treads between them."""
    W, H = 64, 48
    im = np.zeros((H, W, 4), dtype=np.uint8)
    im[:, :] = (231, 222, 156, 255)                                    # (the floor)
    RAIL, GLASS, DARK, TREAD = (255, 90, 41), (198, 222, 206), (82, 82, 107), (148, 148, 165)
    for x in range(0, 30):
        rise = max(0, (16 - x) // 2)
        for y in range(18, 34):
            im[y, x] = (*DARK, 255)
            if y % 2 == 0 and x % 8 in (2, 3, 4, 5):
                im[y, x] = (*TREAD, 255)
        for base, glass_rows in ((10, 6), (34, 6)):
            r = base - rise
            im[r, x] = (*RAIL, 255)
            for j in range(1, glass_rows):
                im[r + j, x] = (*GLASS, 255)
    return Sprite(Image.fromarray(im, "RGBA"))


def test_an_escalator_is_built_climbing_where_its_handrails_climb():
    s = _escalator_sprite()
    o = {"own": {(x, y) for y in range(48) for x in range(30) if tuple(s.rgba[y, x, :3]) != (231, 222, 156)},
         "bbox": [0, 2, 30, 40]}
    assert room._treads(s, 0, 16) and not room._treads(s, 40, 16)
    g = voxel.Grid()
    room.build_escalator(g, s, o, {}, "escalator")
    rail = (255, 90, 41)
    tops = {}
    for (x, y, z), c in g.cubes.items():
        if c == rail:
            tops[x] = max(tops.get(x, 0), y)
    assert tops[0] > tops[25]                                         # (higher at the left)
    assert len({z for (x, y, z), c in g.cubes.items() if c == rail}) >= 2   # (two balustrades)


def test_what_hangs_on_the_wall_stands_proud_of_it():
    s = Sprite(draw_room())
    room.mark_void(s)
    g = room.grid(s)
    info = room.label(s, g)
    things = room.runs(info, g)
    grid, objs = room.build_voxels(s, g, info, things)
    k = next(i for i, o in enumerate(objs) if o["kind"] == "on-wall")
    cubes = [key for key, o in grid.owner.items() if o == "thing-%d" % k]
    wall_z = (info["wall_first"] + info["wall_rows"]) * 16
    assert cubes and min(z for _, _, z in cubes) >= wall_z            # (in front of the wall's face)
    assert max(z for _, _, z in cubes) - min(z for _, _, z in cubes) >= 1
    # (moved up as far as it stands out: its picture where the GBA draws it)
    mesh = voxel.mesh_of(grid, strip_v=s.H, strip_w=s.W)
    view = np.array(render(mesh, s.rgba, "gba", scale=1, frame=(0, 0, s.W, s.H), shade=False))
    poster = objs[k]["own"]
    same = sum(1 for (x, y) in poster if tuple(view[y, x, :3]) == s.colour(x, y))
    assert same >= 0.8 * len(poster)


def test_holes_in_a_thing_are_closed():
    im = np.zeros((16, 16, 4), dtype=np.uint8)
    for y in range(2, 14):
        for x in range(2, 14):
            im[y, x] = (200, 200, 210, 255)
    s = Sprite(Image.fromarray(im, "RGBA"), scale=1)
    info = {"sprite": s, "floor_colours": [], "wall_colours": []}
    own = {(x, y) for y in range(2, 14) for x in range(2, 14)} - {(7, y) for y in range(6, 14)} - {(x, 9) for x in range(4, 12)}
    holes = room._holes(info, own, {})
    assert (7, 9) in holes and (5, 9) in holes and (7, 12) in holes
    assert (1, 9) not in holes


def test_a_things_shadow_on_the_wall_is_not_painted_on_the_wall():
    img = draw_room()
    im = np.array(img)
    for y in range(21, 57):
        im[y, 78] = (123, 123, 123, 255)                # (right of the machine's outline)
    s = Sprite(Image.fromarray(im, "RGBA"), scale=1)
    room.mark_void(s)
    g = room.grid(s)
    info = room.label(s, g)
    things = room.runs(info, g)
    grid, objs = room.build_voxels(s, g, info, things)
    wall = [c[:3] for (x, y, z), c in grid.cubes.items() if x == 78 and grid.owner[(x, y, z)] == "wall"]
    assert wall and (123, 123, 123) not in wall


def test_a_lying_recess_sinks_along_the_line_of_sight():
    s = _box_sprite()
    own = {(x, y) for y in range(16) for x in range(16) if s.drawn[y, x]}
    kinds = voxel.box_kinds(lambda x, t, zf: 5, own)
    flat, sunk = voxel.Grid(), voxel.Grid()
    voxel.unproject_profile(flat, s, own, kinds)
    tray = {(x, v): 2 for x in range(5, 11) for v in range(4, 7)}
    shown = voxel.unproject_profile(sunk, s, own, kinds, recess=tray)
    # (its top two cubes lower than the rim round it, its picture where the GBA draws it)
    assert shown[(7, 5)][0] == shown[(3, 5)][0] - 2
    score, _ = faithful(voxel.mesh_of(sunk, strip_v=16, strip_w=16), s)
    was, _ = faithful(voxel.mesh_of(flat, strip_v=16, strip_w=16), s)
    assert score["exact"] == was["exact"], (score, was)


def test_a_thing_is_outlined_round_its_sides_and_back():
    s = _box_sprite()
    own = {(x, y) for y in range(16) for x in range(16) if s.drawn[y, x]}
    g = voxel.Grid()
    voxel.unproject_profile(g, s, own, voxel.box_kinds(lambda x, t, zf: 5, own), owner="thing-0")
    before = dict(g.cubes)
    was, _ = faithful(voxel.mesh_of(g, strip_v=16, strip_w=16), s)
    LINE = (60, 60, 80)
    assert voxel.outline(g, "thing-0", LINE) > 0
    # (what the GBA shows unchanged; the back's upright edges and its foot outlined, its
    # middle the body's colour)
    score, _ = faithful(voxel.mesh_of(g, strip_v=16, strip_w=16), s)
    assert score["exact"] == was["exact"], (score, was)
    zmin = min(z for _, _, z in g.cubes)
    back = {(x, y): c for (x, y, z), c in g.cubes.items() if z == zmin}
    xs = [x for x, _ in back]
    assert back[(min(xs), 2)] == LINE and back[(max(xs), 2)] == LINE and back[(7, 0)] == LINE
    assert back[(7, 2)] != LINE
    assert any(g.cubes[k] != before[k] for k in before)


def test_a_round_thing_is_outlined_only_where_its_picture_is():
    im = np.zeros((16, 16, 4), dtype=np.uint8)
    HEAD, POT, LINE = (240, 240, 250), (200, 170, 60), (60, 60, 80)
    for y in range(2, 8):
        for x in range(4, 12):
            im[y, x] = (*HEAD, 255)                       # (a head of petals, unoutlined)
    for y in range(8, 14):
        for x in range(5, 11):
            im[y, x] = (*(LINE if x in (5, 10) or y == 13 else POT), 255)
    s = Sprite(Image.fromarray(im, "RGBA"), scale=1)
    own = {(x, y) for y in range(16) for x in range(16) if s.drawn[y, x]}
    g = voxel.Grid()
    voxel.lathe(g, s, own, 0, 20, "thing-0", depth=0.5)
    voxel.outline(g, "thing-0", LINE, keep_shown=False, where=voxel.lathe_outlined(s, own, 0))
    head = [c for (x, y, z), c in g.cubes.items() if y >= 6]
    pot = [c for (x, y, z), c in g.cubes.items() if 1 <= y <= 4]
    assert LINE not in head and LINE in pot


def test_a_canopys_fronds_lie_level_each_pixel_where_its_drawn():
    own = {(3, v) for v in range(0, 10)} | {(6, v) for v in range(2, 10)}
    kinds = voxel.columns_kinds([{"cols": [3, 3], "profile": [["up", 2], ["back", "rest"]]}],
                                voxel.segments_kinds([["up", 5], ["back", "rest"]], 10), 0, 10)
    cells = voxel.canopy(own, kinds, 10)
    for (x, v), (y, z) in cells.items():
        assert z - y == v if kinds(x, v) == "up" else z - y - 1 == v      # (on its line of sight)
    # (both columns' fronds at height 0, their leaves hanging under them)
    assert {y for (x, v), (y, z) in cells.items() if kinds(x, v) == "back"} == {0}
    assert min(y for y, _ in cells.values()) < 0


def test_a_things_sides_painted_plain_a_panel_inset():
    s = _box_sprite()
    own = {(x, y) for y in range(16) for x in range(16) if s.drawn[y, x]}
    g = voxel.Grid()
    voxel.unproject_profile(g, s, own, voxel.box_kinds(lambda x, t, zf: 5, own), owner="thing-0")
    was, _ = faithful(voxel.mesh_of(g, strip_v=16, strip_w=16), s)
    SIDE, PANEL = (120, 120, 120), (180, 180, 170)
    voxel.paint_sides(g, "thing-0", SIDE, PANEL)
    score, _ = faithful(voxel.mesh_of(g, strip_v=16, strip_w=16), s)
    assert score["exact"] == was["exact"]                         # (the GBA's view unchanged)
    left = {(y, z): c for (x, y, z), c in g.cubes.items() if x == 2}
    assert left[(2, 9)] == PANEL and left[(0, 9)] == SIDE         # (inset, its foot the frame)


def test_one_side_painted_the_other_left():
    s = _box_sprite()
    own = {(x, y) for y in range(16) for x in range(16) if s.drawn[y, x]}
    g = voxel.Grid()
    voxel.unproject_profile(g, s, own, voxel.box_kinds(lambda x, t, zf: 5, own), owner="thing-0")
    right = {k: c for k, c in g.cubes.items() if k[0] == 13}
    SCREEN = (82, 82, 107)
    voxel.paint_sides(g, "thing-0", (180, 180, 170), SCREEN, which=(0,))
    assert SCREEN in {c for k, c in g.cubes.items() if k[0] == 2}
    assert all(g.cubes[k] == c for k, c in right.items())


def test_front_corners_rounded_their_bands_running_round():
    g = voxel.Grid()
    BAND, LINE = (200, 140, 40), (60, 60, 80)
    for x in range(0, 20):
        for z in range(0, 10):
            for y in range(3):
                g.set(x, y, z, LINE if x in (0, 19) else BAND, "thing-0")
    n = voxel.round_corners(g, "thing-0", 5)
    assert n == 5 * 3 * 2                                     # (both front corners, each layer)
    assert (0, 1, 9) not in g.cubes and (0, 1, 0) in g.cubes   # (its back corners left square)
    assert g.cubes[(1, 1, 8)] == BAND                          # (the band, not the outline, round it)


def test_a_top_stretched_deeper_than_drawn():
    s = _box_sprite()
    own = {(x, y) for y in range(16) for x in range(16) if s.drawn[y, x]}
    kinds = voxel.box_kinds(lambda x, t, zf: 5, own)
    drawn, deep = voxel.Grid(), voxel.Grid()
    voxel.unproject_profile(drawn, s, own, kinds)
    voxel.unproject_profile(deep, s, own, kinds, stretch=2.0)
    depth = lambda g: max(z for _, _, z in g.cubes) - min(z for _, _, z in g.cubes) + 1
    assert depth(deep) > depth(drawn) + 3
    # (its front where the GBA draws it, its top's colours the same, only deeper)
    assert {c for (x, y, z), c in deep.cubes.items() if z == 12} == {c for (x, y, z), c in drawn.cubes.items() if z == 12}
    assert {c for (x, y, z), c in deep.cubes.items() if y == 4} == {c for (x, y, z), c in drawn.cubes.items() if y == 4}


def test_a_column_drawn_only_high_starts_at_the_things_foot():
    s = _box_sprite()
    own = {(x, y) for y in range(16) for x in range(16) if s.drawn[y, x]}
    own -= {(2, v) for v in range(6, 13)}                     # (its left column's foot not drawn)
    kinds = voxel.segments_kinds([["up", 5], ["back", "rest"]], 13)
    g = voxel.Grid()
    shown = voxel.unproject_profile(g, s, own, kinds, foot=13)
    # (its left column's top lies with the others' tops, not stood up from its own lowest)
    assert shown[(2, 4)] == shown[(6, 4)]



def _front(colours, rows=6):
    """A sprite of a box's front: each row the given colours across (a list a column)."""
    im = np.zeros((rows, len(colours), 4), dtype=np.uint8)
    for x, c in enumerate(colours):
        im[:, x, :3] = c
    im[:, :, 3] = 255
    return Sprite(Image.fromarray(im, "RGBA"), scale=1)


def test_a_side_wears_the_bands_body_not_the_fronts_edge_pixel():
    body, light, dark = (200, 160, 60), (255, 255, 240), (40, 40, 60)
    s = _front([dark, light, body, body, body, body, body, body, dark])
    own = {(x, y) for y in range(6) for x in range(9)}
    g = voxel.Grid()
    voxel.unproject_profile(g, s, own, voxel.segments_kinds([["up", 3], ["back", "rest"]], 6))
    left = {c for (x, y, z), c in g.cubes.items() if x == 0 and y < 3 and z < 5}
    assert light not in left                              # (a pixel of highlight: not a stripe down the side)
    # (two pixels of it: a panel, carried round)
    s2 = _front([dark, light, light, body, body, body, body, body, dark])
    g2 = voxel.Grid()
    voxel.unproject_profile(g2, s2, own, voxel.segments_kinds([["up", 3], ["back", "rest"]], 6))
    assert light in {c for (x, y, z), c in g2.cubes.items() if x == 0 and y < 3 and z < 5}


def test_a_tops_edges_wear_the_colour_next_in_not_the_outline():
    body, dark = (200, 160, 60), (40, 40, 60)
    s = _front([dark, body, body, body, body, dark])
    own = {(x, y) for y in range(6) for x in range(6)}
    g = voxel.Grid()
    voxel.unproject_profile(g, s, own, voxel.segments_kinds([["up", 2], ["back", "rest"]], 6))
    top = [(x, c) for (x, y, z), c in g.cubes.items() if y == 1 and z < 4]
    assert all(c == body for x, c in top if x in (0, 5))    # (no dark stripe down the top's edges)
    assert any(c == dark for (x, y, z), c in g.cubes.items() if y < 2 and x == 0)   # (the front's corner still dark)


def test_things_on_a_shelf_stand_proud_of_its_back():
    back, item = (120, 120, 120), (220, 60, 60)
    s = _front([back, back, item, item, back, back], rows=4)
    own = {(x, y) for y in range(4) for x in range(6)}
    g = voxel.Grid()
    recess = {(x, y): 3 for x in range(6) for y in range(4)}
    stand = {(x, y): 2 for x in (2, 3) for y in range(4)}
    voxel.unproject_profile(g, s, own, voxel.segments_kinds([["up", "rest"]], 4), recess=recess, stand=stand)
    depths = sorted({z for (x, y, z), c in g.cubes.items() if c == item})
    assert len(depths) == 3                               # (the item 3 deep: its back and 2 proud)
    assert len({z for (x, y, z), c in g.cubes.items() if c == back and x == 0}) == 1   # (the back wall, 1)
