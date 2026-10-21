import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from synthetic_room import room as draw_room                  # noqa: E402

from imagetomodel.picture import Sprite                        # noqa: E402
from imagetomodel import room, gltf                            # noqa: E402
from imagetomodel.render import faithful, render               # noqa: E402
from imagetomodel.cli import main                              # noqa: E402


def _labelled(scale=1, spec=None):
    s = Sprite(draw_room(scale=scale))
    room.mark_void(s)
    g = room.grid(s)
    info = room.label(s, g, spec)
    return s, g, info


def test_grid_is_found_on_a_scrolled_screen():
    s = Sprite(draw_room())
    room.mark_void(s)
    assert room.grid(s) == {"ox": 0, "oy": 0, "cols": 12, "rows": 8}
    # (the same room, the screen scrolled 5 right and 3 down)
    shifted = np.zeros_like(s.rgba)
    shifted[3:, 5:] = s.rgba[:-3, :-5]
    from PIL import Image
    s2 = Sprite(Image.fromarray(shifted, "RGBA"))
    room.mark_void(s2)
    g = room.grid(s2)
    assert (g["ox"], g["oy"]) == (5, 3)


def test_tiles_are_told_from_their_pictures():
    s, g, info = _labelled()
    L = info["labels"]
    assert info["wall_rows"] == 2 and info["wall_first"] == 0
    assert all(L[0][tx] == "wall" for tx in range(12) if tx not in (5, 6))   # (the poster's top may count as wall)
    assert L[1][5] == "thing" and L[1][4] == "thing" and L[1][1] == "thing" and L[1][8] == "wall"
    assert L[2][0] == "wall_side" and L[2][11] == "wall_side"
    assert L[2][1] == "thing"                        # (the pot)
    assert all(L[3][tx] == "thing" for tx in range(2, 9)) and all(L[4][tx] == "thing" for tx in range(2, 9))
    assert all(L[4][tx] == "floor" for tx in (0, 1, 9, 10, 11))      # (the shadowed floor)
    assert L[6][9] == "thing" and L[5][5] == "floor"
    assert L[7][5] == "floor" and L[7][6] == "floor" and L[7][0] == "void"


def test_things_stand_as_the_port_builds_them():
    s, g, info = _labelled()
    things = {(t["x"], tuple(t["rows"])): t["kind"] for t in room.runs(info, g)}
    assert things[(1, (1, 2))] == "against-wall"     # (the plant and its pot)
    assert things.get((5, (0, 1)), things.get((5, (1, 1)))) == "on-wall"   # (the poster)
    assert things[(4, (1, 3))] == "against-wall"     # (the machine, its foot on the counter's back edge)
    assert things[(4, (3, 4))] == "free"              # (the counter under it, its own run)
    assert things[(2, (3, 4))] == "free"
    assert things[(9, (6, 6))] == "free"


def test_a_spec_forces_a_tile_and_a_kind():
    s, g, info = _labelled(spec={"tiles": {"9,6": "box", "8,1": "thing"}, "wall_rows": 2})
    things = {(t["x"], tuple(t["rows"])): t["kind"] for t in room.runs(info, g)}
    assert things[(9, (6, 6))] == "tall"
    assert things[(8, (1, 1))] == "on-wall"


def test_room_is_faithful_from_the_gba_angle():
    s, g, info = _labelled()
    things = room.runs(info, g)
    mesh = room.build(s, g, info, things)
    score, _ = faithful(mesh, s)
    # (the machine's painted top covers the wall's band above it from the GBA's angle)
    assert score["exact"] >= 0.975, score
    assert score["covered"] >= 0.995, score
    exact = faithful(room.build(s, g, info, things, tops="picture"), s)[0]["exact"]
    assert exact >= 0.985, exact
    tq = np.array(render(mesh, s.rgba, "three-quarter", scale=1))
    assert (tq[:, :, 3] > 0).sum() > 100 * 80


def test_objects_are_split_for_the_kit():
    s, g, info = _labelled()
    things = room.runs(info, g)
    mesh = room.build(s, g, info, things)
    names = {n.split(":")[0] for n, _, _ in mesh.polys}
    assert "floor" in names and "wall" in names and "side-left" in names and "side-right" in names
    objs = room.objects(info, g, things)
    assert sum(1 for n in names if n.startswith("thing-")) == sum(1 for o in objs if o["kind"] not in ("on-wall", "flat"))
    # (the counter's seven columns are one object, the mat on the void's edge lies flat)
    assert any(o["columns"] == [2, 8] and o["rows"] == [3, 4] for o in objs)
    assert any(o["kind"] == "flat" for o in objs)
    with tempfile.TemporaryDirectory() as d:
        gltf.write_glb(os.path.join(d, "r.glb"), mesh, s, split=True)
        raw = open(os.path.join(d, "r.glb"), "rb").read()
        import struct
        jl, = struct.unpack_from("<I", raw, 12)
        doc = json.loads(raw[20:20 + jl])
        assert {n["name"] for n in doc["nodes"]} == names


def test_room_cli_writes_everything():
    with tempfile.TemporaryDirectory() as d:
        draw_room(scale=3).save(os.path.join(d, "r.png"))
        assert main(["room", os.path.join(d, "r.png"), "-o", os.path.join(d, "out"), "--views", "three-quarter", "--style", "faces"]) == 0
        for f in ("r.lgfr.json", "r.glb", "r.obj", "r.png", "r-labels.json", "r-labels.png", "r-faithful.png", "r-three-quarter.png"):
            assert os.path.exists(os.path.join(d, "out", f)), f
        things = sorted(os.listdir(os.path.join(d, "out", "r-things")))
        assert len(things) >= 3
        painted = 0
        for name in things:
            folder = os.path.join(d, "out", "r-things", name)
            files = os.listdir(folder)
            assert any(f.endswith(".lgfr.json") for f in files) and any(f.endswith(".glb") for f in files) and any(f.endswith(".obj") for f in files)
            one = json.load(open(os.path.join(folder, [f for f in files if f.endswith(".lgfr.json")][0])))
            assert one["pieces"][0]["kind"] == "mesh" and one["size"][0] >= 1
            # (a box's sides are painted in the canvas's strip: pixels past the picture)
            strip = one["pieces"][0]["canvas"][1] - one["size"][1] * 16
            assert strip >= 32
            if any(y >= one["size"][1] * 16 + 1 for xy in one["pieces"][0]["paint"].values() for y in xy[1::2]):
                painted += 1
        assert painted >= 2
        lab = json.load(open(os.path.join(d, "out", "r-labels.json")))
        assert lab["wall_rows"] == 2


def test_a_flower_in_a_pot_is_a_pot_leaves_and_a_head():
    import numpy as np
    from PIL import Image
    from imagetomodel import voxel
    HEAD, LEAF, POT, LINE = (80, 140, 200), (100, 200, 100), (170, 150, 70), (60, 60, 80)
    im = np.zeros((16, 16, 4), dtype=np.uint8)
    for y in range(1, 6):
        for x in range(5, 11):
            im[y, x] = (*HEAD, 255)                       # (its head, standing)
    for y in range(6, 8):
        for x in range(3, 13):
            im[y, x] = (*LEAF, 255)                       # (its leaves, lying)
    for y in range(8, 13):
        for x in range(4, 12):
            im[y, x] = (*(LINE if y == 12 else POT), 255)  # (its pot's front)
    s = Sprite(Image.fromarray(im, "RGBA"), scale=1)
    own = {(x, y) for y in range(16) for x in range(16) if s.drawn[y, x]}
    o = {"bbox": [3, 1, 13, 13]}
    fix = {"shape": "plant", "crown": {"rows": [0, 6], "profile": [["back", 2], ["up", "rest"]], "thick": 3, "mirror": True},
           "profile": [["up", 5], ["back", "rest"]], "pot_depth": 6, "rim": "d6c66b", "soil": {"colour": "52526b"}}
    g = voxel.Grid()
    room.build_plant(g, s, o, own, fix, "thing-0")
    head = [k for k, c in g.cubes.items() if c == HEAD]
    leaves = [k for k, c in g.cubes.items() if c == LEAF]
    pot = [k for k, c in g.cubes.items() if c in (POT, LINE, (0xd6, 0xc6, 0x6b), (0x52, 0x52, 0x6b))]
    assert len({z for _, _, z in head}) == 3                                      # (thick)
    zs = {z for _, _, z in head}
    assert any(z < min(zs) for _, _, z in leaves) and any(z > max(zs) for _, _, z in leaves)  # (round it)
    assert max(z for _, _, z in pot) - min(z for _, _, z in pot) + 1 == 6        # (its pot square)
    assert (0xd6, 0xc6, 0x6b) in g.cubes.values() and (0x52, 0x52, 0x6b) in g.cubes.values()
    assert min(y for _, y, _ in head) > max(y for _, y, _ in pot)                # (over the pot)


class _Pic:
    """A picture as the builders read one: {(x, y): colour}."""
    def __init__(self, px):
        self.px = px

    def colour(self, x, y):
        return self.px.get((x, y))


def test_a_spec_makes_things_in_a_box_one_and_its_rugs_floor():
    s, g, info = _labelled(spec={"wall_rows": 2, "things": [[2, 3, 8, 4]], "tiles": {"5,5": "floor"}})
    objs = room.objects(info, g, room.runs(info, g))
    inside = [o for o in objs if any(2 <= tx <= 8 and 3 <= ty <= 4 for tx, ty in o["tiles"])]
    assert len(inside) == 1                                   # (the counter and what's on it, one)
    assert (5, 5) in info["rug"] and (6, 6) in info["rug"] and (8, 8) not in info["rug"]


def test_include_takes_a_things_rows_but_the_rug_round_it():
    rug, grey = (106, 197, 106), (123, 123, 123)
    pic = _Pic({(x, y): (grey if 2 <= x <= 5 else rug) for x in range(8) for y in range(4)})
    o = {"bbox": [0, 2, 8, 4], "own": set()}
    got = room.fix_included(pic, o, {"include": [{"rows": [-2, 1], "cols": [0, 7], "except": ["6ac56a"]}]})
    assert got == {(x, y) for x in range(2, 6) for y in range(4)}


def test_stairs_climb_a_step_to_each_tread_drawn():
    from imagetomodel import voxel
    dark, tread, light, rail, side = (82, 82, 106), (213, 197, 106), (246, 238, 148), (172, 148, 65), (180, 180, 164)
    px = {}
    # (4 steps of 4 columns, a dark riser then tread; each drawn 2 rows higher; under the
    # first, a grey side panel 3 rows; the treads long runs down to row 20)
    for k in range(4):
        x0 = 10 + 4 * k
        top = 10 - 2 * k
        for x in range(x0, x0 + 4):
            for y in range(top, 21):
                px[(x, y)] = dark if x == x0 else tread
            px[(x, top - 1)] = light
        px[(x0 + 1, 21)] = rail
    for x in range(10, 14):
        for y in range(21, 24):
            px[(x, y)] = side
    pic = _Pic(px)
    own = set(px)
    xs, ys = [p[0] for p in own], [p[1] for p in own]
    o = {"own": own, "bbox": [min(xs), min(ys), max(xs) + 1, max(ys) + 1]}
    grid = voxel.Grid()
    room.build_stairs(grid, pic, o, {}, "t")
    tops = {}
    for (x, y, z) in grid.cubes:
        tops[x] = max(tops.get(x, -1), y)
    steps = [tops[10 + 4 * k + 2] for k in range(4)]
    assert all(b - a == 2 for a, b in zip(steps, steps[1:]))           # (2 higher a step, as drawn)


def test_a_spec_box_splits_a_thing_with_tiles_past_it():
    s, g, info = _labelled(spec={"wall_rows": 2})
    whole = [o for o in room.objects(info, g, room.runs(info, g)) if any(t == (2, 3) for t in o["tiles"])][0]
    assert any(t[0] > 2 for t in whole["tiles"])                       # (the counter, wider than a tile)
    s, g, info = _labelled(spec={"wall_rows": 2, "things": [[2, 3, 2, 4]]})
    objs = room.objects(info, g, room.runs(info, g))
    mine = [o for o in objs if (2, 3) in o["tiles"]][0]
    assert all(t[0] == 2 for t in mine["tiles"])
    assert all(16 * 2 <= x < 16 * 3 for x, _ in mine["own"])
    assert any(t[0] > 2 for o in objs for t in o["tiles"] if o is not mine and t[1] in (3, 4))


def test_a_stairwell_opens_the_floor_and_steps_down_it():
    from imagetomodel import voxel
    void, tread, dark, white = (123, 123, 123), (213, 197, 106), (82, 82, 106), (255, 255, 255)
    px = {}
    for x in range(0, 24):
        for v in range(0, 20):
            px[(x, v)] = void
    # (3 steps from the right, each drawn 2 rows lower to the left)
    for k in range(3):
        for x in range(20 - 4 * k, 24 - 4 * k):
            for v in range(3 + 2 * k, 20):
                px[(x, v)] = dark if x == 20 - 4 * k else tread
    for x in range(0, 24):
        for v in (20, 21):
            px[(x, v)] = white
    pic = _Pic(px)
    o = {"own": set(px), "bbox": [0, 0, 24, 22]}
    grid = voxel.Grid()
    opened = room.build_stairwell(grid, pic, o, {"opening": [0, 0, 23, 19], "rails": {"front": [20, 21], "height": 3}}, "t")
    assert (10, 5) in opened and len(opened) == 24 * 20
    tops = {x: max(y for (xx, y, z) in grid.cubes if xx == x and z == 10 and y < 0) for x in (22, 18, 14)}
    assert tops[22] > tops[18] > tops[14]                                  # (down to the left)
    assert all(y < 0 for (x, y, z) in grid.cubes if 0 <= z <= 19 and 0 <= x <= 23)


def test_a_fix_the_same_as_anothers_is_that_ones_but_what_it_says():
    s, g, info = _labelled()
    o = room.objects(info, g, room.runs(info, g))[0]
    first = "%d,%d" % (o["columns"][0], o["rows"][0])
    fixes = {"palm": {"name": "Potted palm", "shape": "plant", "thick": 3},
             first: {"name": "The same palm, upstairs", "same_as": "palm"}}
    fix = room.fix_for(s, o, fixes, g)
    assert fix["shape"] == "plant" and fix["thick"] == 3 and fix["name"] == "The same palm, upstairs"


def test_the_floor_under_a_thing_is_what_the_game_draws_under_it():
    from types import SimpleNamespace
    wood, rug, bed = (164, 148, 49), (106, 197, 106), (255, 255, 255)
    shot = np.zeros((48, 48, 4), dtype=np.uint8)
    shot[:, :, :3] = wood
    shot[:, :, 3] = 255
    bottom = np.zeros((48, 48, 3), dtype=int)
    bottom[:, :] = wood
    bottom[16:32, 16:32] = bed                                   # (a bed drawn on the bottom layer)
    bottom[16:32, 32:48] = rug                                   # (a chair's cell: the rug under it)
    info = {"bottom": bottom, "sprite": SimpleNamespace(rgba=shot), "labels": [["floor", "floor", "floor"],
            ["floor", "thing", "thing"], ["floor", "floor", "floor"]], "rug": {(2, 1): {rug}}}
    objs = [{"kind": "free", "tiles": [(1, 1)], "own": set()}, {"kind": "free", "tiles": [(2, 1)], "own": set()}]
    got = room.floor_beneath(info, {"ox": 0, "oy": 0}, objs, {wood})
    assert got[(2, 1)][5][5] == rug                               # (its own bottom layer)
    assert got[(1, 1)][5][5] == wood                              # (the bed's: the boards round it)


def test_stairs_climb_to_the_floor_above_if_told():
    from imagetomodel import voxel
    dark, tread, side = (82, 82, 106), (213, 197, 106), (180, 180, 164)
    px = {}
    for k in range(4):
        x0, top = 10 + 4 * k, 10 - 2 * k
        for x in range(x0, x0 + 4):
            for y in range(top, 21):
                px[(x, y)] = dark if x == x0 else tread
    for x in range(10, 14):
        for y in range(21, 24):
            px[(x, y)] = side
    pic = _Pic(px)
    own = set(px)
    o = {"own": own, "bbox": [min(p[0] for p in own), min(p[1] for p in own), max(p[0] for p in own) + 1, max(p[1] for p in own) + 1]}
    grid = voxel.Grid()
    room.build_stairs(grid, pic, o, {"to": 20, "rail": 0}, "t")
    top = max(y for (x, y, z) in grid.cubes if x == 23)
    assert top == 20



def test_a_boxs_front_is_squared_to_its_foot_and_its_floor_shadow_dropped():
    body, dark, floor = (200, 160, 60), (40, 40, 60), (230, 222, 156)
    px = {}
    for y in range(6):
        for x in range(10):
            px[(x, y)] = dark if x in (0, 9) else body
    for x in range(1, 9):
        px[(x, 6)] = dark if x in (1, 8) else body        # (a row narrower: a rounded corner)
    for x in range(0, 10):
        px[(x, 7)] = floor                                # (its shadow on the floor)
    pic = _Pic(px)
    own, added = room.square_front(pic, set(px), {floor})
    assert not any(v == 7 for _, v in own)                # (the shadow row gone)
    assert (0, 6) in own and (9, 6) in own and added[(0, 6)] == body


def test_things_changed_by_hand_are_learnt_and_placed_again():
    from imagetomodel import voxel
    s, g, info = _labelled()
    things = room.runs(info, g)
    grid, objs = room.build_voxels(s, g, info, things)
    k = next(k for k, o in enumerate(objs) if o["kind"] == "free" and o["columns"] == [9, 9])
    edited = voxel.Grid()
    edited.cubes, edited.owner = dict(grid.cubes), dict(grid.owner)
    mine = [key for key, o in grid.owner.items() if o == "thing-%d" % k]
    top = max(key[1] for key in mine)
    for key in mine:                                      # (a hat on it, and recoloured)
        if key[1] == top:
            edited.set(key[0], top + 1, key[2], (250, 10, 10), "x")
            edited.cubes[key] = (10, 10, 250)
    with tempfile.TemporaryDirectory() as d:
        kept = room.learn_models(grid, objs, edited, s, g, {}, d)
        assert kept == [objs[k]["kind"]]
        models = room.load_models(d)
        grid2, objs2 = room.build_voxels(s, g, info, things, models=models)
    again = {key: c for key, c in grid2.cubes.items() if grid2.owner[key] == "thing-%d" % k}
    theirs = {key: edited.cubes[key] for key in edited.cubes if key in mine or key[1] == top + 1 and (key[0], top, key[2]) in mine}
    assert again == theirs
