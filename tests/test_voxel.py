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
    assert score["exact"] >= 0.96, score
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
