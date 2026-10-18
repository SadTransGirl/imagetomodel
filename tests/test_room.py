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
    assert all(L[0][tx] == "wall" for tx in range(12) if tx not in (5, 6))   # (the poster)
    assert L[0][5] == "thing" and L[1][4] == "thing" and L[1][1] == "thing" and L[1][8] == "wall"
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
    assert things[(5, (0, 1))] == "on-wall"           # (the poster)
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
    assert score["exact"] >= 0.985, score
    assert score["covered"] >= 0.995, score
    tq = np.array(render(mesh, s.rgba, "three-quarter", scale=1))
    assert (tq[:, :, 3] > 0).sum() > 100 * 80


def test_objects_are_split_for_the_kit():
    s, g, info = _labelled()
    things = room.runs(info, g)
    mesh = room.build(s, g, info, things)
    names = {n.split(":")[0] for n, _, _ in mesh.polys}
    assert "floor" in names and "wall" in names and "side-left" in names and "side-right" in names
    assert sum(1 for n in names if n.startswith("thing-")) == sum(1 for t in things if t["kind"] != "on-wall")
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
        assert main(["room", os.path.join(d, "r.png"), "-o", os.path.join(d, "out"), "--views", "three-quarter"]) == 0
        for f in ("r.lgfr.json", "r.glb", "r.obj", "r.png", "r-labels.json", "r-labels.png", "r-faithful.png", "r-three-quarter.png"):
            assert os.path.exists(os.path.join(d, "out", f)), f
        things = os.listdir(os.path.join(d, "out", "r-things"))
        assert len(things) >= 5
        one = json.load(open(os.path.join(d, "out", "r-things", sorted(things)[0])))
        assert one["size"][0] == 1 and one["pieces"][0]["kind"] == "mesh"
        lab = json.load(open(os.path.join(d, "out", "r-labels.json")))
        assert lab["wall_rows"] == 2
