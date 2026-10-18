import json
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from synthetic import house                                   # noqa: E402

from imagetomodel.picture import Sprite, detect_scale          # noqa: E402
from imagetomodel.measure import measure                       # noqa: E402
from imagetomodel.building import build                        # noqa: E402
from imagetomodel.render import faithful, render               # noqa: E402
from imagetomodel import gltf, lgfr                            # noqa: E402
from imagetomodel.cli import main                              # noqa: E402


def test_scale_is_found_and_undone():
    big = house(scale=6)
    assert detect_scale(np.array(big)) == 6
    s = Sprite(big)
    assert (s.W, s.H, s.scale) == (80, 80, 6)
    assert np.array_equal(s.rgba, np.array(house()))


def test_measures_read_back_what_was_drawn():
    s = Sprite(house())
    m = measure(s)
    assert m["top"] == 8
    assert m["wall_top"] == 43
    assert m["band"] == 3
    assert m["panel"] == [12, 67]
    assert m["top_centre"] == 8
    assert m["left"]["edge"] == 0 and m["left"]["drop"] == 6 and m["left"]["top"] == 14
    assert m["right"]["edge"] == 80 and m["right"]["drop"] == 6
    assert m["wall"] == [1, 79]
    assert m["door"] == [15, 64, 31, 80]
    assert len(m["windows"]) == 3
    assert m["windows"][0] == [16, 48, 26, 55]


def test_spec_wins_over_what_was_read():
    s = Sprite(house())
    m = measure(s, {"wall_top": 40, "band": 1, "door": None})
    assert m["wall_top"] == 40 and m["band"] == 1 and m["door"] is None


def test_model_is_faithful_from_the_gba_angle():
    s = Sprite(house())
    mesh = build(s, measure(s))
    score, sheet = faithful(mesh, s)
    assert score["exact"] >= 0.97, score
    assert score["covered"] >= 0.995, score
    assert score["extra"] <= 8                 # (a slope's stepped back edge, straightened)
    assert sheet.size == (3 * 80 + 8, 80)


def test_relief_keeps_the_picture_in_place():
    s = Sprite(house())
    m = measure(s)
    flat = faithful(build(s, m, relief=False), s)[0]
    deep = faithful(build(s, m, inset=3, door_inset=4), s)[0]
    assert abs(flat["exact"] - deep["exact"]) < 0.005


def test_other_views_show_the_roof_and_the_sides():
    s = Sprite(house())
    mesh = build(s, measure(s))
    tq = np.array(render(mesh, s.rgba, "three-quarter", scale=1))
    drawn = tq[:, :, 3] > 0
    assert drawn.sum() > 80 * 40
    # (the top panel's colour is seen from above: the roof is there and in front of the walls)
    panel = (tq[:, :, :3].astype(int) - np.array([239, 148, 115])).__abs__().sum(axis=2) < 90
    assert (panel & drawn).sum() > 500
    side = np.array(render(mesh, s.rgba, "side", scale=1))
    assert (side[:, :, 3] > 0).sum() > 30 * 30


def _read_glb(path):
    raw = open(path, "rb").read()
    assert raw[:4] == b"glTF"
    length, = struct.unpack_from("<I", raw, 8)
    assert length == len(raw)
    jl, jt = struct.unpack_from("<II", raw, 12)
    doc = json.loads(raw[20:20 + jl])
    bl, bt = struct.unpack_from("<II", raw, 20 + jl)
    return doc, raw[28 + jl:28 + jl + bl]


def test_exports_are_in_the_kits_conventions():
    s = Sprite(house())
    mesh = build(s, measure(s))
    with tempfile.TemporaryDirectory() as d:
        gltf.write_glb(os.path.join(d, "h.glb"), mesh, s, note={"kind": "thing", "key": "5x5-abc"})
        doc, blob = _read_glb(os.path.join(d, "h.glb"))
        assert doc["asset"]["extras"]["lgfr"] == {"kind": "thing", "key": "5x5-abc"}
        assert doc["images"][0]["name"] == "lgfr-thing-5x5-abc"
        acc = doc["accessors"][0]
        assert acc["type"] == "VEC3"
        # (a tile a unit, the block's middle at the origin: 5 tiles across, 2 deep at its back)
        assert -2.6 < acc["min"][0] < -2.4 and 2.4 < acc["max"][0] < 2.6
        assert acc["min"][1] >= -0.07 and 2.4 < acc["max"][1] < 2.6   # (a door at the ground, set back along the line of sight, dips a pixel)
        assert 2.4 < acc["max"][2] < 2.6 and -0.6 < acc["min"][2] < 0.6
        uv = doc["accessors"][1]
        assert uv["type"] == "VEC2"
        n = uv["count"]
        view = doc["bufferViews"][uv["bufferView"]]
        vals = struct.unpack_from("<%df" % (n * 2), blob, view["byteOffset"])
        us = vals[0::2]
        assert min(us) >= 0.5 and max(us) <= 1.0      # (the thing alone: the canvas's right half)
        assert all(0.0 <= v <= 1.0 for v in vals[1::2])
        gltf.write_obj(os.path.join(d, "h.obj"), mesh, s)
        text = open(os.path.join(d, "h.obj")).read()
        assert text.count("\nf ") == len(mesh.polys)
        assert os.path.exists(os.path.join(d, "h.png")) and os.path.exists(os.path.join(d, "h.mtl"))


def test_lgfr_model_holds_the_sprite_and_stands_nowhere_until_placed():
    s = Sprite(house())
    mesh = build(s, measure(s))
    model = lgfr.model(mesh, s, "test")
    assert model["size"] == [5, 5]
    assert all(p == lgfr.UNPLACED for row in model["prints"] for p in row)
    piece = model["pieces"][0]
    assert piece["kind"] == "mesh" and piece["canvas"] == [160, 80]
    assert len(piece["tris"]) == len(piece["uvs"]) // 2 * 3
    assert len(piece["tris"]) // 9 == len(mesh)
    painted = sum(len(v) // 2 for v in piece["paint"].values())
    assert painted == int(s.drawn.sum())
    assert len(piece["cut"]) // 2 == 80 * 80 - painted
    assert all(u >= 80 for u in piece["uvs"][0::2])
    thing = {"tiles": [[7] * 5] * 5, "prints": [[123456789] * 5] * 5}
    placed = lgfr.model(mesh, s, "test", thing=thing, thing_id="5x5-abc", paint_all=False)
    assert placed["prints"][0][0] == 123456789 and placed["thing"] == "5x5-abc"
    assert placed["pieces"][0]["paint"] == {}


def test_cli_writes_everything():
    with tempfile.TemporaryDirectory() as d:
        house(scale=4).save(os.path.join(d, "h.png"))
        assert main(["building", os.path.join(d, "h.png"), "-o", os.path.join(d, "out"), "--views", "three-quarter"]) == 0
        for f in ("h.lgfr.json", "h.glb", "h.obj", "h.mtl", "h.png", "h-measure.json", "h-faithful.png", "h-three-quarter.png"):
            assert os.path.exists(os.path.join(d, "out", f)), f
        m = json.load(open(os.path.join(d, "out", "h-measure.json")))
        assert m["wall_top"] == 43
