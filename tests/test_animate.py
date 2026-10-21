import json
import os
import struct
import tempfile

import numpy as np
from PIL import Image

from imagetomodel import animate, voxel
from imagetomodel.picture import Sprite

GREEN, TRAY = (0x6b, 0xc6, 0x6b), (82, 82, 107)


def _machine():
    """A tray with two rows of two slots, each a 2x2 of green a cube proud."""
    g = voxel.Grid()
    for x in range(10):
        for z in range(8):
            g.set(x, 0, z, TRAY, "thing-0")
    for x0, z0 in ((1, 1), (6, 1), (1, 5), (6, 5)):
        for x in (x0, x0 + 1):
            for z in (z0, z0 + 1):
                g.set(x, 1, z, GREEN, "thing-0")
    return g


def test_slots_found_back_to_front_left_to_right():
    found = animate.slots(_machine(), GREEN)
    assert [min(x for x, _, _ in p) for p in found] == [1, 6, 1, 6]
    assert [min(z for _, _, z in p) for p in found] == [1, 1, 5, 5]


def test_heal_sets_balls_one_by_one_flashes_and_takes_them_away():
    g = _machine()
    spec = {"slots": {"colour": "6bc66b"}, "balls": 3, "flashes": 2}
    frames = animate.heal(g, spec)
    red = (0xf0, 0x40, 0x30)
    balls = [sum(1 for c in f.cubes.values() if c == red) for f, _ in frames]
    assert balls[:4] == [0, 4, 8, 12]                       # (a ball, 4 red cubes, at a time)
    assert 0 in balls[4:-2]                                 # (lit, they flash)
    assert balls[-1] == 0 and frames[-1][0].cubes == g.cubes


def test_animated_glb_and_vox():
    g = _machine()
    frames = animate.heal(g, {"balls": 2, "flashes": 1})
    s = Sprite(Image.fromarray(np.zeros((16, 16, 4), dtype=np.uint8), "RGBA"), scale=1)
    with tempfile.TemporaryDirectory() as d:
        animate.write_glb(os.path.join(d, "a.glb"), frames, s, name="heal")
        raw = open(os.path.join(d, "a.glb"), "rb").read()
        jl, = struct.unpack_from("<I", raw, 12)
        doc = json.loads(raw[20:20 + jl])
        anim = doc["animations"][0]
        assert anim["name"] == "heal" and len(anim["channels"]) == len(doc["nodes"])
        assert all(s["interpolation"] == "STEP" for s in anim["samplers"])
        assert sum(1 for n in doc["nodes"] if n.get("scale") != [0.0, 0.0, 0.0]) == 1   # (at rest, one frame)
        animate.write_vox_frames(os.path.join(d, "a.vox"), frames)
        raw = open(os.path.join(d, "a.vox"), "rb").read()
        tags, off = [], 20
        while off < len(raw):
            n, m = struct.unpack_from("<ii", raw, off + 4)
            tags.append(raw[off:off + 4])
            off += 12 + n + m
        assert tags.count(b"XYZI") == len(frames) and b"nSHP" in tags


def _screen():
    # (a screen 4 wide on a box: the GBA's view sees each front cube at row z - y)
    g = voxel.Grid()
    for x in range(4):
        for y in range(3):
            g.set(x, y, 5, (40, 40, 60), "t")
    return g


def test_a_pc_turns_on_as_the_game_flickers_it():
    g = _screen()
    lit = {(x, 5 - y): (150, 200, 230) for x in range(4) for y in range(3)}
    ctx = {"cells": {(0, 0): 0x062}, "changed": lambda st: lit if st.get((0, 0)) == 0x063 else {}}
    frames = animate.frames_of(g, "turn_on", {"kind": "flicker"}, ctx)
    on = [any(c == (150, 200, 230) for c in f.cubes.values()) for f, _ in frames]
    assert on == [False, True, False, True, False, True]                 # (on, off, on, off, on)
    assert [round(d, 2) for _, d in frames] == [0.6, 0.1, 0.1, 0.1, 0.1, 1.2]


def test_an_escalator_runs_its_stages_round_and_only_its_steps_change():
    g = voxel.Grid()
    dark, light, side = (82, 82, 107), (148, 148, 165), (198, 222, 206)
    for x in range(31):
        g.set(x, 0, 5, dark, "t")
    # (each stage its steps' light edge 10 pixels on, and once its side's panel)
    pictures = {s: {(x, 5): light for x in range(10 * (s - 1), 10 * s)} for s in (1, 2, 3)}
    pictures[1][(30, 5)] = side
    ctx = {"cells": {(0, 0): 3}, "old": lambda p: dark, "changed": lambda st: pictures[st[(0, 0)]]}
    frames = animate.frames_of(g, "run", {"kind": "cycle", "sections": [["0x1", "0x2", "0x3"]]}, ctx)
    assert len(frames) == 3                                                 # (its stages, looped)
    assert all(f.cubes[(30, 0, 5)] == dark for f, _ in frames)             # (its side's panel still)
    lit = sorted(min(k[0] for k, c in f.cubes.items() if c == light) for f, _ in frames)
    assert lit == [0, 10, 20]                                               # (its steps moving on)
