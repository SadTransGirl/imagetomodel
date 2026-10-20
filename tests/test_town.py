"""A town from a decompilation the test draws itself: a tileset of two tiles (grass, a
wall), a 6x6 map with a house of 2x2 cells, labelled as LGFR3D labels them."""
import json
import os
import struct
import tempfile

from PIL import Image

from imagetomodel import town, voxel
from imagetomodel.decomp import Decomp
from imagetomodel.render import faithful

GRASS, WALL, ROOF, LINE = (100, 200, 100), (230, 230, 240), (230, 110, 90), (50, 50, 70)


def _decomp(root, houses=(2,)):
    os.makedirs(os.path.join(root, "src/data/tilesets"))
    with open(os.path.join(root, "src/data/tilesets/headers.h"), "w") as f:
        for sym in ("General", "Test"):
            f.write("const struct Tileset gTileset_%s =\n{\n    .tiles = gTilesetTiles_%s,\n    .palettes = gTilesetPalettes_%s,\n};\n" % (sym, sym, sym))
    for kind, name in (("primary", "general"), ("secondary", "test")):
        d = os.path.join(root, "data/tilesets", kind, name)
        os.makedirs(os.path.join(d, "palettes"))
        pal = [(0, 0, 0), GRASS, WALL, ROOF, LINE] + [(0, 0, 0)] * 11
        with open(os.path.join(d, "palettes", "00.pal"), "w") as f:
            f.write("JASC-PAL\n0100\n16\n" + "\n".join("%d %d %d" % c for c in pal) + "\n")
        # (tiles: 0 clear (a top layer's nothing); 1 grass; 2 a wall, outlined at its foot; 3 a
        # roof, outlined at its top)
        im = Image.new("P", (32, 8), 0)
        for y in range(8):
            for x in range(8):
                im.putpixel((8 + x, y), 1)
                im.putpixel((16 + x, y), 4 if y == 7 else 2)
                im.putpixel((24 + x, y), 4 if y == 0 else 3)
        im.putpalette([v for c in pal for v in c] + [0] * (768 - 48))
        im.save(os.path.join(d, "tiles.png"))
        # (metatiles: 0 grass; 1 a house's front cell (wall); 2 its top cell (roof))
        meta = b"".join(struct.pack("<8H", t, t, t, t, 0, 0, 0, 0) for t in (1, 2, 3))
        open(os.path.join(d, "metatiles.bin"), "wb").write(meta)
        open(os.path.join(d, "metatile_attributes.bin"), "wb").write(struct.pack("<3I", 0, 0, 0))
    os.makedirs(os.path.join(root, "data/layouts/Test"))
    cells = [[0] * 6 for _ in range(6)]
    for h in houses:
        for x in (h, h + 1):
            cells[2][x] = 2
            cells[3][x] = 1
    with open(os.path.join(root, "data/layouts/Test/map.bin"), "wb") as f:
        f.write(struct.pack("<36H", *[c | ((1 if c else 0) << 10) for row in cells for c in row]))
    json.dump({"layouts": [{"id": "LAYOUT_TEST", "width": 6, "height": 6, "primary_tileset": "gTileset_General",
                            "secondary_tileset": "gTileset_Test", "blockdata_filepath": "data/layouts/Test/map.bin"}]},
              open(os.path.join(root, "data/layouts/layouts.json"), "w"))
    os.makedirs(os.path.join(root, "data/maps/Test"))
    json.dump({"id": "MAP_TEST", "layout": "LAYOUT_TEST"}, open(os.path.join(root, "data/maps/Test/map.json"), "w"))


def test_a_town_from_its_decompilation():
    with tempfile.TemporaryDirectory() as root:
        _decomp(root)
        d = Decomp(root)
        m = d.map("Test")
        assert (m.width, m.height) == (6, 6) and m.grid[3][2] == 1 and m.collision[3][2] == 1
        pic = d.draw(m)
        assert pic.getpixel((40, 50))[:3] == WALL and pic.getpixel((40, 33))[:3] == ROOF
        tw = town.Town(d, m, {"general:0x001": "building", "general:0x002": "building"})
        assert [t["class"] for t in tw.things] == ["building"]
        assert sorted(tw.things[0]["tiles"]) == [(2, 2), (2, 3), (3, 2), (3, 3)]
        # (its front the wall, under its roof read from the top down)
        assert town.front_rows(tw.sprite, tw.things[0]) == 16
        grid = town.build_voxels(tw, fixes={})
        house = grid.only("thing-0")
        assert max(y for _, y, _ in house.cubes) == 15 and len(grid.only("ground")) == 96 * 96
        # (its roof deeper than the GBA draws it (16 rows), so drawn a little higher)
        zs = [z for _, _, z in house.cubes]
        assert max(zs) - min(zs) + 1 >= 23
        score, _ = faithful(voxel.mesh_of(grid, strip_v=96, strip_w=96), tw.sprite)
        assert score["exact"] > 0.95, score


def test_a_town_wider_than_magicavoxel_holds_is_written_in_blocks():
    g = voxel.Grid()
    for x in range(300):
        g.set(x, 0, 0, (10, 20, 30))
    with tempfile.TemporaryDirectory() as d:
        voxel.write_vox(os.path.join(d, "a.vox"), g)
        raw = open(os.path.join(d, "a.vox"), "rb").read()
        tags, off = [], 20
        while off < len(raw):
            n, k = struct.unpack_from("<ii", raw, off + 4)
            tags.append(raw[off:off + 4])
            off += 12 + n + k
        assert tags.count(b"XYZI") == 2 and tags.count(b"nSHP") == 2


def test_the_same_thing_drawn_twice_is_one_model():
    with tempfile.TemporaryDirectory() as root:
        _decomp(root, houses=(0, 3))
        d = Decomp(root)
        tw = town.Town(d, d.map("Test"), {"general:0x001": "building", "general:0x002": "building"})
        assert len(tw.things) == 2
        assert tw.things[0]["model"] == tw.things[1]["model"]
        grid = town.build_voxels(tw, fixes={})
        a, b = grid.only("thing-0"), grid.only("thing-1")
        # (the same cubes, moved)
        assert {(x, y, z): c for (x, y, z), c in a.cubes.items()} == {(x - 48, y, z): c for (x, y, z), c in b.cubes.items()}


def test_a_roofs_pattern_repeats_in_its_period():
    tile = [[ROOF] * 6] * 3 + [[LINE] * 6]
    rows = [[LINE] * 6] + tile * 5 + [[WALL] * 6]          # (rims either end)
    assert town.period(rows) == 4
    assert town.period([[c] * 6 for c in (ROOF, LINE, WALL, GRASS, ROOF, WALL, LINE, GRASS)]) is None



def test_a_hipped_roof_rises_to_a_ridge_over_eaves_all_round():
    class Pic:
        def colour(self, x, v):
            return (ROOF if v < 6 else WALL) + (255,)
    # (a front 10 wide, 4 high; a roof drawn 12 wide, 6 rows, a cube past it either side)
    own = {(x, v) for x in range(1, 11) for v in range(6, 10)} | {(x, v) for x in range(12) for v in range(6)}
    kinds = lambda x, v: "up" if v >= 6 else "back"
    shown = {(x, v): (9 - v, 10) for x in range(1, 11) for v in range(6, 10)}
    part = voxel.Grid()
    for x in range(1, 11):
        for y in range(4):
            for z in range(4, 11):
                part.set(x, y, z, WALL, "t")
    town.hip_roof(part, Pic(), own, kinds, shown, "t", {"hip": 2, "depth": 1.5})
    roof = {k for k in part.cubes if k[1] >= 4}
    assert min(x for x, _, _ in roof) == 0 and max(x for x, _, _ in roof) == 11      # (eaves past the walls)
    assert max(z for _, _, z in roof) == 11                                           # (and past the front)
    tops = {}
    for x, y, z in roof:
        tops[(x, z)] = max(tops.get((x, z), 0), y)
    assert tops[(0, 6)] == 4 and tops[(6, 6)] > tops[(3, 6)] > tops[(0, 6)]           # (rising to a ridge)
    assert not any(x in (0, 11) and y < 4 for x, y, _ in part.cubes)                  # (on nothing past the walls)
