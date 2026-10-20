"""A map from pret's decompilation (github.com/pret/pokefirered), as LGFR3D reads it
(tools/object_library.py): its layout (a metatile a cell), its tilesets' metatiles drawn
from their tiles and palettes, each tile's behaviour, and the map's events. Nothing of the
game's is kept here: it's read from a copy of the decompilation the user points to.

    d = Decomp("external/pokefirered")
    m = d.map("PalletTown")          # m.width, m.height, m.grid[y][x] (metatile), m.events
    picture = d.draw(m)              # the map as the GBA draws it, 16 pixels a cell
"""
import json
import os
import re
import struct

from PIL import Image

PRIMARY_METATILES = 640          # (FireRed's: the secondary set's metatiles count from here)
PRIMARY_TILES = 640
PRIMARY_PALETTES = 7


def snake(sym):
    """A tileset's symbol as its folder's name (gTileset_PalletTown: pallet_town; SSAnne's
    capitals in a row stay together)."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[a-z])(?=[0-9])", "_",
                  sym.replace("gTileset_", "")).lower()


class Map:
    def __init__(self, name, data, layout, primary, secondary):
        self.name = name
        self.data = data
        self.width, self.height = layout["width"], layout["height"]
        self.primary, self.secondary = primary, secondary
        self.grid = [[0] * self.width for _ in range(self.height)]
        self.collision = [[0] * self.width for _ in range(self.height)]
        self.elevation = [[0] * self.width for _ in range(self.height)]
        self.events = data


class Decomp:
    def __init__(self, root):
        self.root = root
        self.layouts = {l["id"]: l for l in json.load(open(os.path.join(root, "data/layouts/layouts.json")))["layouts"] if "id" in l}
        self.heads = {}
        text = open(os.path.join(root, "src/data/tilesets/headers.h")).read()
        for m in re.finditer(r"const struct Tileset (gTileset_\w+) =\s*\{(.*?)\};", text, re.S):
            t = re.search(r"\.tiles = gTilesetTiles_(\w+)", m.group(2))
            p = re.search(r"\.palettes = gTilesetPalettes_(\w+)", m.group(2))
            self.heads[snake(m.group(1))] = (snake(t.group(1)) if t else None, snake(p.group(1)) if p else None)
        self.sets = {}
        self.cache = {}
        self.bottoms = {}

    def map(self, name):
        data = json.load(open(os.path.join(self.root, "data/maps", name, "map.json")))
        lay = self.layouts[data["layout"]]
        m = Map(name, data, lay, snake(lay["primary_tileset"]), snake(lay["secondary_tileset"]))
        raw = open(os.path.join(self.root, lay["blockdata_filepath"]), "rb").read()
        v = struct.unpack("<%dH" % (len(raw) // 2), raw)
        for y in range(m.height):
            for x in range(m.width):
                c = v[y * m.width + x]
                m.grid[y][x] = c & 0x3FF
                m.collision[y][x] = (c >> 10) & 3
                m.elevation[y][x] = c >> 12
        return m

    def _folder(self, name, which):
        for kind in ("primary", "secondary"):
            own = os.path.join(self.root, "data/tilesets", kind, name)
            if os.path.isdir(own):
                other = self.heads.get(name, (None, None))[which]
                d = os.path.join(self.root, "data/tilesets", kind, other) if other else own
                return d if os.path.isdir(d) else own
        raise FileNotFoundError(name)

    def _palettes(self, d):
        out = []
        for k in range(16):
            p = os.path.join(d, "palettes", "%02d.pal" % k)
            if os.path.exists(p):
                lines = open(p).read().split("\n")[3:19]
                out.append([tuple(map(int, l.split())) for l in lines if l.strip()])
            else:
                out.append([(0, 0, 0)] * 16)
        return out

    def _set(self, name):
        if name not in self.sets:
            own = None
            for kind in ("primary", "secondary"):
                if os.path.isdir(os.path.join(self.root, "data/tilesets", kind, name)):
                    own = os.path.join(self.root, "data/tilesets", kind, name)
            tiles = Image.open(os.path.join(self._folder(name, 0), "tiles.png"))
            self.sets[name] = {"tiles": tiles if tiles.mode == "P" else tiles.convert("P"),
                               "meta": open(os.path.join(own, "metatiles.bin"), "rb").read(),
                               "attrs": open(os.path.join(own, "metatile_attributes.bin"), "rb").read(),
                               "pals": self._palettes(self._folder(name, 1))}
        return self.sets[name]

    def attributes(self, m, mid):
        """(behaviour, layer type) of a metatile."""
        s = self._set(m.primary if mid < PRIMARY_METATILES else m.secondary)
        a = struct.unpack_from("<I", s["attrs"], (mid % PRIMARY_METATILES) * 4)[0]
        return a & 0x1FF, (a >> 29) & 3

    def metatile(self, m, mid):
        """(both layers, the top layer alone) of a metatile, 16x16 RGBA."""
        key = (m.primary, m.secondary, mid)
        if key in self.cache:
            return self.cache[key]
        p, s = self._set(m.primary), self._set(m.secondary)
        palette = p["pals"][:PRIMARY_PALETTES] + s["pals"][PRIMARY_PALETTES:16]
        src = p if mid < PRIMARY_METATILES else s
        ent = struct.unpack_from("<8H", src["meta"], (mid % PRIMARY_METATILES) * 16)
        layers = []
        for layer in range(2):
            im = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
            px = im.load()
            for k in range(4):
                e = ent[layer * 4 + k]
                t, hf, vf, pal = e & 0x3FF, (e >> 10) & 1, (e >> 11) & 1, e >> 12
                sheet = p["tiles"] if t < PRIMARY_TILES else s["tiles"]
                t2 = t % PRIMARY_TILES
                tw = sheet.width // 8
                if (t2 // tw) * 8 >= sheet.height:
                    continue
                tile = sheet.crop(((t2 % tw) * 8, (t2 // tw) * 8, (t2 % tw) * 8 + 8, (t2 // tw) * 8 + 8))
                if hf:
                    tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
                if vf:
                    tile = tile.transpose(Image.FLIP_TOP_BOTTOM)
                for y in range(8):
                    for x in range(8):
                        c = tile.getpixel((x, y))
                        if c == 0:
                            if layer == 0:
                                px[(k % 2) * 8 + x, (k // 2) * 8 + y] = palette[0][0] + (255,)
                            continue
                        px[(k % 2) * 8 + x, (k // 2) * 8 + y] = palette[pal][c] + (255,)
            layers.append(im)
        whole = layers[0].copy()
        whole.alpha_composite(layers[1])
        self.cache[key] = (whole, layers[1])
        self.bottoms[key] = layers[0]
        return self.cache[key]

    def bottom(self, m, mid):
        """A metatile's bottom layer alone (the ground under what its top layer draws)."""
        self.metatile(m, mid)
        return self.bottoms[(m.primary, m.secondary, mid)]

    def draw(self, m):
        """The map as the GBA draws it: a metatile a 16-pixel cell, both its layers."""
        im = Image.new("RGBA", (m.width * 16, m.height * 16))
        for y in range(m.height):
            for x in range(m.width):
                im.paste(self.metatile(m, m.grid[y][x])[0], (x * 16, y * 16))
        return im

    def key(self, m, mid):
        """A metatile's name as LGFR3D's labels key it ("pallet_town:0x296")."""
        return "%s:0x%03X" % (m.primary if mid < PRIMARY_METATILES else m.secondary, mid)
