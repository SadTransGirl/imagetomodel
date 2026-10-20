"""A town (or any outdoor map) in true voxels, from the game's own data (decomp.py): what
each cell is, from LGFR3D's tile labels (godot/tile_labels.json: ground, water, tree,
fence, sign, building, prop, ...) or, without them, from its behaviour; the ground a layer
of cubes, each cell its picture (under a thing, the ground most like what's round it);
the water sunk a cube in its banks; each thing (a building, a fence, a sign, a tree) cut
from the ground round it and unprojected as a room's things are (voxel.py), by its fix
(the same fixes, by the same fingerprints: its tiles' pictures), outlined all round.

The trees of a border (the outdoor set's pines) are each a thing: two cells wide and two
high from its top left (an even metatile of the pines' tops)."""
import colorsys
from collections import Counter

import numpy as np

from . import room, voxel
from .picture import PX, Sprite

OBJECT_LABELS = {"building", "tree", "hedge", "fence", "sign", "prop"}
WATER_BEHAVIOURS = set(range(0x10, 0x1C))
# (the outdoor set's pines, the even ones a tree's left half: a tree is its body's two cells,
# the tip in the cell above (a tip's own, or a junction's lower half: the tree above's foot
# over it), and its foot in the cell below (a foot's own, or a junction's upper half))
PINE_TIPS = {0x0E, 0x0F}
PINE_BODIES = {0x1C, 0x1D, 0x1E, 0x1F}
PINE_JUNCTIONS = {0x14, 0x15, 0x16, 0x17}
PINE_FEET = {0x24, 0x25, 0x26, 0x27}
PINES = PINE_TIPS | PINE_BODIES | PINE_JUNCTIONS | PINE_FEET


class Town:
    """A map read: its picture (a Sprite), each cell's label, its things."""

    def __init__(self, decomp, m, labels=None):
        self.decomp, self.map = decomp, m
        self.sprite = Sprite(decomp.draw(m), scale=1)
        self.W, self.H = m.width, m.height
        labels = labels or {}
        self.labels = [[self._label(labels, x, y) for x in range(self.W)] for y in range(self.H)]
        self.things = self._things()

    def _label(self, labels, x, y):
        m = self.map
        mid = m.grid[y][x]
        l = labels.get(self.decomp.key(m, mid))
        if mid < 640 and mid in PINES:
            return "tree"
        if l:
            parts = set(l.split("+"))
            if "water" in parts:
                return "water"
            hit = parts & OBJECT_LABELS
            return sorted(hit)[0] if hit else "ground"
        beh, _ = self.decomp.attributes(m, mid)
        if beh in WATER_BEHAVIOURS:
            return "water"
        return "object" if m.collision[y][x] else "ground"

    def tile(self, x, y):
        return self.decomp.metatile(self.map, self.map.grid[y][x])

    def ground_colours(self):
        """The colours of the map's ground (its cells labelled ground, as drawn)."""
        if not hasattr(self, "_ground"):
            out = set()
            for y in range(self.H):
                for x in range(self.W):
                    if self.labels[y][x] == "ground":
                        out |= {tuple(c) for c in np.asarray(self.tile(x, y)[0])[:, :, :3].reshape(-1, 3).tolist()}
            self._ground = out
        return self._ground

    def _own_share(self, x, y):
        """How much of a cell is drawn other than the ground: its top layer, or colours the
        ground hasn't."""
        whole, top = (np.asarray(im) for im in self.tile(x, y))
        g = self.ground_colours()
        cs = whole[:, :, :3].reshape(-1, 3).tolist()
        up = top[:, :, 3].reshape(-1).tolist()
        return sum(1 for c, t in zip(cs, up) if t > 0 or tuple(c) not in g) / 256.0

    def _things(self):
        m = self.map
        out = []
        taken = set()
        # (the pines: a tree from each body's left cell, its tip above and its foot below)
        pine = lambda x, y, ids: 0 <= x < self.W and 0 <= y < self.H and m.grid[y][x] < 640 and m.grid[y][x] in ids
        for y in range(self.H):
            for x in range(self.W - 1):
                if not (pine(x, y, PINE_BODIES) and m.grid[y][x] % 2 == 0 and pine(x + 1, y, PINE_BODIES)):
                    continue
                pixels = set()
                for i in (0, 1):
                    pixels |= self._cell(x + i, y)
                    if pine(x + i, y - 1, PINE_TIPS):
                        pixels |= self._cell(x + i, y - 1)
                    elif pine(x + i, y - 1, PINE_JUNCTIONS):
                        pixels |= self._cell(x + i, y - 1, rows=range(8, PX))
                    if pine(x + i, y + 1, PINE_FEET):
                        pixels |= self._cell(x + i, y + 1)
                    elif pine(x + i, y + 1, PINE_JUNCTIONS):
                        pixels |= self._cell(x + i, y + 1, rows=range(0, 8))
                out.append({"class": "tree", "tiles": [(x, y), (x + 1, y)], "pixels": pixels})
                taken |= {(x, y), (x + 1, y)}
                for i in (0, 1):
                    for j in (-1, 1):
                        if pine(x + i, y + j, PINE_TIPS | PINE_FEET):
                            taken.add((x + i, y + j))
                        elif pine(x + i, y + j, PINE_JUNCTIONS):
                            taken.add((x + i, y + j))
        # (a tip with no body in the map (its tree beyond its edge): a tree of its own)
        for y in range(self.H):
            for x in range(self.W - 1):
                if (x, y) not in taken and pine(x, y, PINE_TIPS) and m.grid[y][x] % 2 == 0:
                    out.append({"class": "tree", "tiles": [(x, y), (x + 1, y)],
                                "pixels": self._cell(x, y) | self._cell(x + 1, y)})
                    taken |= {(x, y), (x + 1, y)}
        # (the others: the cells of each kind of thing, together; a cell drawn hardly at all
        # but the ground (a grass patch's corner labelled with the house beside it) is ground)
        cls = {}
        for y in range(self.H):
            for x in range(self.W):
                l = self.labels[y][x]
                if l in OBJECT_LABELS | {"object"} and (x, y) not in taken and l != "tree":
                    if self._own_share(x, y) >= 0.05:
                        cls[(x, y)] = l
                    else:
                        self.labels[y][x] = "ground"
        seen = set()
        for start, c in sorted(cls.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if start in seen:
                continue
            group, todo = set(), [start]
            while todo:
                p = todo.pop()
                if p in group or cls.get(p) != c:
                    continue
                group.add(p)
                todo += [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]
            seen |= group
            out.append({"class": c, "tiles": sorted(group)})
        for t in out:
            t["own"] = self._own(t.get("pixels") or self._cells(t["tiles"]))
            if t["class"] != "tree":
                # (the shadow the GBA draws beside it, a house's down a mailbox's cell, isn't it)
                t["own"] = room._main_pieces(t["own"] - room._drop_shadow(self.sprite, t["own"]))
            xs = [x for x, _ in t["own"]] or [t["tiles"][0][0] * PX]
            ys = [y for _, y in t["own"]] or [t["tiles"][0][1] * PX]
            t["bbox"] = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
            t["columns"] = [min(x for x, _ in t["tiles"]), max(x for x, _ in t["tiles"])]
            t["rows"] = [min(y for _, y in t["tiles"]), max(y for _, y in t["tiles"])]
            t["kind"] = "free"
        return [t for t in out if t["own"]]

    def _cell(self, tx, ty, rows=range(PX)):
        """A cell's pixels (those rows of it)."""
        return {(tx * PX + i, ty * PX + j) for j in rows for i in range(PX)}

    def _cells(self, tiles):
        out = set()
        for (tx, ty) in tiles:
            out |= self._cell(tx, ty)
        return out

    def _own(self, pixels):
        """A thing's own pixels: its cells' pixels but the ground round it, reached from its
        outside through the ground's colours (a gap between two trees too; not a white wall
        that the sand's specks share, walled in by its outline), and never its top layer's."""
        g = self.ground_colours()
        s = self.sprite
        cells = {}
        for (x, y) in pixels:
            cells[(x, y)] = self.tile(x // PX, y // PX)[1].getpixel((x % PX, y % PX))[3] > 0
        back = {p for p, up in cells.items() if not up and s.colour(*p)[:3] in g}
        todo = [p for p in back if any((p[0] + dx, p[1] + dy) not in cells for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
        seen = set()
        while todo:
            p = todo.pop()
            if p in seen or p not in back:
                continue
            seen.add(p)
            todo += [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]
        return set(cells) - seen


def front_rows(sprite, o, gap=3):
    """How many of a thing's rows are its front: those under its roof, the roof read from
    the top down (the rows its top's colour (its upper half's commonest light colour) is
    the commonest of, on as long as it comes back within `gap` rows; not up from the foot,
    where a band of the roof's colour along it (a lab's) would stop it at once)."""
    x0, y0, x1, y1 = o["bbox"]
    s = sprite
    upper = Counter(s.colour(x, y)[:3] for (x, y) in o["own"] if y < (y0 + y1) / 2.0 and not voxel._dark(s.colour(x, y)))
    if not upper:
        return (y1 - y0) // 2
    top = upper.most_common(1)[0][0]
    roofed = []
    for y in range(y0, y1):
        row = Counter(s.colour(x, y)[:3] for x in range(x0, x1) if (x, y) in o["own"])
        if row and row.most_common(1)[0][0] == top:
            roofed.append(y)
    if not roofed:
        return room._front_rows({"sprite": s}, o)
    end = roofed[0]
    for y in roofed[1:]:
        if y - end > gap:
            break
        end = y
    return max(1, y1 - end - 1)


def fingerprint(town, t):
    return room.fingerprint(town.sprite, t)


def _is_water(c):
    h, s, v = colorsys.rgb_to_hsv(*(k / 255.0 for k in c[:3]))
    return 0.5 < h < 0.72 and s > 0.3


def build_voxels(town, fixes=None, outlines=True):
    """The town as cubes, owned by "ground", "water" and "thing-N" (N an index into
    town.things)."""
    fixes = fixes if fixes is not None else room.load_fixes()
    s = town.sprite
    grid = voxel.Grid()
    # the ground: each cell its picture, a thing's the ground most like what's round it
    counts = Counter(town.map.grid[y][x] for y in range(town.H) for x in range(town.W) if town.labels[y][x] == "ground")

    def under(x, y):
        near = Counter()
        for r in (1, 2, 3):
            for j in range(-r, r + 1):
                for i in range(-r, r + 1):
                    if 0 <= x + i < town.W and 0 <= y + j < town.H and town.labels[y + j][x + i] == "ground":
                        near[town.map.grid[y + j][x + i]] += 1
            if near:
                break
        mid = (near or counts).most_common(1)[0][0]
        return town.decomp.metatile(town.map, mid)[0].load()
    for y in range(town.H):
        for x in range(town.W):
            l = town.labels[y][x]
            pic = None if l in ("ground", "water") else under(x, y)
            for j in range(PX):
                for i in range(PX):
                    px, py = x * PX + i, y * PX + j
                    c = pic[i, j][:3] if pic else s.colour(px, py)[:3]
                    if l == "water" and _is_water(c):
                        grid.set(px, -2, py - 1, c, "water")          # (sunk a cube, along the line of sight)
                    else:
                        grid.set(px, -1, py, c, "ground")
    # the things
    turned = {}
    for k, t in enumerate(town.things):
        fix = fixes.get(fingerprint(town, t)) or {}
        owner = "thing-%d" % k
        o = dict(t)
        own = (set(t["own"]) | room.fix_included(s, o, fix)) - room.fix_dropped(o, fix)
        o["own"] = own
        shape = fix.get("shape") or ("round" if t["class"] == "tree" else "box")
        part = voxel.Grid()
        if shape == "round":
            rd = float(fix.get("round_depth", 1.0 if t["class"] == "tree" else 0.5))
            foot = max(v for _, v in own)
            xs = [x for x, _ in own]
            bottom = [x for x, v in own if v >= foot - 2]
            r_foot = (max(bottom) - min(bottom) + 1) / 2.0
            r_all = (max(xs) - min(xs) + 1) / 2.0
            z_axis = foot - r_foot * rd if t["class"] != "tree" else foot - r_all * rd
            voxel.lathe(part, s, own, 0, z_axis, owner, depth=rd)
            turned[k] = voxel.lathe_outlined(s, own, 0)
        elif shape == "slab" or (shape == "box" and t["class"] in ("fence", "sign") and "profile" not in fix):
            # (a fence, a sign: drawn face on, standing, as thick as a post)
            voxel.unproject_slab(part, s, own, 0, owner, thick=int(fix.get("thick", 3)))
        elif shape == "plant":
            room.build_plant(part, s, o, own, fix, owner, recess=room.fix_recess(o, fix), raised=room.fix_raised(s, o, fix))
        else:
            if "profile" in fix or "profiles" in fix:
                kinds = voxel.segments_kinds(fix.get("profile", [["up", "rest"]]), o["bbox"][3])
                if "profiles" in fix:
                    kinds = voxel.columns_kinds(fix["profiles"], kinds, o["bbox"][0], o["bbox"][3])
            else:
                # (its front rows up from its foot, the rest lying: from the thing's foot, not
                # each column's, so a roof's edge past its walls lies with the roof)
                h = int(fix.get("front") or front_rows(s, o))
                kinds = voxel.segments_kinds([["up", h], ["back", "rest"]], o["bbox"][3])
            stretch = float(fix.get("stretch", 1.0))
            voxel.unproject_profile(part, s, own, kinds, owner=owner, recess=room.fix_recess(o, fix),
                                    raised=room.fix_raised(s, o, fix), stretch=stretch, foot=o["bbox"][3])
        for key, c in part.cubes.items():
            grid.set(*key, c, owner)
    # every thing finished as a room's are: its corners, its sides, its outline
    hexc = lambda h: tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    for k, t in enumerate(town.things):
        fix = fixes.get(fingerprint(town, t)) or {}
        owner = "thing-%d" % k
        if "round" in fix:
            voxel.round_corners(grid, owner, int(fix["round"]))
        if "sides" in fix:
            sides = fix["sides"]
            for which, side in (((0, 1), sides), ((0,), sides.get("left")), ((1,), sides.get("right"))):
                if side and "colour" in side:
                    voxel.paint_sides(grid, owner, hexc(side["colour"]), hexc(side["panel"]) if "panel" in side else None, which)
        if outlines and fix.get("outline", True):
            voxel.outline(grid, owner, voxel._darkest(s, t["own"]), keep_shown=k not in turned, where=turned.get(k))
    return grid
