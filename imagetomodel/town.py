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
import hashlib
from collections import Counter

import numpy as np

from . import room, voxel
from .picture import PX, Sprite

OBJECT_LABELS = {"building", "tree", "hedge", "fence", "sign", "prop"}
SINGLE_CELLS = {"fence"}         # (things the game draws a cell at a time, each a model reused)
TREE_DEPTH = 0.4                 # (a tree as deep as wide, this much: round, but thin)
BUILDING_STRETCH = 1.5           # (a building's roof deeper than the GBA draws it)
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

    def _flowers(self):
        """The flowers drawn on the ground (a bed of them): cells labelled ground with many
        bright petals, each a thing standing up; found before the ground's colours are, so
        the petals aren't taken for the ground's."""
        out = []
        for y in range(self.H):
            for x in range(self.W):
                if self.labels[y][x] != "ground":
                    continue
                px = np.asarray(self.tile(x, y)[0])[:, :, :3].reshape(-1, 3) / 255.0
                petals = sum(1 for r, g, b in px.tolist() if _petal(r, g, b))
                if petals >= 10:
                    self.labels[y][x] = "flower"
                    out.append({"class": "flower", "tiles": [(x, y)]})
        return out

    def _things(self):
        m = self.map
        out = self._flowers()
        taken = set()
        # (the pines: one tree, the same model wherever it stands (as the game draws the
        # same tree again and again): its picture from the pines' own tiles alone (a tip's
        # top layer, a body, a foot, less the grass), not from a row of them overlapping or
        # the variants that end one; placed at each body's left cell, or below a tip whose
        # body is past the map's edge)
        pine = lambda x, y, ids: 0 <= x < self.W and 0 <= y < self.H and m.grid[y][x] < 640 and m.grid[y][x] in ids
        has_pines = any(pine(x, y, PINES) for y in range(self.H) for x in range(self.W))
        grass = self.grass() if has_pines else set()
        model = self.pine_picture(grass) if has_pines else {}

        def place(x, y):
            out.append({"class": "tree", "model": "pine", "tiles": [(x, y), (x + 1, y)],
                        "picture": {(x * PX + px, y * PX + py): c for (px, py), c in model.items()}})
        for y in range(self.H):
            for x in range(self.W - 1):
                if not (pine(x, y, PINE_BODIES) and m.grid[y][x] % 2 == 0 and pine(x + 1, y, PINE_BODIES)):
                    continue
                place(x, y)
                taken |= {(x, y), (x + 1, y)}
                for i in (0, 1):
                    for j in (-1, 1):
                        if pine(x + i, y + j, PINE_TIPS | PINE_FEET | PINE_JUNCTIONS):
                            taken.add((x + i, y + j))
        for y in range(self.H):
            for x in range(self.W - 1):
                if (x, y) not in taken and pine(x, y, PINE_TIPS) and m.grid[y][x] % 2 == 0:
                    place(x, y + 1)
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
                if c in SINGLE_CELLS:
                    break                       # (a fence a cell: the game's one tile repeated)
                todo += [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]
            seen |= group
            out.append({"class": c, "tiles": sorted(group)})
        for t in out:
            extra = set()
            if t["class"] == "flower":
                # (a flower's grass: the colours round its cell's edge, a shade of its own)
                (tx, ty), = t["tiles"]
                extra = {self.sprite.colour(tx * PX + i, ty * PX + j)[:3] for i in range(PX) for j in range(PX)
                         if i in (0, PX - 1) or j in (0, PX - 1)}
                extra = {c for c in extra if not _petal(*(k / 255.0 for k in c))}
            if "picture" in t:
                # (a tree's own: its picture's pixels in the map)
                t["own"] = {p for p in t["picture"] if 0 <= p[0] < self.W * PX and 0 <= p[1] < self.H * PX}
            elif t["class"] == "flower":
                # (a flower's own: where its tile differs from its bottom layer, the grass under
                # it: its leaves too, the ground's greens as they are)
                (tx, ty), = t["tiles"]
                whole = np.asarray(self.tile(tx, ty)[0])[:, :, :3]
                under = np.asarray(self.decomp.bottom(m, m.grid[ty][tx]))[:, :, :3]
                t["own"] = {(tx * PX + i, ty * PX + j) for j in range(PX) for i in range(PX)
                            if tuple(whole[j, i]) != tuple(under[j, i])}
            else:
                t["own"] = self._own(self._cells(t["tiles"]), extra, layers=t["class"] != "flower")
            if t["class"] not in ("tree", "flower"):
                # (the shadow the GBA draws beside it, a house's down a mailbox's cell, isn't it)
                t["own"] = room._main_pieces(t["own"] - room._drop_shadow(self.sprite, t["own"]))
            xs = [x for x, _ in t["own"]] or [t["tiles"][0][0] * PX]
            ys = [y for _, y in t["own"]] or [t["tiles"][0][1] * PX]
            t["bbox"] = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
            t["columns"] = [min(x for x, _ in t["tiles"]), max(x for x, _ in t["tiles"])]
            t["rows"] = [min(y for _, y in t["tiles"]), max(y for _, y in t["tiles"])]
            t["kind"] = "free"
            if "model" not in t and t["own"]:
                # (its model's name: what it is and its picture, from its own corner: the same
                # thing drawn again (a house, a mailbox, a flower) the same model)
                x0, y0 = t["bbox"][0], t["bbox"][1]
                h = hashlib.sha1(repr(sorted((x - x0, y - y0, self.sprite.colour(x, y)[:3]) for x, y in t["own"])).encode())
                t["model"] = "%s-%s" % (t["class"], h.hexdigest()[:8])
        return [t for t in out if t["own"]]

    def grass(self):
        """The grass the pines stand in: the colours of a pine tip's bottom layer."""
        b = np.asarray(self.decomp.bottom(self.map, 0x0E))
        return {tuple(c) for c in b[:, :, :3].reshape(-1, 3).tolist()}

    def pine_picture(self, grass):
        """The pine, as one tree: {(x, y): colour} from its body's top left, its tip (a tip
        tile's top layer) the cell above, its foot (a foot tile) the cell below, less the
        grass round it."""
        m = self.map
        pic = {}
        for i in (0, 1):
            tip = self.decomp.metatile(m, 0x0E + i)[1]
            self._paint(pic, i, -1, tip, lambda a, b: True, grass)
            self._paint(pic, i, 0, self.decomp.metatile(m, 0x1C + i)[0], lambda a, b: True, grass)
            self._paint(pic, i, 1, self.decomp.metatile(m, 0x24 + i)[0], lambda a, b: True, grass)
        return pic

    def _paint(self, picture, tx, ty, im, keep, grass):
        """A tile's pixels (those `keep(i, j)` says, not the grass's) into a tree's picture at
        the cell (tx, ty) (which may be past the map's edge)."""
        a = np.asarray(im)
        for j in range(PX):
            for i in range(PX):
                c = tuple(int(k) for k in a[j, i, :3])
                if a[j, i, 3] > 0 and keep(i, j) and c not in grass:
                    picture[(tx * PX + i, ty * PX + j)] = c

    def _cell(self, tx, ty, rows=range(PX)):
        """A cell's pixels (those rows of it)."""
        return {(tx * PX + i, ty * PX + j) for j in rows for i in range(PX)}

    def _cells(self, tiles):
        out = set()
        for (tx, ty) in tiles:
            out |= self._cell(tx, ty)
        return out

    def _own(self, pixels, extra=(), layers=True):
        """A thing's own pixels: its cells' pixels but the ground round it, reached from its
        outside through the ground's colours (a gap between two trees too; not a white wall
        that the sand's specks share, walled in by its outline), and never its top layer's."""
        g = self.ground_colours() | set(extra)
        s = self.sprite
        cells = {}
        for (x, y) in pixels:
            # (a pixel on the top layer is the thing's, but for a flower's tile, the whole of
            # it on its top layer, its grass too)
            cells[(x, y)] = layers and self.tile(x // PX, y // PX)[1].getpixel((x % PX, y % PX))[3] > 0
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


def openings(town, t, own, kinds, depth=1):
    """A building's windows and doors, set in `depth` cubes: on its front (its rows that
    stand), its glass (light blue), and in each door's cell (the map's warps) what isn't
    its wall (its front's two commonest colours) or an outline."""
    if depth <= 0:
        return {}
    s = town.sprite
    front = {(x, v) for (x, v) in own if kinds(x, v) == "up"}
    wall = {c for c, _ in Counter(s.colour(x, v)[:3] for (x, v) in front).most_common(2)}
    doors = {(w["x"], w["y"]) for w in town.map.events.get("warp_events", [])} & set(map(tuple, t["tiles"]))
    out = {}
    for (x, v) in front:
        c = s.colour(x, v)[:3]
        door = (x // PX, v // PX) in doors and c not in wall and not voxel._dark(c)
        if _is_glass(c) or door:
            out[(x, v)] = depth
    return out


def fingerprint(town, t):
    return room.fingerprint(town.sprite, t)


def _petal(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return s > 0.45 and v > 0.5 and (h < 0.12 or h > 0.88 or 0.12 <= h < 0.18)


def _is_glass(c):
    h, s, v = colorsys.rgb_to_hsv(*(k / 255.0 for k in c[:3]))
    return 0.5 < h < 0.72 and s > 0.3 and v > 0.5


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

    trees = any(town.labels[y][x] == "tree" for y in range(town.H) for x in range(town.W))
    tree_ground = town.decomp.bottom(town.map, 0x0E).load() if trees else None
    g_colours = town.ground_colours()

    def under(x, y):
        # (the ground the game draws under it: a tree's, the grass; a thing drawn on its
        # tile's top layer (a flower), its tile's bottom layer; else the ground most like
        # what's round it)
        if town.labels[y][x] == "tree":
            return tree_ground
        b = np.asarray(town.decomp.bottom(town.map, town.map.grid[y][x]))
        cs = [tuple(c) for c in b[:, :, :3].reshape(-1, 3).tolist()]
        if (b[:, :, 3] > 0).all() and sum(1 for c in cs if c in g_colours) >= 0.9 * len(cs):
            return town.decomp.bottom(town.map, town.map.grid[y][x]).load()
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
        shape = fix.get("shape") or ("billboard" if t["class"] in ("tree", "flower") else "box")
        part = voxel.Grid()
        if shape == "billboard":
            # (a tree, a flower: its picture standing facing the front, as the GBA's sprites
            # are, a few cubes thick)
            pic = t.get("picture") or {p: s.colour(*p)[:3] for p in own}
            voxel.billboard(part, pic, thick=int(fix.get("thick", 3 if t["class"] == "tree" else 2)), owner=owner)
        elif shape == "round":
            rd = float(fix.get("round_depth", TREE_DEPTH if t["class"] == "tree" else 0.5))
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
            building = t["class"] == "building"
            stretch = float(fix.get("stretch", BUILDING_STRETCH if building else 1.0))
            inset = openings(town, t, own, kinds, int(fix.get("inset", 1))) if building else None
            voxel.unproject_profile(part, s, own, kinds, owner=owner, recess=room.fix_recess(o, fix),
                                    raised=room.fix_raised(s, o, fix), stretch=stretch, foot=o["bbox"][3],
                                    inset=inset)
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
