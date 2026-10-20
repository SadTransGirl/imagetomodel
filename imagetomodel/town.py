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
BILLBOARDS = {"tree": 3, "flower": 2, "fence": 3, "sign": 3}   # (drawn face on: standing so, this thick)
SINGLE_CELLS = {"fence"}         # (things the game draws a cell at a time, each a model reused)
TREE_DEPTH = 0.4                 # (a tree as deep as wide, this much: round, but thin)
BUILDING_STRETCH = 1.5           # (a flat-roofed building deeper than the GBA draws it)
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
        return clean_tree(pic)

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


def square_walls(part, sprite, own, kinds, shown, owner):
    """A building's walls a box: under its roof, as wide as its front stands at its foot,
    from its front back to its back (the roof's rounded corners as drawn don't round its
    walls)."""
    base = [(x, y, z) for (x, v), (y, z) in shown.items() if kinds(x, v) == "up" and (x, v) in own]
    if not base:
        return
    height = max(y for _, y, _ in base) + 1
    foot = [x for x, y, _ in base if y == 0]
    if not foot:
        return
    x0, x1 = min(foot), max(foot)
    # (a roof's eave past its walls stands on nothing: what's under its lowest cube drawn,
    # filled down to the ground, taken away)
    low = {}
    for (x, v), (y, _) in shown.items():
        if (x, v) in own and not x0 <= x <= x1:
            low[x] = min(low.get(x, y), y)
    for key in [k for k in part.cubes if k[0] in low and k[1] < low[k[0]]]:
        del part.cubes[key]
        part.owner.pop(key, None)
    z_back = min(z for _, _, z in part.cubes)
    fronts = {}
    for x, y, z in base:
        fronts[(x, y)] = max(fronts.get((x, y), z), z)
    body = Counter(c for (x, y, z), c in part.cubes.items() if y < height and not voxel._dark(c)).most_common(1)
    if not body:
        return
    roof = {}                                 # (its roof's level: the nearest roof cube's colour)
    for (x, y, z), c in part.cubes.items():
        if y == height - 1 and (x, y + 1, z) not in part.cubes:
            roof.setdefault(z, []).append((x, c))
    for x in range(x0, x1 + 1):
        for y in range(height):
            zf = fronts.get((x, y))
            if zf is None:
                continue
            for z in range(z_back, zf):
                if (x, y, z) in part.cubes:
                    continue
                c = body[0][0]
                if y == height - 1:
                    near = roof.get(z) or roof.get(min(roof, key=lambda k: abs(k - z))) if roof else None
                    if near:
                        c = min(near, key=lambda r: abs(r[0] - x))[1]
                part.set(x, y, z, c, owner)


def roof_drops(sprite, own, split, hip):
    """How far a roof's ends come down, as the GBA draws them: {x: rows} for each column
    within `hip` of either end, by how much lower its eave is drawn than the middle's (the
    eave's rows (its gutter, its trim) found again down the column), so its ends' stripes
    slope down to them as the picture's eave does."""
    xs = [x for x, v in own if v == split]
    if not xs:
        return {}
    x0, x1 = min(x for x, _ in own), max(x for x, _ in own)
    cx = (min(xs) + max(xs)) // 2
    eave = [sprite.colour(cx, split + i)[:3] for i in range(4)]
    drops = {}
    for x in list(range(x0, x0 + hip)) + list(range(x1 - hip + 1, x1 + 1)):
        best, at = -1, 0
        for k in range(hip):
            same = sum(1 for i in range(4) if (x, split + k + i) in own and sprite.colour(x, split + k + i)[:3] == eave[i])
            if same > best:
                best, at = same, k
        if best >= 2:
            drops[x] = at
    # (a column the eave isn't found down (its outline) as its nearest neighbour's)
    for x in list(range(x0, x0 + hip)) + list(range(x1 - hip + 1, x1 + 1)):
        if x not in drops and drops:
            drops[x] = drops[min(drops, key=lambda q: abs(q - x))]
    return drops


def roof_ends(part, owner, drops):
    """A roof's ends (their columns come down, so drawn less deep) carried back as deep as
    its middle: each column's cubes at its back again behind it."""
    zb = min(z for _, _, z in part.cubes)
    for x in drops:
        col = [(y, z, c) for (xx, y, z), c in part.cubes.items() if xx == x]
        if not col:
            continue
        back = min(z for _, z, _ in col)
        for y, z, c in col:
            if z == back:
                for zz in range(zb, back):
                    part.set(x, y, zz, c, owner)


def period(rows):
    """The shortest run of rows a picture repeats in ([colour, ...] a row), judged on its
    middle half (past its rims): a roof's tiles, a course of its shingles; None if it
    doesn't (as near as its best)."""
    n = len(rows)
    mid = range(n // 4, n - n // 4)
    score = {}
    for p in range(2, n // 2 + 1):
        pairs = [(rows[i], rows[i + p]) for i in mid if i + p < n]
        total = sum(len(r) for r, _ in pairs)
        if total:
            score[p] = sum(sum(1 for a, b in zip(r, q) if a == b) for r, q in pairs) / float(total)
    if not score:
        return None
    best = max(score.values())
    if best < 0.75:
        return None
    return min(p for p, sc in score.items() if sc >= best - 0.02)


def deepen(part, sprite, own, kinds, owner, factor):
    """A building `factor` times as deep as the GBA draws it, its roof's pattern carried on
    (whole tiles of it, so they stay square, where the stretch of a row at a time draws
    every other row twice): where its roof is plainest, near its middle, as many layers as
    its pattern repeats in, again and again, everything behind moved back."""
    lying = [(x, v) for (x, v) in own if kinds(x, v) != "up"]
    if not lying or factor <= 1:
        return
    x0, x1 = min(x for x, _ in lying), max(x for x, _ in lying)
    v0, v1 = min(v for _, v in lying), max(v for _, v in lying)
    rows = [[sprite.colour(x, v)[:3] if (x, v) in own else None for x in range(x0, x1 + 1)] for v in range(v0, v1 + 1)]
    p = period(rows)
    if not p:
        return
    extra = p * max(1, int(round((factor - 1) * len(rows) / float(p))))
    tops = {}
    for (x, y, z) in part.cubes:
        tops[(x, z)] = max(tops.get((x, z), -1), y)
    level = Counter(tops.values()).most_common(1)[0][0]
    zs = sorted({z for _, z in tops})
    plain = {z for z in zs if all(tops.get((x, z), -1) <= level for x in range(x0, x1 + 1))}
    roof = [z for z in zs if sum(1 for x in range(x0, x1 + 1) if tops.get((x, z)) == level) > (x1 - x0) // 2]
    if not roof:
        return
    mid = (roof[0] + roof[-1]) / 2.0
    cuts = [z for z in roof if all(z + i in plain for i in range(p))]
    if not cuts:
        return
    cut = min(cuts, key=lambda z: abs(z + p / 2.0 - mid))
    # (behind the cut moved back; the layers in front of it, cut to cut + p, again behind it)
    moved = {}
    for (x, y, z), c in part.cubes.items():
        if z < cut:
            moved[(x, y, z - extra)] = c
        else:
            moved[(x, y, z)] = c
            if z < cut + p:
                for k in range(1, extra // p + 1):
                    moved[(x, y, z - k * p)] = c
    part.cubes = {}
    part.owner = {}
    for key, c in moved.items():
        part.set(*key, c, owner)


def outline_colour(sprite, own):
    """The colour a thing's picture is outlined in: the commonest dark colour where its own
    pixels meet the outside (a house's navy, not its roof's darkest red)."""
    edge = Counter()
    for (x, y) in own:
        if any((x + dx, y + dy) not in own for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            c = sprite.colour(x, y)[:3]
            if 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] < 110:
                edge[c] += 1
    return edge.most_common(1)[0][0] if edge else voxel._darkest(sprite, own)


def wrap_sides(part, sprite, own, kinds, shown, inset):
    """A building's front carried round its sides and its back, row by row as the GBA
    draws its front (its siding's stripes, its bricks' joints), but for its glass, its door
    and its outlines: each side's cubes at a height take its front's wall at that height,
    going on round from the corner; the back's, from its left. Not the cubes the GBA's view
    shows, nor its top (its roof)."""
    rows = {}
    for (x, v), (y, z) in shown.items():
        if kinds(x, v) == "up" and (x, v) not in inset:
            rows.setdefault(y, []).append((x, z, sprite.colour(x, v)[:3]))
    if not rows:
        return
    # (its wall: its front's commonest colours (not its glass, not its outlines); at each
    # height the longest run of them unbroken along the front, between its windows)
    counts = Counter(c for r in rows.values() for _, _, c in r if not voxel._dark(c) and not _is_glass(c))
    total = sum(counts.values()) or 1
    wall = {c for c, n in counts.most_common(4) if n >= 0.04 * total}
    seq = {}
    for y, r in rows.items():
        best, run = [], []
        last = None
        for x, _, c in sorted(r):
            if c in wall and (last is None or x == last + 1):
                run.append(c)
            else:
                run = [c] if c in wall else []
            last = x
            if len(run) > len(best):
                best = list(run)
        if best:
            seq[y] = best
    if not seq:
        return
    front = {y: max(z for _, z, _ in r) for y, r in rows.items()}
    keep = voxel.gba_shown(set(part.cubes))
    ends, backs = {}, {}
    for (x, y, z) in part.cubes:
        lo, hi = ends.get((y, z), (x, x))
        ends[(y, z)] = (min(lo, x), max(hi, x))
        backs[(x, y)] = min(backs.get((x, y), z), z)
    left = {y: min(x for x, _, _ in r) for y, r in rows.items()}
    # (a height with no wall of its own (all glass, a sill) takes the nearest that has)
    known = sorted(seq)
    for y in rows:
        if y not in seq:
            seq[y] = seq[min(known, key=lambda k: abs(k - y))]
    for key in list(part.cubes):
        x, y, z = key
        if key in keep or y not in seq or (x, y + 1, z) not in part.cubes:
            continue                                    # (its top's a roof, not a wall)
        line = seq[y]
        lo, hi = ends[(y, z)]
        d = max(0, front[y] - z)
        if x == lo:
            part.cubes[key] = line[d % len(line)]
        elif x == hi:
            part.cubes[key] = line[::-1][d % len(line)]
        elif z == backs[(x, y)]:
            part.cubes[key] = line[(x - left[y]) % len(line)]


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


def clean_tree(pic):
    """A tree's picture without what the GBA draws round it: the glow round its edge (a
    colour only on its edge), its outline (its darkest colour, where it meets the outside),
    and its shadow on the ground (below its crown's last light row, all but its trunk, the
    trunk carried up to the crown)."""
    pic = dict(pic)
    edge = lambda p: any((p[0] + dx, p[1] + dy) not in pic for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    sat = lambda c: colorsys.rgb_to_hsv(*(k / 255.0 for k in c))[1]
    # (a colour only on the edge: the glow)
    on_edge = Counter(c for p, c in pic.items() if edge(p))
    inside = Counter(c for p, c in pic.items() if not edge(p))
    for c in [c for c in on_edge if on_edge[c] > 3 * inside.get(c, 0)]:
        pic = {p: k for p, k in pic.items() if k != c}
    # (the outline: its darkest green where it meets the outside)
    greens = [c for c in set(pic.values()) if sat(c) > 0.3]
    if greens:
        dark = min(greens, key=lum)
        pic = {p: c for p, c in pic.items() if not (c == dark and edge(p))}
    # (the shadow: below the crown's light, but its trunk (not green))
    light = sorted(set(pic.values()), key=lum)[-2:]
    crown = max((y for (x, y), c in pic.items() if c in light), default=None)
    if crown is not None:
        trunk = {p: c for p, c in pic.items() if p[1] > crown and sat(c) <= 0.3}
        pic = {p: c for p, c in pic.items() if p[1] <= crown or p in trunk}
        if trunk:
            xs = {x for x, _ in trunk}
            top = min(y for _, y in trunk)
            core = Counter(trunk.values()).most_common(1)[0][0]
            for x in xs:
                for y in range(crown + 1, top):
                    pic[(x, y)] = core
    return pic


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
        shape = fix.get("shape") or ("billboard" if t["class"] in BILLBOARDS else "box")
        part = voxel.Grid()
        if shape == "billboard":
            # (a tree, a flower: its picture standing facing the front, as the GBA's sprites
            # are, a few cubes thick)
            pic = t.get("picture") or {p: s.colour(*p)[:3] for p in own}
            voxel.billboard(part, pic, thick=int(fix.get("thick", BILLBOARDS[t["class"]])), owner=owner)
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
        elif shape == "slab":
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
            hipped = building and isinstance(fix.get("roof"), dict) and "hip" in fix["roof"]
            drops = {}
            if hipped:
                # (a hipped roof as the GBA draws it: lying on its walls as drawn, its ends'
                # columns coming down as far as their eave is drawn lower, so they slope)
                split = o["bbox"][3] - h
                drops = roof_drops(s, own, split, int(fix["roof"]["hip"]))
                kinds = (lambda split, drops: lambda x, v: "up" if v >= split + drops.get(x, 0) else "back")(split, drops)
            # (a building's rows lie as drawn; it's made deeper after, its roof's pattern carried
            # on (a hipped roof sets its own depth))
            stretch = float(fix.get("stretch", 1.0))
            inset = openings(town, t, own, kinds, int(fix.get("inset", 1))) if building else None
            # (its parts, each a region built by a profile of its own (a chimney: a box on the
            # roof, the roof behind it still the roof), and the rest by the thing's)
            x0, y0 = o["bbox"][0], o["bbox"][1]
            rest = set(own)
            groups = []
            for prt in fix.get("parts", []):
                pix = {(x, v) for (x, v) in rest if prt["rows"][0] <= v - y0 <= prt["rows"][1]
                       and prt["cols"][0] <= x - x0 <= prt["cols"][1]}
                rest -= pix
                groups.append((pix, voxel.segments_kinds(prt["profile"], o["bbox"][3])))
            groups.append((rest, kinds))
            shown = {}
            for pix, kd in groups:
                if pix:
                    shown.update(voxel.unproject_profile(part, s, pix, kd, owner=owner, recess=room.fix_recess(o, fix),
                                                         raised=room.fix_raised(s, o, fix), stretch=stretch,
                                                         foot=o["bbox"][3], inset=inset))
            if building:
                square_walls(part, s, own, kinds, shown, owner)
            if hipped:
                roof_ends(part, owner, drops)
            elif building:
                deepen(part, s, own, kinds, owner, float(fix.get("deepen", BUILDING_STRETCH)))
            if building and fix.get("wrap", True):
                wrap_sides(part, s, own, kinds, shown, inset or {})
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
            voxel.outline(grid, owner, outline_colour(s, t["own"]), keep_shown=k not in turned, where=turned.get(k))
    return grid
