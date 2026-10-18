"""A room from a screenshot of it (the GBA's own picture, or art drawn its way): cut on its
16-pixel grid, each tile told from its picture alone (the void beyond the room, the floor,
the back wall and the things hanging on it, the walls down the sides, the things standing
in the room), and built as the port builds rooms: the floor flat; the back wall standing,
its rows a pixel a pixel; a wall down each side; each thing a box, its front standing at
its foot and its top lying back over it (a thing against the back wall standing tall), the
picture staying exactly where the GBA drew it, seen from the GBA's angle. Heights are as
they'd be unsquashed: the game squashes rooms as it builds them (Models.build does the
same to a model), so a row of a picture here is 16 pixels standing."""
import hashlib
from collections import Counter

import numpy as np

from .geometry import Mesh
from .picture import PX, hsv

LABELS = ("void", "floor", "wall", "wall_side", "thing", "box", "counter")   # (box, counter: a thing, its kind forced)


def _tile(sprite, ox, oy, tx, ty):
    return sprite.rgba[oy + ty * PX:oy + ty * PX + PX, ox + tx * PX:ox + tx * PX + PX]


def grid(sprite):
    """Where the tile grid lies on the picture: the offset (0 to 15 each way) at which the
    most tiles repeat (a screen scrolled with the player); then the tiles across and down."""
    best = None
    for oy in range(PX):
        for ox in range(PX):
            cols = (sprite.W - ox) // PX
            rows = (sprite.H - oy) // PX
            if cols < 1 or rows < 1:
                continue
            seen = Counter()
            partial = 0                               # (tiles half on the void: the room's edge off the grid)
            for ty in range(rows):
                for tx in range(cols):
                    d = sprite.drawn[oy + ty * PX:oy + ty * PX + PX, ox + tx * PX:ox + tx * PX + PX]
                    share = d.mean()
                    if 0.02 < share < 0.98:
                        partial += 1
                    if share < 0.5:
                        continue                      # (the void's tiles all repeat: not counted)
                    seen[hashlib.md5(_tile(sprite, ox, oy, tx, ty).tobytes()).hexdigest()] += 1
            # (a floor's pattern repeats every few pixels, so its tiles repeat at the wrong
            # offsets too, and a plain wall's rows at any offset down: the right offset is
            # where the most tiles repeat and the most different ones do, together)
            kinds = sum(1 for n in seen.values() if n >= 2)
            repeats = sum(n - 1 for n in seen.values())
            key = (kinds + repeats - partial, sum(seen.values()), -(ox + oy))
            if best is None or key > best[0]:
                best = (key, ox, oy, cols, rows)
    _, ox, oy, cols, rows = best
    return {"ox": ox, "oy": oy, "cols": cols, "rows": rows}


def mark_void(sprite, threshold=0.08):
    """The black beyond a room (the GBA draws it black) is not the room: cleared."""
    rgb = sprite.rgba[:, :, :3].astype(int)
    black = rgb.max(axis=2) <= int(threshold * 255)
    sprite.drawn &= ~black
    sprite.rgba[black] = 0


def _colours(tile, drawn):
    """A tile's drawn colours, counted."""
    c = Counter()
    px = tile[:, :, :3][drawn]
    for col in map(tuple, px.tolist()):
        c[col] += 1
    return c


def _alike(c, colours):
    """Whether a colour is one of the floor's, or a shade of one (the floor in a shadow, or
    under a lamp's glow): the same hue (within 20 degrees), as saturated (within 0.15), and
    lighter or darker in every channel alike; or near white, lighter than one."""
    if c in colours:
        return True
    h, s, v = hsv(c)
    for f in colours:
        lighter = all(c[k] >= f[k] - 4 for k in range(3))
        darker = all(c[k] <= f[k] + 4 for k in range(3))
        if not (lighter or darker):
            continue
        if s < 0.08 and v > 0.95 and lighter:
            return True
        fh, fs, _ = hsv(f)
        dh = abs(h - fh)
        if min(dh, 360 - dh) <= 20 and abs(s - fs) <= 0.15:
            return True
    return False


def _dark(c):
    return hsv(c)[2] < 0.45


def label(sprite, g, spec=None):
    """Each tile's label, and what was learnt: the wall's rows, the floor's and the wall's
    colours. A spec's "tiles" ({"x,y": label}) and "wall_rows" win."""
    spec = dict(spec or {})
    ox, oy, cols, rows = g["ox"], g["oy"], g["cols"], g["rows"]
    tiles = {}
    spr = sprite
    counts = {}
    hashes = {}
    for ty in range(rows):
        for tx in range(cols):
            t = _tile(sprite, ox, oy, tx, ty)
            d = sprite.drawn[oy + ty * PX:oy + ty * PX + PX, ox + tx * PX:ox + tx * PX + PX]
            tiles[(tx, ty)] = (t, d)
            counts[(tx, ty)] = _colours(t, d)
            hashes[(tx, ty)] = hashlib.md5(t.tobytes()).hexdigest() if d.all() else None
    drawn_share = {k: float(d.mean()) for k, (t, d) in tiles.items()}
    # the floor: the tile repeated most (whole tiles only), its colours; a tile mostly of
    # colours like them (the floor under a lamp's glow is lighter) is floor
    whole = Counter(h for h in hashes.values() if h)
    modal = whole.most_common(1)[0][0] if whole else None
    floor_colours = set()
    for k, h in hashes.items():
        if h == modal:
            floor_colours = set(counts[k])
            break
    floor_share = {}
    for k, c in counts.items():
        n = sum(c.values())
        floor_share[k] = sum(m for col, m in c.items() if _alike(col, floor_colours)) / n if n else 0.0
    # the wall's rows: from the first row drawn, while the row's tiles are mostly of the
    # colours of its plain tile (the one repeated most in the row), not the floor's
    wall_colours = set()
    wall_rows = 0
    first = 0
    while first < rows and all(drawn_share[(tx, first)] < 0.1 for tx in range(cols)):
        first += 1
    for ty in range(first, rows):
        row_hashes = Counter(hashes[(tx, ty)] for tx in range(cols) if hashes[(tx, ty)])
        if not row_hashes:
            break
        plain, repeats = row_hashes.most_common(1)[0]
        if plain == modal or repeats < 2:
            break                                     # (the row's plain tile is the floor's, or it has none)
        plain_colours = set(next(counts[k] for k, h in hashes.items() if h == plain))
        wally = 0
        for tx in range(cols):
            c = counts[(tx, ty)]
            n = sum(c.values())
            if n == 0:
                continue
            share = sum(m for col, m in c.items() if col in plain_colours) / n
            if share >= 0.4:
                wally += 1
        if wally * 2 < sum(1 for tx in range(cols) if drawn_share[(tx, ty)] >= 0.1):
            break
        wall_colours |= plain_colours
        wall_rows = ty + 1 - first
    wall_rows = int(spec.get("wall_rows", wall_rows))
    wall_last = first + wall_rows - 1
    # the walls down the sides: in the outermost drawn columns, the tiles like the column's
    # plain tile, which isn't the floor
    side = set()
    for tx in (0, cols - 1):
        ty = wall_last + 1
        while ty < rows and drawn_share[(tx, ty)] >= 0.1 and floor_share[(tx, ty)] < 0.8 and \
                any(col in wall_colours for col in counts[(tx, ty)]):
            side.add((tx, ty))
            ty += 1
        col_hashes = Counter(hashes[(tx, ty)] for ty in range(wall_last + 1, rows) if hashes[(tx, ty)])
        if not col_hashes:
            continue
        plain = col_hashes.most_common(1)[0][0]
        plain_at = next(k for k, h in hashes.items() if h == plain)
        plain_colours = set(counts[plain_at])
        if plain == modal or col_hashes[plain] < 2 or floor_share[plain_at] >= 0.5:
            continue                                  # (the column is the floor, or a shade of it)
        for ty in range(wall_last + 1, rows):
            c = counts[(tx, ty)]
            n = sum(c.values())
            if n and sum(m for col, m in c.items() if col in plain_colours) / n >= 0.6:
                side.add((tx, ty))
    labels = [[None] * cols for _ in range(rows)]
    for ty in range(rows):
        for tx in range(cols):
            k = (tx, ty)
            c = counts[k]
            n = sum(c.values())
            if drawn_share[k] < 0.1:
                lab = "void"
            elif k in side:
                lab = "wall_side"
            elif first <= ty <= wall_last:
                share = sum(m for col, m in c.items() if col in wall_colours) / n
                lab = "wall" if share >= 0.85 or drawn_share[k] < 0.75 or tx in (0, cols - 1) else "thing"
            elif floor_share[k] >= 0.8 or drawn_share[k] < 0.75:
                lab = "floor"                         # (a door mat on the void's edge, a corner cut: flat)
            else:
                lab = "thing"
            labels[ty][tx] = lab
    forced = {}
    for key, lab in (spec.get("tiles") or {}).items():
        x, y = (int(v) for v in key.split(","))
        if 0 <= x < cols and 0 <= y < rows and lab in LABELS:
            if lab in ("box", "counter"):
                forced[(x, y)] = lab
                lab = "thing"
            labels[y][x] = lab
    # a plain wall tile's column, for the sides of walls and boxes to wear
    plain_wall = None
    for ty in range(first, wall_last + 1):
        for tx in range(cols):
            if labels[ty][tx] == "wall" and hashes[(tx, ty)]:
                plain_wall = (tx, ty)
                break
        if plain_wall:
            break
    return {"labels": labels, "wall_first": first, "wall_rows": wall_rows,
            "floor_colours": ["%02x%02x%02x" % c for c in sorted(floor_colours)],
            "wall_colours": ["%02x%02x%02x" % c for c in sorted(wall_colours)],
            "plain_wall": list(plain_wall) if plain_wall else None, "counts": counts, "forced": forced,
            "sprite": spr, "floor_share": floor_share, "plain_floor": next((list(k) for k, h in hashes.items() if h == modal), None)}


def _same_hue(a, b):
    ha, sa, _ = hsv(a)
    hb, sb, _ = hsv(b)
    if sa < 0.12 or sb < 0.12:
        return sa < 0.12 and sb < 0.12               # (a grey with a grey)
    dh = abs(ha - hb)
    return min(dh, 360 - dh) <= 20


def _family(a, b):
    """Whether two tiles, one over the other, look like one thing: one of the two commonest
    colours of each (outlines aside) of the same hue (a counter's top and its darker front;
    not a yellow cushion over a blue one, though both have a grey foot)."""
    def top(c):
        return [col for col, _ in c.most_common() if not _dark(col)][:2]
    return any(_same_hue(x, y) for x in top(a) for y in top(b))


def _main_colour(c):
    for col, _ in c.most_common():
        if not _dark(col):
            return col
    return None


def _outline_pixels(sprite, g, tx, ty):
    """A tile's dark pixels that are a counter's edges, not a thing's own outline: in runs
    across or down most of the tile, or in rows mostly dark."""
    ox, oy = g["ox"], g["oy"]
    out = set()
    x0, y0 = ox + tx * PX, oy + ty * PX
    dark = [[(sprite.colour(x, y) is not None and _dark(sprite.colour(x, y))) for x in range(x0, x0 + PX)] for y in range(y0, y0 + PX)]
    for j in range(PX):
        row = [i for i in range(PX) if dark[j][i]]
        if len(row) >= 10:
            out.update((x0 + i, y0 + j) for i in row)
        run = []
        for i in range(PX + 1):
            if i < PX and dark[j][i]:
                run.append(i)
            else:
                if len(run) >= 12:
                    out.update((x0 + i_, y0 + j) for i_ in run)
                run = []
    for i in range(PX):
        run = []
        for j in range(PX + 1):
            if j < PX and dark[j][i]:
                run.append(j)
            else:
                if len(run) >= 12:
                    out.update((x0 + i, y0 + j_) for j_ in run)
                run = []
    return out


def _background(info, tx, ty, bands=None):
    """The colours round a thing in its tile that aren't it: the floor's shades, the
    wall's colours on the wall's rows, and the band's colour where it stands on a counter."""
    floor_colours = {tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in info["floor_colours"]}
    wall_colours = {tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in info["wall_colours"]}
    wall_last = info["wall_first"] + info["wall_rows"] - 1
    back = set()
    for c in info["counts"][(tx, ty)]:
        if _alike(c, floor_colours) or (ty <= wall_last and c in wall_colours):
            back.add(c)
    if bands and (tx, ty) in bands:
        back.add(bands[(tx, ty)])
    return back


def _is_cutout(info, g, tx, a, b, bands, outlines=False):
    """Whether a run is a thing standing on its own on a background (a plant, a flower):
    its own pixels at least as tall as wide, clear of the tile's sides, on a background of
    a fifth of its tiles or more."""
    ox, oy = g["ox"], g["oy"]
    sprite = info["sprite"]
    xs, ys, own, total, saturated = [], [], 0, 0, 0
    for ty in range(a, b + 1):
        back = _background(info, tx, ty, bands)
        edges = _outline_pixels(sprite, g, tx, ty) if outlines else set()
        for y in range(oy + ty * PX, oy + ty * PX + PX):
            for x in range(ox + tx * PX, ox + tx * PX + PX):
                c = sprite.colour(x, y)
                if c is None:
                    continue
                total += 1
                if c not in back and (x, y) not in edges:
                    own += 1
                    if hsv(c)[1] > 0.25:
                        saturated += 1
                    xs.append(x - ox - tx * PX)
                    ys.append(y)
    if not xs or total == 0 or own * 5 > total * 4:
        return False
    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1
    if min(xs) < 1 or max(xs) > PX - 2:
        return False
    if outlines:
        # (on a counter: a flower, a sign stand; a sheet of paper lies: pale)
        return height >= 0.6 * width and saturated * 5 >= own * 2
    return height >= width


def runs(info, g):
    """The things, column by column: each a run of thing tiles that look alike, with how it
    stands: "on-wall" (hanging on the back wall: painted on it), "against-wall" (standing
    tall from the wall, as a shelf or a machine), "free" (a box in the room). A thing on the
    wall's rows with a thing under it is one with it (a plant and its pot). A run ends over
    a band of things alike right across (a counter) when the tile above isn't of it."""
    labels = info["labels"]
    counts = info["counts"]
    cols, rows = g["cols"], g["rows"]
    wall_last = info["wall_first"] + info["wall_rows"] - 1
    forced = info.get("forced", {})

    floor_colours = {tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in info["floor_colours"]}

    def top3(tx, ty):
        return [col for col, _ in counts[(tx, ty)].most_common() if not _dark(col)][:3]

    bands = {}                                        # ((tx, ty): the band's colour)
    for ty in range(rows):
        candidates = {c for tx in range(cols) if labels[ty][tx] == "thing" for c in top3(tx, ty)[:1]}
        for colour in candidates:
            if colour in floor_colours:
                continue
            run = []
            for tx in range(cols + 1):
                if tx < cols and labels[ty][tx] == "thing" and colour in top3(tx, ty):
                    run.append(tx)
                    continue
                if len(run) >= 3:
                    for x in run:
                        bands.setdefault((x, ty), colour)
                run = []

    info["bands"] = bands

    def band_colour(tx, ty):
        return bands.get((tx, ty))

    def n_rows(a, b):
        return b - a + 1

    def joined(tx, a, b):
        # (whether the tile b, under a, is part of the same thing)
        if a <= wall_last < b:
            return True
        return _family(counts[(tx, a)], counts[(tx, b)])
    floor_share = info["floor_share"]
    found = []
    for tx in range(cols):
        ty = 0
        while ty < rows:
            if labels[ty][tx] != "thing":
                ty += 1
                continue
            a = ty
            b = ty
            shared = False
            # (a floor tile with something drawn on it, right under a thing on the wall's
            # rows: the thing's foot, a plant's pot)
            if b == wall_last and b + 1 < rows and labels[b + 1][tx] == "floor" and floor_share[(tx, b + 1)] < 0.95:
                labels[b + 1][tx] = "thing"
            while b + 1 < rows and labels[b + 1][tx] == "thing" and joined(tx, b, b + 1) \
                    and forced.get((tx, b)) == forced.get((tx, b + 1)):
                if a <= wall_last and b + 1 > wall_last and band_colour(tx, b + 1) is not None \
                        and band_colour(tx, b + 1) not in top3(tx, b):
                    b += 1                            # (its foot on the counter's back edge)
                    shared = True
                    break
                b += 1
            if b <= wall_last:
                kind = "on-wall"
            elif a <= wall_last:
                kind = "against-wall"
            else:
                kind = "free"
            f = forced.get((tx, a))
            if f == "box":
                kind = "against-wall" if a <= wall_last else "tall"
            elif f == "counter":
                kind = "free"
            thing = {"x": tx, "rows": [a, b], "kind": kind}
            if f is None and kind != "on-wall":
                thing["cutout"] = _is_cutout(info, g, tx, a, b, bands)
            found.append(thing)
            # (a thing drawn on a counter's top, its own pixels tall and narrow on the
            # band's colour: a flower, a sign: standing on the top, at its front edge)
            if kind == "free" and n_rows(a, b) >= 2 and (tx, a) in bands and not thing["cutout"] \
                    and _is_cutout(info, g, tx, a, a, bands, outlines=True):
                found.append({"x": tx, "rows": [a, a], "kind": "on-band", "cutout": True, "under": len(found) - 1})
            ty = b if shared else b + 1
    return found


def _mean_colour(counts, dark_too=False):
    r = g = b = n = 0
    for c, m in counts.items():
        if not dark_too and _dark(c):
            continue
        r += c[0] * m
        g += c[1] * m
        b += c[2] * m
        n += m
    if n == 0:
        return _mean_colour(counts, True) if not dark_too and counts else (128, 128, 128)
    return (r // n, g // n, b // n)


def _shade(c, f):
    return tuple(max(0, min(255, int(round(v * f)))) for v in c)


def build(sprite, g, info, things, front_height=16.0, low_height=8.0, wall_thick=4.0, spec=None):
    """The room's mesh, in the picture's pixels (x across, y up, z down the picture); each
    polygon named for the object it belongs to ("floor", "wall", "side-left", "thing-3").
    The walls are thin, their top band plain; a thing is a box with its sides its own
    colour, or, standing on its own on a background, a cutout; what stands behind a
    cutout (the wall, the floor, a counter's top) wears a plain tile, so the thing isn't
    drawn twice."""
    spec = dict(spec or {})
    ox, oy, cols, rows = g["ox"], g["oy"], g["cols"], g["rows"]
    labels = info["labels"]
    counts = info["counts"]
    first = info["wall_first"]
    wall_last = first + info["wall_rows"] - 1
    wall_h = info["wall_rows"] * PX
    wall_z = oy + (wall_last + 1) * PX                 # (the back wall's face)
    thick = float(spec.get("wall_thick", wall_thick))
    mesh = Mesh(strip_v=sprite.H)

    def px(tx, ty):
        return ox + tx * PX, oy + ty * PX

    def tile_uv(at, tx, ty, x, z):
        """The picture of tile `at` where a point of tile (tx, ty) would be drawn."""
        ax, ay = px(*at)
        x0, z0 = px(tx, ty)
        return (ax + (x - x0), ay + (z - z0))
    plain_wall = tuple(info["plain_wall"]) if info["plain_wall"] else None
    plain_floor = tuple(info["plain_floor"]) if info.get("plain_floor") else None
    wall_colour = _mean_colour(counts[plain_wall]) if plain_wall else (200, 190, 170)
    band_tile = (plain_wall[0], first) if plain_wall else None
    band_colour = _mean_colour(counts[band_tile]) if band_tile else wall_colour
    # what stands behind a cutout, to wear a plain tile instead of the thing's picture
    cut_things = [t for t in things if t.get("cutout")]
    behind = {(t["x"], ty) for t in cut_things for ty in range(t["rows"][0], t["rows"][1] + 1)}
    on_band = {(t["x"], t["rows"][0]) for t in cut_things if t["kind"] == "on-band"}

    # the floor: flat, each tile its picture (a plain floor tile under a cutout)
    for ty in range(rows):
        for tx in range(cols):
            lab = labels[ty][tx]
            if lab in ("floor", "thing", "wall_side"):
                x0, z0 = px(tx, ty)
                pts = [(x0, 0, z0), (x0, 0, z0 + PX), (x0 + PX, 0, z0 + PX), (x0 + PX, 0, z0)]
                if (tx, ty) in behind and (tx, ty) not in on_band and plain_floor:
                    mesh.face(pts, [tile_uv(plain_floor, tx, ty, p[0], p[2]) for p in pts], name="floor")
                else:
                    mesh.face(pts, name="floor")
    # the back wall: a thin slab the wall's rows tall, its face the rows a pixel a pixel
    # (a thing hanging on it painted there; a plain wall behind a cutout), its top the
    # band's colour, its ends the wall's
    wall_cols = [tx for tx in range(cols) if any(labels[ty][tx] in ("wall", "thing") for ty in range(first, wall_last + 1))]
    # (the wall's end cut by the void on the diagonal: the art's corner: the wall turns
    # there, from the end of its straight run down to the side, a tile along)
    diagonals = {}
    for tx in (wall_cols[:1] + wall_cols[-1:]) if wall_cols else []:
        x0, _ = px(tx, 0)
        cut = any(0.05 < float(sprite.drawn[oy + ty * PX:oy + ty * PX + PX, x0:x0 + PX].mean()) < 0.95 for ty in range(first, wall_last + 1))
        if cut and tx == wall_cols[0]:
            diagonals[tx] = [(x0 + PX, wall_z), (x0, wall_z + PX)]
        elif cut and tx == wall_cols[-1]:
            diagonals[tx] = [(x0, wall_z), (x0 + PX, wall_z + PX)]
    if wall_cols and info["wall_rows"] > 0:
        z_back = wall_z - thick
        for tx in wall_cols:
            x0, _ = px(tx, 0)
            if tx in diagonals:
                (ax, az), (bx, bz) = diagonals[tx]
                quad = [(ax, 0, az), (bx, 0, bz), (bx, wall_h, bz), (ax, wall_h, az)]
                if tx != wall_cols[0]:
                    quad = quad[::-1]
                mesh.face(quad, name="wall")          # (the picture where the GBA draws it)
                mesh.face(quad[::-1], mesh.plain(_shade(wall_colour, 0.85)), name="wall")
                continue
            for ty in range(first, wall_last + 1):
                y0 = (wall_last - ty) * PX
                pts = [(x0, y0, wall_z), (x0 + PX, y0, wall_z), (x0 + PX, y0 + PX, wall_z), (x0, y0 + PX, wall_z)]
                if (tx, ty) in behind and (tx, ty) not in on_band and plain_wall:
                    at = (plain_wall[0], ty)
                    mesh.face(pts, [tile_uv(at, tx, ty, p[0], p[2] - p[1]) for p in pts], name="wall")
                else:
                    mesh.face(pts, name="wall")
            mesh.face([(x0, wall_h, z_back), (x0, wall_h, wall_z), (x0 + PX, wall_h, wall_z), (x0 + PX, wall_h, z_back)],
                      mesh.plain(_shade(band_colour, 1.1)), name="wall")
        straight = [tx for tx in wall_cols if tx not in diagonals] or wall_cols
        xa, _ = px(min(straight), 0)
        xb, _ = px(max(straight) + 1, 0)
        end = mesh.plain(_shade(wall_colour, 0.85))
        mesh.face([(xa, 0, z_back), (xa, 0, wall_z), (xa, wall_h, wall_z), (xa, wall_h, z_back)], end, name="wall")
        mesh.face([(xb, 0, wall_z), (xb, 0, z_back), (xb, wall_h, z_back), (xb, wall_h, wall_z)], end, name="wall")
        mesh.face([(xb, 0, z_back), (xa, 0, z_back), (xa, wall_h, z_back), (xb, wall_h, z_back)], end, name="wall")
    # the walls down the sides: the tile's picture flat (the GBA's view of the wall from
    # above), a thin wall standing along the tile's inner edge, as tall as the back wall
    face_colour = mesh.plain(wall_colour)
    top_colour = mesh.plain(_shade(band_colour, 1.1))
    for side_name, tx in (("side-left", 0), ("side-right", cols - 1)):
        tys = [ty for ty in range(rows) if labels[ty][tx] == "wall_side" and not (tx in diagonals and ty == wall_last + 1)]
        if not tys:
            continue
        inner = tx == 0
        x0, _ = px(tx, 0)
        xi = x0 + PX - thick if inner else x0
        xo = x0 + PX if inner else x0 + thick
        ty0 = tys[0]
        while ty0 <= tys[-1]:
            ty1 = ty0
            while ty1 + 1 in tys:
                ty1 += 1
            _, z0 = px(tx, ty0)
            _, z1 = px(tx, ty1 + 1)
            if inner:
                mesh.face([(xo, 0, z0), (xo, 0, z1), (xo, wall_h, z1), (xo, wall_h, z0)], face_colour, name=side_name)
            else:
                mesh.face([(xi, 0, z1), (xi, 0, z0), (xi, wall_h, z0), (xi, wall_h, z1)], face_colour, name=side_name)
            mesh.face([(xi, 0, z1), (xo, 0, z1), (xo, wall_h, z1), (xi, wall_h, z1)], name=side_name)
            mesh.face([(xo, 0, z0), (xi, 0, z0), (xi, wall_h, z0), (xo, wall_h, z0)], face_colour, name=side_name)
            mesh.face([(xi, wall_h, z0), (xi, wall_h, z1), (xo, wall_h, z1), (xo, wall_h, z0)], name=side_name)
            ty0 = ty1 + 1
    # the things
    for k, t in enumerate(things):
        tx, (a, b) = t["x"], t["rows"]
        kind = t["kind"]
        name = "thing-%d" % k
        x0, za = px(tx, a)
        zb = oy + (b + 1) * PX                        # (its foot)
        n = b - a + 1
        if kind == "on-wall" and not t.get("cutout"):
            continue                                   # (painted on the wall's face already)
        if t.get("cutout"):
            # standing at its foot, a hair forward along the GBA's line of sight, its
            # background cut away in the canvas (what's behind it wears a plain tile)
            d = 0.5
            y_foot = -d
            z_foot = zb + d
            if kind == "on-wall":
                z_foot = wall_z + d
            if kind == "on-band":
                # (on the counter's top, at its front edge: the top's height up, as far forward)
                under = things[t["under"]]
                ua, ub = under["rows"]
                h_under = min(float(spec.get("front_height", front_height if ub - ua + 1 > 1 else low_height)), (ub - ua + 1) * PX)
                y_foot = h_under - d
                z_foot = zb + h_under + d
            mesh.face([(x0, y_foot, z_foot), (x0 + PX, y_foot, z_foot), (x0 + PX, y_foot + n * PX, z_foot), (x0, y_foot + n * PX, z_foot)], name=name + ":cutout")
            continue
        if kind == "against-wall":
            h = n * PX
            z_front = zb
            z_back = min(wall_z, zb - 1.0)
        elif kind == "tall":
            h = n * PX
            z_front = zb
            z_back = zb - PX
        else:
            h = float(spec.get("front_height", front_height if n > 1 else low_height))
            h = min(h, n * PX)
            z_front = zb
            z_back = za + h
        # the front, standing at its foot; the top lying back from its head; the sides and
        # the back its own colour (the front's, outlines aside), a little darker
        mesh.face([(x0, 0, z_front), (x0 + PX, 0, z_front), (x0 + PX, h, z_front), (x0, h, z_front)], name=name)
        if z_front - z_back > 1e-6:
            top = [(x0, h, z_back), (x0, h, z_front), (x0 + PX, h, z_front), (x0 + PX, h, z_back)]
            if (tx, a) in on_band:
                mesh.face(top, mesh.plain(info["bands"][(tx, a)]), name=name)
            else:
                mesh.face(top, name=name)
        own = _mean_colour(counts[(tx, b)])
        side = mesh.plain(_shade(own, 0.85))
        back = mesh.plain(_shade(own, 0.75))
        left_free = tx == 0 or not any(o["x"] == tx - 1 and o["rows"][1] == b and o["kind"] == kind for o in things)
        right_free = tx == cols - 1 or not any(o["x"] == tx + 1 and o["rows"][1] == b and o["kind"] == kind for o in things)
        if left_free:
            mesh.face([(x0, 0, z_back), (x0, 0, z_front), (x0, h, z_front), (x0, h, z_back)], side, name=name)
        if right_free:
            mesh.face([(x0 + PX, 0, z_front), (x0 + PX, 0, z_back), (x0 + PX, h, z_back), (x0 + PX, h, z_front)], side, name=name)
        if kind in ("free", "tall"):
            mesh.face([(x0 + PX, 0, z_back), (x0, 0, z_back), (x0, h, z_back), (x0 + PX, h, z_back)], back, name=name)
    return mesh


def cutouts(sprite, g, info, things):
    """The pixels to cut from the canvas for the things standing as cutouts: their tiles'
    pixels of the background round them. (x, y) pairs, flat."""
    bands = info.get("bands", {})
    cut = []
    for t in things:
        if not t.get("cutout"):
            continue
        tx, (a, b) = t["x"], t["rows"]
        for ty in range(a, b + 1):
            back = _background(info, tx, ty, bands)
            edges = _outline_pixels(sprite, g, tx, ty) if t["kind"] == "on-band" else set()
            x0, y0 = g["ox"] + tx * PX, g["oy"] + ty * PX
            for y in range(y0, y0 + PX):
                for x in range(x0, x0 + PX):
                    c = sprite.colour(x, y)
                    if c is not None and (c in back or (x, y) in edges):
                        cut.extend((x, y))
    return cut


def labels_picture(sprite, g, info, things, scale=4):
    """The screenshot with its grid and each tile's label on it, to check and correct."""
    from PIL import Image, ImageDraw
    colours = {"void": (0, 0, 0, 0), "floor": (80, 200, 80, 70), "wall": (80, 120, 240, 90),
               "wall_side": (160, 100, 240, 110), "thing": (240, 120, 60, 110)}
    letters = {"void": "", "floor": "", "wall": "W", "wall_side": "S", "thing": "T"}
    im = sprite.image().resize((sprite.W * scale, sprite.H * scale), Image.NEAREST).convert("RGBA")
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    ox, oy = g["ox"], g["oy"]
    kinds = {}
    for k, t in enumerate(things):
        for ty in range(t["rows"][0], t["rows"][1] + 1):
            kinds[(t["x"], ty)] = ("c" if t.get("cutout") else {"on-wall": "h", "against-wall": "A", "free": "b", "tall": "B"}[t["kind"]]) if t["kind"] != "on-band" else "c"
    for ty in range(g["rows"]):
        for tx in range(g["cols"]):
            lab = info["labels"][ty][tx]
            x0, y0 = (ox + tx * PX) * scale, (oy + ty * PX) * scale
            d.rectangle([x0, y0, x0 + PX * scale - 1, y0 + PX * scale - 1], fill=colours[lab], outline=(255, 0, 0, 120))
            text = letters[lab] + kinds.get((tx, ty), "")
            if text:
                d.text((x0 + 3, y0 + 2), text, fill=(0, 0, 0, 255))
                d.text((x0 + 2, y0 + 1), text, fill=(255, 255, 255, 255))
    return Image.alpha_composite(im, over)
