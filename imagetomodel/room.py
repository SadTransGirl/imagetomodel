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
from .picture import PX, hsv, luma

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
        if (s < 0.1) != (fs < 0.1):
            continue                                  # (a grey is no shade of a coloured floor)
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
        # (the row's own colours: in most of its tiles, the floor's shades aside)
        present = Counter()
        drawn_tiles = 0
        for tx in range(cols):
            if drawn_share[(tx, ty)] < 0.1:
                continue
            drawn_tiles += 1
            for col in counts[(tx, ty)]:
                present[col] += 1
        row_colours = {col for col, n in present.items() if n * 10 >= drawn_tiles * 6}
        if not row_colours:
            break
        wally = 0
        for tx in range(cols):
            c = counts[(tx, ty)]
            n = sum(c.values())
            if n == 0:
                continue
            share = sum(m for col, m in c.items() if col in row_colours) / n
            if share >= 0.4:
                wally += 1
        if wally * 2 < sum(1 for tx in range(cols) if drawn_share[(tx, ty)] >= 0.1):
            break
        wall_colours |= row_colours
        wall_rows = ty + 1 - first
    # (a plain tile a wall row, made up: each pixel row's commonest colour across the
    # row's tiles, the wall's bands running across; the wall's colours are its)
    made_up = {}
    for ty in range(first, first + wall_rows):
        pixels = []
        for j in range(PX):
            c = Counter()
            for tx in range(cols):
                t, d = tiles[(tx, ty)]
                for i in range(PX):
                    if d[j, i]:
                        c[tuple(int(v) for v in t[j, i, :3])] += 1
            pixels.append(c.most_common(1)[0][0] if c else None)
        made_up[ty] = pixels
        wall_colours |= {p for p in pixels if p is not None}
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
            if drawn_share[k] < 0.05:
                lab = "void"
            elif k in side:
                lab = "wall_side"
            elif first <= ty <= wall_last:
                # (the wall where its upper half is the wall's colours: its lower half may be
                # the back of something standing before it)
                t, d = tiles[k]
                plain = made_up.get(ty, [])
                same = total_ = 0
                for j in range(PX // 2):
                    for i in range(PX):
                        if d[j, i]:
                            total_ += 1
                            if j < len(plain) and plain[j] is not None and tuple(int(v) for v in t[j, i, :3]) == plain[j]:
                                same += 1
                share = same / total_ if total_ else 0.0
                lab = "wall" if share >= 0.7 or drawn_share[k] < 0.75 or tx in (0, cols - 1) else "thing"
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
    # a plain wall tile a row (the row's most repeated), for the walls' other faces and
    # the wall behind a thing; the floor's tiles (the repeated tiles that are floor), for
    # the floor under a thing
    plain_wall = None
    plain_wall_rows = {}
    for ty in range(first, wall_last + 1):
        best, best_share = None, -1.0
        for tx in range(cols):
            c = counts[(tx, ty)]
            n = sum(c.values())
            if n < PX * PX or labels[ty][tx] != "wall":
                continue
            share = sum(m for col, m in c.items() if col in wall_colours) / n
            if share > best_share:
                best, best_share = (tx, ty), share
        if best:
            plain_wall_rows[ty] = best
            if plain_wall is None:
                plain_wall = best
    floor_tiles = {}
    for k, h in hashes.items():
        if h and whole[h] >= 2 and floor_share[k] >= 0.95 and h not in floor_tiles:
            floor_tiles[h] = k
    return {"labels": labels, "wall_first": first, "wall_rows": wall_rows,
            "floor_colours": ["%02x%02x%02x" % c for c in sorted(floor_colours)],
            "wall_colours": ["%02x%02x%02x" % c for c in sorted(wall_colours)],
            "plain_wall": list(plain_wall) if plain_wall else None, "counts": counts, "forced": forced,
            "sprite": spr, "floor_share": floor_share, "plain_wall_rows": plain_wall_rows, "made_up_wall": made_up,
            "floor_tiles": list(floor_tiles.values()), "plain_floor": next((list(k) for k, h in hashes.items() if h == modal), None)}


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
    not a yellow cushion over a blue one, though both have a grey foot: a grey counts only
    as a tile's main colour)."""
    def top(c):
        return [col for col, _ in c.most_common() if not _dark(col)][:2]
    ta, tb = top(a), top(b)
    for i, x in enumerate(ta):
        for j, y in enumerate(tb):
            if hsv(x)[1] < 0.12 or hsv(y)[1] < 0.12:
                if i == 0 and j == 0 and _same_hue(x, y):
                    return True
            elif _same_hue(x, y):
                return True
    return False


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


def _background(info, tx, ty, bands=None, on_band=False):
    """The colours round a thing in its tile that aren't it: the floor's shades, the
    wall's colours on the wall's rows, and the band's colour where it stands on a counter."""
    floor_colours = {tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in info["floor_colours"]}
    wall_colours = {tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in info["wall_colours"]}
    wall_last = info["wall_first"] + info["wall_rows"] - 1
    back = set()
    for c in info["counts"][(tx, ty)]:
        if _alike(c, floor_colours) or (ty <= wall_last and c in wall_colours):
            back.add(c)
    if on_band and bands and (tx, ty) in bands:
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
        back = _background(info, tx, ty, bands, on_band=outlines)
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


def _reaches(info, g, tx, ty, bands, edge):
    """Whether a tile's own pixels (not its background) reach its top or bottom two rows."""
    own = _own_pixels(info, g, tx, ty, bands, "thing")
    y0 = g["oy"] + ty * PX
    rows_ = (y0, y0 + 1) if edge == "top" else (y0 + PX - 2, y0 + PX - 1)
    return any(y in rows_ for _, y in own)


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
            return _reaches(info, g, tx, b, bands, "top")
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
            # rows, the two touching: the thing's foot, a plant's pot)
            if b == wall_last and b + 1 < rows and labels[b + 1][tx] == "floor" and floor_share[(tx, b + 1)] < 0.95 \
                    and _reaches(info, g, tx, b, bands, "bottom") and _reaches(info, g, tx, b + 1, bands, "top"):
                labels[b + 1][tx] = "thing"
            while b + 1 < rows and labels[b + 1][tx] == "thing" and forced.get((tx, b)) == forced.get((tx, b + 1)):
                if b == wall_last and not _reaches(info, g, tx, b, bands, "bottom"):
                    break                             # (hanging on the wall, clear of what's under it)
                if a <= wall_last and b + 1 > wall_last and band_colour(tx, b + 1) is not None \
                        and band_colour(tx, b + 1) not in top3(tx, b):
                    b += 1                            # (its foot on the counter's back edge)
                    shared = True
                    break
                if not joined(tx, b, b + 1):
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


def _own_pixels(info, g, tx, ty, bands, kind):
    """A thing's own pixels in a tile: all but the background's (the floor's shades, the
    wall's colours on the wall's rows, a counter's colour and edges under what stands on it)."""
    sprite = info["sprite"]
    ox, oy = g["ox"], g["oy"]
    back = _background(info, tx, ty, bands, on_band=(kind == "on-band"))
    edges = _outline_pixels(sprite, g, tx, ty) if kind == "on-band" else set()
    own = set()
    for y in range(oy + ty * PX, oy + ty * PX + PX):
        for x in range(ox + tx * PX, ox + tx * PX + PX):
            c = sprite.colour(x, y)
            if c is not None and c not in back and (x, y) not in edges:
                own.add((x, y))
    return own


def objects(info, g, things):
    """The things as objects: runs side by side with the same rows and kind are one (a
    counter, a table); each with its tiles, its own pixels, and the box round them
    ("bbox": [x0, y0, x1, y1] in the picture's pixels, y1 and x1 past the last)."""
    bands = info.get("bands", {})
    by_run = {}
    for k, t in enumerate(things):
        by_run[k] = {"kind": t["kind"], "rows": tuple(t["rows"]), "x": t["x"], "runs": [k], "cutout": t.get("cutout", False),
                     "under": t.get("under")}
    groups = []
    taken = set()
    for k in sorted(by_run, key=lambda k: (by_run[k]["rows"], by_run[k]["x"])):
        if k in taken:
            continue
        o = by_run[k]
        group = [k]
        taken.add(k)
        x = o["x"]
        while True:
            nxt = next((j for j in by_run if j not in taken and by_run[j]["x"] == x + 1 and by_run[j]["rows"] == o["rows"]
                        and by_run[j]["kind"] == o["kind"] and by_run[j]["cutout"] == o["cutout"] and not o["cutout"]
                        and _family(info["counts"][(x, o["rows"][1])], info["counts"][(x + 1, o["rows"][1])])), None)
            if nxt is None:
                break
            group.append(nxt)
            taken.add(nxt)
            x += 1
        groups.append(group)
    out = []
    for group in groups:
        first = by_run[group[0]]
        a, b = first["rows"]
        tiles = [(by_run[k]["x"], ty) for k in group for ty in range(a, b + 1)]
        own = set()
        block = set()
        for tx, ty in tiles:
            own |= _own_pixels(info, g, tx, ty, bands, first["kind"])
            x0, y0 = g["ox"] + tx * PX, g["oy"] + ty * PX
            block |= {(x, y) for y in range(y0, y0 + PX) for x in range(x0, x0 + PX)}
        sprite = info["sprite"]
        drawn = {p for p in block if sprite.colour(*p) is not None}
        # (background by colour that can't be reached from the block's border through
        # background is inside the thing: a highlight in a pot, a screen in a frame)
        back = (drawn - own)
        seen = set()
        todo = [p for p in back if not all((p[0] + dx, p[1] + dy) in block for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
        todo += [p for p in back if any((p[0] + dx, p[1] + dy) not in drawn and (p[0] + dx, p[1] + dy) in block for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
        while todo:
            p = todo.pop()
            if p in seen or p not in back:
                continue
            seen.add(p)
            todo += [(p[0] + dx, p[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        own |= back - seen
        if not own:
            continue
        xs = [p[0] for p in own]
        ys = [p[1] for p in own]
        labels = info["labels"]
        rows_n = len(labels)
        # (on the void's edge, its pixels running off the floor into it: a door's mat, flat)
        def void_below(tx, ty):
            if ty + 1 >= rows_n or labels[ty + 1][tx] == "void":
                return True
            d = info["sprite"].drawn[g["oy"] + (ty + 1) * PX:g["oy"] + (ty + 2) * PX, g["ox"] + tx * PX:g["ox"] + (tx + 1) * PX]
            return float(d.mean()) < 0.75                 # (half on the void)
        def runs_off(tx, ty):
            # (its pixels going on into the tile below, half on the void: a mat into the exit)
            return ty + 1 < rows_n and labels[ty + 1][tx] != "void" and _reaches(info, g, tx, ty + 1, bands, "top")
        at_edge = any(void_below(tx, ty) and _reaches(info, g, tx, ty, bands, "bottom") and runs_off(tx, ty) for tx, ty in tiles) \
            or any(info["floor_share"][(tx, ty)] < 0.99 and float(info["sprite"].drawn[g["oy"] + ty * PX:g["oy"] + ty * PX + PX, g["ox"] + tx * PX:g["ox"] + tx * PX + PX].mean()) < 0.9 for tx, ty in tiles)
        kind = "flat" if at_edge and first["kind"] == "free" else first["kind"]
        out.append({"kind": kind, "rows": [a, b], "columns": [by_run[group[0]]["x"], by_run[group[-1]]["x"]],
                    "tiles": tiles, "own": own, "bbox": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
                    "cutout": first["cutout"], "under": first["under"], "runs": group})
    return out


def _front_rows(info, obj):
    """How many of a thing's own rows are its front, as the GBA shades it: the darker band
    along its bottom (its outline and the rows under its lighter top)."""
    sprite = info["sprite"]
    x0, y0, x1, y1 = obj["bbox"]
    rows = []
    for y in range(y0, y1):
        vals = [luma(sprite.colour(x, y)) for x in range(x0, x1) if (x, y) in obj["own"]]
        rows.append(sum(vals) / len(vals) if vals else None)
    light = sorted(v for v in rows if v is not None)
    if not light:
        return 0
    median = light[len(light) // 2]
    n = 0
    for v in reversed(rows):
        if v is not None and v < median - 0.06:
            n += 1
        elif n > 0:
            break
        else:
            n += 1 if v is None else 0
            if v is not None:
                break
    return n


def _profile(sprite, o, rows, side):
    """A thing's edge down its front rows, as a pixel artist extrudes it: for each row, its
    outermost own pixel's colour (its outline) and the one inside it (its body), from the
    left or the right; None for a row with nothing."""
    x0, _, x1, _ = o["bbox"]
    out = []
    for y in rows:
        xs = sorted(x for (x, yy) in o["own"] if yy == y)
        if not xs:
            out.append((None, None))
            continue
        if side == "right":
            xs = xs[::-1]
        edge = sprite.colour(xs[0], y)
        body = sprite.colour(xs[1], y) if len(xs) > 1 else edge
        if body is not None and _dark(body) and len(xs) > 2 and not _dark(sprite.colour(xs[2], y) or (0, 0, 0)):
            body = sprite.colour(xs[2], y)
        out.append((edge, body))
    return out


def _darkest(sprite, o):
    best = None
    for (x, y) in o["own"]:
        c = sprite.colour(x, y)
        if c is not None and (best is None or luma(c) < luma(best)):
            best = c
    return best or (40, 40, 60)


def painted_faces(sprite, o, front_rows, w, d, with_top):
    """Textures for the faces the GBA never drew, painted from the thing's own art: its
    sides (the front's edge profile across the depth, the outline along the back edge), its
    back (a panel in the front's row colours, outlined), and, `with_top`, its top (its top
    row's colours back over it, outlined). Rows of (r, g, b) or None."""
    outline = _darkest(sprite, o)
    d = max(1, int(round(d)))
    w = max(1, int(round(w)))
    faces = {}
    for side in ("left", "right"):
        rows = []
        for edge, body in _profile(sprite, o, front_rows, side):
            if edge is None:
                rows.append([None] * d)
                continue
            line = edge if _dark(edge) else outline
            fill = body if body is not None else edge
            rows.append([fill] * max(0, d - 1) + [line] if d > 1 else [fill])
        faces[side] = rows
    rows = []
    for y in front_rows:
        own = [sprite.colour(x, y) for (x, yy) in o["own"] if yy == y]
        own = [c for c in own if c is not None]
        if not own:
            rows.append([None] * w)
            continue
        body = Counter(c for c in own if not _dark(c)).most_common(1)
        fill = body[0][0] if body else own[0]
        line = outline if any(_dark(c) for c in own) else fill
        rows.append([line] + [fill] * max(0, w - 2) + ([line] if w > 1 else []))
    faces["back"] = rows
    if with_top:
        x0, y0, x1, y1 = o["bbox"]
        tops = []
        for x in range(x0, x0 + w):
            ys = sorted(y for (xx, y) in o["own"] if xx == x)
            if not ys:
                tops.append((None, None))
                continue
            edge = sprite.colour(x, ys[0])
            body = sprite.colour(x, ys[1]) if len(ys) > 1 else edge
            tops.append((edge, body))
        rows = []
        for j in range(d):
            row = []
            for i, (edge, body) in enumerate(tops):
                if edge is None:
                    row.append(None)
                elif j == 0 or i == 0 or i == w - 1:
                    row.append(edge if _dark(edge) else outline)
                else:
                    row.append(body if body is not None else edge)
            rows.append(row)
        faces["top"] = rows
    return faces


RAY = (0.0, 1.0, 1.0)             # (a step along the GBA's line of sight: up and forward, alike)


def depth_map(sprite, o, x0, y0, w, h, levels):
    """How far each pixel of a thing's face stands out, as voxel art of a sprite does it:
    its brightness, over `levels` steps (the lightest out, the darkest in), outlines at the
    base; None where nothing is drawn. Rows of ints."""
    own = o["own"]
    vals = []
    for y in range(y0, y0 + h):
        for x in range(x0, x0 + w):
            c = sprite.colour(x, y) if (x, y) in own else None
            if c is not None and not _dark(c):
                vals.append(luma(c))
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    span = max(hi - lo, 0.05)
    rows = []
    for y in range(y0, y0 + h):
        row = []
        for x in range(x0, x0 + w):
            c = sprite.colour(x, y) if (x, y) in own else None
            if c is None:
                row.append(None)
            elif _dark(c) or levels <= 0:
                row.append(0)
            else:
                row.append(int(round(levels * (luma(c) - lo) / span)))
        rows.append(row)
    return rows


def relief(mesh, origin, du, dv, depths, name, uv_at=None, thick=0.0):
    """A face in relief: its pixel grid from `origin` (a corner, 3D) along `du` (a pixel
    across) and `dv` (a pixel down the picture), each pixel raised by its depth along the
    GBA's line of sight (so its picture stays put from the GBA's angle), with walls between
    pixels of different depths and round the edge (down to -`thick` behind a cutout, so it
    is a slab). `uv_at(i, j)`: a pixel's texture coordinate (its picture's, by default)."""
    ox, oy, oz = origin
    h = len(depths)
    w = len(depths[0]) if h else 0

    def at(i, j, d):
        return (ox + i * du[0] + j * dv[0] + d * RAY[0], oy + i * du[1] + j * dv[1] + d * RAY[1], oz + i * du[2] + j * dv[2] + d * RAY[2])

    def uv(i, j):
        return uv_at(i, j) if uv_at else None
    base = -thick
    for j in range(h):
        i = 0
        while i < w:
            d = depths[j][i]
            if d is None:
                i += 1
                continue
            i1 = i
            while i1 + 1 < w and depths[j][i1 + 1] == d:
                i1 += 1
            pts = [at(i, j, d), at(i, j + 1, d), at(i1 + 1, j + 1, d), at(i1 + 1, j, d)]
            if uv_at:
                mesh.face(pts, [uv(i, j), uv(i, j), uv(i1, j), uv(i1, j)], name=name)
            else:
                mesh.face(pts, name=name)
            if thick > 0:
                mesh.face(pts[::-1] if False else [at(i, j, base), at(i1 + 1, j, base), at(i1 + 1, j + 1, base), at(i, j + 1, base)],
                          [uv(i, j), uv(i1, j), uv(i1, j), uv(i, j)] if uv_at else None, name=name)
            i = i1 + 1
    # the walls between pixels of different depths: across (between columns) and down
    def wall(a, b, d_lo, d_hi, colour_ij):
        if d_lo is None:
            d_lo = base
        if d_hi is None:
            d_hi = base
        if abs(d_hi - d_lo) < 1e-9:
            return
        pts = [at(*a, d_lo), at(*b, d_lo), at(*b, d_hi), at(*a, d_hi)]
        t = uv(*colour_ij) if uv_at else _pixel_uv(origin, du, dv, colour_ij)
        mesh.face(pts, t, name=name)
    for j in range(h):
        for i in range(w + 1):
            left = depths[j][i - 1] if i > 0 else None
            right = depths[j][i] if i < w else None
            if left is None and right is None:
                continue
            owner = (i, j) if right is not None and (left is None or right >= left) else (i - 1, j)
            wall((i, j), (i, j + 1), left, right, owner)
    for i in range(w):
        for j in range(h + 1):
            up = depths[j - 1][i] if j > 0 else None
            down = depths[j][i] if j < h else None
            if up is None and down is None:
                continue
            owner = (i, j) if down is not None and (up is None or down >= up) else (i, j - 1)
            wall((i, j), (i + 1, j), up, down, owner)


def _pixel_uv(origin, du, dv, ij):
    """A pixel's own texture coordinate: where the GBA draws its middle."""
    i, j = ij
    p = (origin[0] + (i + 0.5) * du[0] + (j + 0.5) * dv[0], origin[1] + (i + 0.5) * du[1] + (j + 0.5) * dv[1],
         origin[2] + (i + 0.5) * du[2] + (j + 0.5) * dv[2])
    return (p[0], p[2] - p[1])


def add_thing(mesh, sprite, o, objs, wall_z, front_height="auto", low_height=8.0, spec=None, info=None, name="thing", tops="painted", levels=3, thick=2.0):
    """One thing's faces added to a mesh: a cutout standing, or a box round its own pixels,
    its front standing at its foot, its top lying back from its head (or painted, against
    the wall), its sides and back painted from its art."""
    spec = dict(spec or {})
    kind = o["kind"]
    x0, y0, x1, y1 = (float(v) for v in o["bbox"])
    zb = y1
    tall = y1 - y0

    def front_height_of(obj):
        n_rows = obj["rows"][1] - obj["rows"][0] + 1
        bx0, by0, bx1, by1 = obj["bbox"]
        if "front_height" in spec:
            h = float(spec["front_height"])
        elif front_height == "auto":
            h = float(_front_rows(info, obj)) if info else float(PX)
            if h <= 1:
                h = float(low_height if n_rows == 1 else PX)
        else:
            h = float(front_height if n_rows > 1 else low_height)
        return max(1.0, min(h, float(by1 - by0)))
    if kind == "flat" or (kind == "on-wall" and not o["cutout"]):
        return
    if o["cutout"]:
        # (a hair forward along the GBA's line of sight: up and forward alike keeps its
        # picture where the GBA draws it)
        d = 0.5
        y_foot = d
        z_foot = zb + d
        if kind == "on-wall":
            z_foot = wall_z + d
        if kind == "on-band":
            under = next((u for u in objs if o["under"] in u["runs"]), None)
            h_under = front_height_of(under) if under else float(low_height)
            y_foot = h_under + d
            z_foot = zb + h_under + d
        # (a slab: each pixel a voxel `thick` deep, raised by its brightness, its silhouette
        # walled round)
        w, h = int(round(x1 - x0)), int(round(tall))
        depths = depth_map(sprite, o, int(round(x0)), int(round(y0)), w, h, levels)
        relief(mesh, (x0, y_foot + tall, z_foot), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), depths, name + ":cutout", thick=thick)
        return
    if kind == "against-wall":
        h = tall
        z_front = zb
        z_back = min(wall_z, zb - 1.0)
    elif kind == "tall":
        h = tall
        z_front = zb
        z_back = zb - PX
    else:
        h = front_height_of(o)
        z_front = zb
        z_back = y0 + h
    depth = z_front - z_back
    w_px, h_px = int(round(x1 - x0)), int(round(h))
    front_rows = range(int(round(y1 - h)), int(round(y1)))
    if levels > 0 and h_px >= 1:
        depths = depth_map(sprite, o, int(round(x0)), int(round(y1 - h)), w_px, h_px, levels)
        relief(mesh, (x0, h, z_front), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), depths, name)
    else:
        mesh.face([(x0, 0, z_front), (x1, 0, z_front), (x1, h, z_front), (x0, h, z_front)], name=name)
    faces = painted_faces(sprite, o, front_rows, x1 - x0, depth, with_top=(kind in ("against-wall", "tall")))
    w = x1 - x0
    if depth > 1e-6:
        top = [(x0, h, z_back), (x0, h, z_front), (x1, h, z_front), (x1, h, z_back)]
        if kind in ("against-wall", "tall") and tops == "painted":
            # (painted from its art: from the GBA's angle it covers the wall's band above
            # it, as the port's own tops do; "picture" wears the band there instead)
            pu, pv = mesh.patch(faces["top"])
            dp = len(faces["top"])
            mesh.face(top, [(pu, pv), (pu, pv + dp), (pu + w, pv + dp), (pu + w, pv)], name=name)
        elif levels > 0 and int(round(depth)) >= 1:
            # (the top as drawn, in relief: a pixel row of the picture a pixel back)
            d_px = int(round(depth))
            depths = depth_map(sprite, o, int(round(x0)), int(round(y0)), w_px, d_px, levels)
            relief(mesh, (x0, h, z_back), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), depths, name)
        else:
            mesh.face(top, name=name)
        dp = len(faces["left"][0]) if faces["left"] else 1
        pu, pv = mesh.patch(faces["left"])
        mesh.face([(x0, 0, z_back), (x0, 0, z_front), (x0, h, z_front), (x0, h, z_back)],
                  [(pu + dp, pv + h), (pu, pv + h), (pu, pv), (pu + dp, pv)], name=name)
        pu, pv = mesh.patch(faces["right"])
        mesh.face([(x1, 0, z_front), (x1, 0, z_back), (x1, h, z_back), (x1, h, z_front)],
                  [(pu, pv + h), (pu + dp, pv + h), (pu + dp, pv), (pu, pv)], name=name)
    if kind in ("free", "tall") and depth > 1e-6:
        pu, pv = mesh.patch(faces["back"])
        mesh.face([(x1, 0, z_back), (x0, 0, z_back), (x0, h, z_back), (x1, h, z_back)],
                  [(pu, pv + h), (pu + w, pv + h), (pu + w, pv), (pu, pv)], name=name)


def build_object(sprite, g, info, things, k, front_height="auto", low_height=8.0, spec=None, tops="painted", levels=3, thick=2.0):
    """One thing as a model of its own: (the sprite cropped to its tiles, its mesh in that
    sprite's pixels, the pixels to cut, its block (tx, ty, cols, rows)); None for a thing
    that is only paint on the wall or the floor."""
    from .geometry import rebased
    objs = objects(info, g, things)
    o = objs[k]
    if o["kind"] == "flat" or (o["kind"] == "on-wall" and not o["cutout"]):
        return None
    (ta, tb), (ca, cb) = o["rows"], o["columns"]
    bx, by = g["ox"] + ca * PX, g["oy"] + ta * PX
    cols, rows = cb - ca + 1, tb - ta + 1
    crop = sprite.crop(bx, by, cols * PX, rows * PX)
    wall_z = g["oy"] + (info["wall_first"] + info["wall_rows"]) * PX
    mesh = Mesh(strip_v=sprite.H, strip_w=crop.W)
    add_thing(mesh, sprite, o, objs, wall_z, front_height, low_height, spec, info, name="thing-%d" % k, tops=tops, levels=levels, thick=thick)
    moved = rebased(mesh, bx, by, sprite.H, crop.H, crop.W)
    cut = []
    for tx, ty in o["tiles"]:
        x0, y0 = g["ox"] + tx * PX, g["oy"] + ty * PX
        for y in range(y0, y0 + PX):
            for x in range(x0, x0 + PX):
                if sprite.colour(x, y) is not None and (x, y) not in o["own"]:
                    cut.extend((x - bx, y - by))
    return crop, moved, cut, (bx // PX, by // PX, cols, rows)


def build(sprite, g, info, things, front_height="auto", low_height=8.0, wall_thick=4.0, spec=None, tops="painted", levels=3, thick=2.0):
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
    mesh = Mesh(strip_v=sprite.H, strip_w=sprite.W)

    def px(tx, ty):
        return ox + tx * PX, oy + ty * PX

    # (a plain tile a wall row, made up in label(): a patch in the strip, for the wall
    # behind a thing and the walls' other faces)
    wall_patches = {ty: mesh.patch([[c] * PX for c in rows_]) for ty, rows_ in info.get("made_up_wall", {}).items()}

    def tile_uv(at, tx, ty, x, z):
        """The picture of tile `at` where a point of tile (tx, ty) would be drawn."""
        ax, ay = px(*at)
        x0, z0 = px(tx, ty)
        return (ax + (x - x0), ay + (z - z0))
    plain_floor = tuple(info["plain_floor"]) if info.get("plain_floor") else None
    made = info.get("made_up_wall", {})
    wall_colour = _mean_colour(Counter(c for c in made.get(wall_last, []) if c)) if made.get(wall_last) else (200, 190, 170)
    band_colour = _mean_colour(Counter(c for c in made.get(first, [])[:PX // 2] if c)) if made.get(first) else wall_colour
    # what stands behind a thing (the floor, the wall, a counter's top) wears a plain
    # tile instead of the thing's picture: the thing is cut from its background
    objs = objects(info, g, things)
    behind = {tile for o in objs if o["kind"] != "flat" and (o["kind"] != "on-wall" or o["cutout"]) for tile in o["tiles"]}
    on_band = {tile for o in objs if o["kind"] == "on-band" for tile in o["tiles"]}
    floor_colours = {tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in info["floor_colours"]}

    def floor_like_tile(tx, ty):
        """The floor tile most like the floor round a thing in its tile (its shade)."""
        candidates = [tuple(t) for t in info.get("floor_tiles", [])] or ([plain_floor] if plain_floor else [])
        if not candidates:
            return None
        x0, y0 = px(tx, ty)
        best, best_score = candidates[0], -1
        for cand in candidates:
            cx, cy = px(*cand)
            score = 0
            for j in range(PX):
                for i in range(PX):
                    c = sprite.colour(x0 + i, y0 + j)
                    if c is not None and _alike(c, floor_colours) and sprite.colour(cx + i, cy + j) == c:
                        score += 1
            if score > best_score:
                best, best_score = cand, score
        return best

    # the floor: flat, each tile its picture (a plain floor tile under a cutout)
    for ty in range(rows):
        for tx in range(cols):
            lab = labels[ty][tx]
            if lab in ("floor", "thing", "wall_side"):
                x0, z0 = px(tx, ty)
                pts = [(x0, 0, z0), (x0, 0, z0 + PX), (x0 + PX, 0, z0 + PX), (x0 + PX, 0, z0)]
                under = floor_like_tile(tx, ty) if (tx, ty) in behind and (tx, ty) not in on_band else None
                if under:
                    mesh.face(pts, [tile_uv(under, tx, ty, p[0], p[2]) for p in pts], name="floor")
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
        # (how far in the void cuts the corner: the cut runs where the void's pixels end,
        # a line at 45 degrees c pixels from the outer corner of the wall's top tile)
        y0 = oy + first * PX
        void = ~sprite.drawn[y0:y0 + PX, x0:x0 + PX]
        ys_, xs_ = np.nonzero(void)
        if len(xs_) == 0 or len(xs_) == PX * PX:
            continue
        left = tx == wall_cols[0]
        c = int((xs_ + ys_).max() + 1) if left else int(((PX - 1 - xs_) + ys_).max() + 1)
        c = max(2, min(PX, c))
        # (a hair past each end, so the pixels along its edge are drawn)
        if left:
            diagonals[tx] = [(x0 + c + 0.5, wall_z - 0.5), (x0 - 0.5, wall_z + c + 0.5), c]
        else:
            diagonals[tx] = [(x0 + PX - c - 0.5, wall_z - 0.5), (x0 + PX + 0.5, wall_z + c + 0.5), c]
    if wall_cols and info["wall_rows"] > 0:
        z_back = wall_z - thick
        for tx in wall_cols:
            x0, _ = px(tx, 0)
            if tx in diagonals:
                (ax, az), (bx, bz), c = diagonals[tx]
                quad = [(ax, 0, az), (bx, 0, bz), (bx, wall_h, bz), (ax, wall_h, az)]
                if tx != wall_cols[0]:
                    quad = quad[::-1]
                # (the wall's band rows, made up, run round the corner; seen from both sides)
                if wall_patches:
                    length = ((bx - ax) ** 2 + (bz - az) ** 2) ** 0.5
                    uvs = []
                    # (one row's patch stretched along it, a face per wall row)
                    for ty in range(first, wall_last + 1):
                        y_top = wall_h - (ty - first) * PX
                        y_bot = y_top - PX
                        pu, pv = wall_patches[ty]
                        q = [(ax, y_bot, az), (bx, y_bot, bz), (bx, y_top, bz), (ax, y_top, az)]
                        if tx != wall_cols[0]:
                            q = q[::-1]
                        uvs = []
                        for p in q:
                            along = ((p[0] - ax) ** 2 + (p[2] - az) ** 2) ** 0.5
                            uvs.append((pu + min(PX - 0.01, along * (PX - 0.02) / max(length, 1e-6)), pv + (y_top - p[1])))
                        mesh.face(q, uvs, name="wall")
                else:
                    mesh.face(quad, name="wall")
                # (the straight wall on from the cut to the tile's end)
                xs0, xs1 = (x0 + c, x0 + PX) if tx == wall_cols[0] else (x0, x0 + PX - c)
                if xs1 - xs0 > 0.5:
                    for ty in range(first, wall_last + 1):
                        y0_ = (wall_last - ty) * PX
                        mesh.face([(xs0, y0_, wall_z), (xs1, y0_, wall_z), (xs1, y0_ + PX, wall_z), (xs0, y0_ + PX, wall_z)], name="wall")
                continue
            for ty in range(first, wall_last + 1):
                y0 = (wall_last - ty) * PX
                pts = [(x0, y0, wall_z), (x0 + PX, y0, wall_z), (x0 + PX, y0 + PX, wall_z), (x0, y0 + PX, wall_z)]
                if (tx, ty) in behind and (tx, ty) not in on_band and ty in wall_patches:
                    pu, pv = wall_patches[ty]
                    mesh.face(pts, [(pu + (p[0] - x0), pv + (p[2] - p[1] - oy - ty * PX)) for p in pts], name="wall")
                else:
                    mesh.face(pts, name="wall")
            mesh.face([(x0, wall_h, z_back), (x0, wall_h, wall_z), (x0 + PX, wall_h, wall_z), (x0 + PX, wall_h, z_back)],
                      mesh.plain(_shade(band_colour, 1.1)), name="wall")
        straight = [tx for tx in wall_cols if tx not in diagonals] or wall_cols
        xa, _ = px(min(straight), 0)
        xb, _ = px(max(straight) + 1, 0)
        if wall_cols[0] in diagonals:
            xa = px(wall_cols[0], 0)[0] + diagonals[wall_cols[0]][2]
        if wall_cols[-1] in diagonals:
            xb = px(wall_cols[-1], 0)[0] + PX - diagonals[wall_cols[-1]][2]
        end = mesh.plain(_shade(wall_colour, 0.85))
        mesh.face([(xa, 0, z_back), (xa, 0, wall_z), (xa, wall_h, wall_z), (xa, wall_h, z_back)], end, name="wall")
        mesh.face([(xb, 0, wall_z), (xb, 0, z_back), (xb, wall_h, z_back), (xb, wall_h, wall_z)], end, name="wall")
        mesh.face([(xb, 0, z_back), (xa, 0, z_back), (xa, wall_h, z_back), (xb, wall_h, z_back)], end, name="wall")
    # the walls down the sides: the tile's picture flat (the GBA's view of the wall from
    # above), a thin wall standing along the tile's inner edge, as tall as the back wall
    face_colour = mesh.plain(wall_colour)
    top_colour = mesh.plain(_shade(band_colour, 1.1))
    for side_name, tx in (("side-left", 0), ("side-right", cols - 1)):
        tys = [ty for ty in range(rows) if labels[ty][tx] == "wall_side"]
        if not tys:
            continue
        z_from = wall_z + diagonals[tx][2] if tx in diagonals else None
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
            if z_from is not None and z0 < z_from < z1:
                z0 = z_from                           # (from the corner's cut on)
            if inner:
                mesh.face([(xo, 0, z0), (xo, 0, z1), (xo, wall_h, z1), (xo, wall_h, z0)], face_colour, name=side_name)
            else:
                mesh.face([(xi, 0, z1), (xi, 0, z0), (xi, wall_h, z0), (xi, wall_h, z1)], face_colour, name=side_name)
            mesh.face([(xi, 0, z1), (xo, 0, z1), (xo, wall_h, z1), (xi, wall_h, z1)], name=side_name)
            mesh.face([(xo, 0, z0), (xi, 0, z0), (xi, wall_h, z0), (xo, wall_h, z0)], face_colour, name=side_name)
            mesh.face([(xi, wall_h, z0), (xi, wall_h, z1), (xo, wall_h, z1), (xo, wall_h, z0)], name=side_name)
            ty0 = ty1 + 1
    # the things: each a cutout standing, or a box round its own pixels (the floor showing
    # round it), its sides and back painted from its art
    for k, o in enumerate(objs):
        add_thing(mesh, sprite, o, objs, wall_z, front_height, low_height, spec, info, name="thing-%d" % k, tops=tops, levels=levels, thick=thick)
    return mesh


def cutouts(sprite, g, info, things):
    """The pixels to cut from the canvas: every thing's background in its tiles (the floor
    round a cushion, the wall round a plant, the counter under a flower). (x, y) pairs."""
    cut = []
    for o in objects(info, g, things):
        if o["kind"] == "flat" or (o["kind"] == "on-wall" and not o["cutout"]):
            continue
        for tx, ty in o["tiles"]:
            x0, y0 = g["ox"] + tx * PX, g["oy"] + ty * PX
            for y in range(y0, y0 + PX):
                for x in range(x0, x0 + PX):
                    if sprite.colour(x, y) is not None and (x, y) not in o["own"]:
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
