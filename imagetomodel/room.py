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
import math
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
                for j in range(PX):
                    for i in range(PX):
                        if d[j, i]:
                            total_ += 1
                            if j < len(plain) and plain[j] is not None and tuple(int(v) for v in t[j, i, :3]) == plain[j]:
                                same += 1
                share = same / total_ if total_ else 0.0
                # (or the same as another tile of the row: the wall's own pattern, a scallop)
                twin = 0.0
                if share >= 0.5:
                    for ox_ in range(cols):
                        if ox_ == tx:
                            continue
                        t2, d2 = tiles[(ox_, ty)]
                        if d2.all() and d.all():
                            twin = max(twin, float((t2[:, :, :3] == t[:, :, :3]).all(axis=2).mean()))
                lab = "wall" if share >= 0.8 or twin >= 0.95 or drawn_share[k] < 0.75 or tx in (0, cols - 1) else "thing"
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
    return {"_oy": oy, "labels": labels, "wall_first": first, "wall_rows": wall_rows,
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
        mine = _own_pixels(info, g, tx, ty, bands, "on-band" if outlines else "thing")
        for y in range(oy + ty * PX, oy + ty * PX + PX):
            for x in range(ox + tx * PX, ox + tx * PX + PX):
                c = sprite.colour(x, y)
                if c is None:
                    continue
                total += 1
                if (x, y) in mine:
                    own += 1
                    if hsv(c)[1] > 0.25:
                        saturated += 1
                    xs.append(x - ox - tx * PX)
                    ys.append(y)
    if not xs or total == 0 or own * 5 > total * 4:
        return False
    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1
    # (clear of the tile's sides but for a few rows: a plant's leaves may touch them)
    touching = {y for x, y in zip(xs, ys) if x < 1 or x > PX - 2}
    if len(touching) * 4 > height:
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
    for ty in range(wall_last + 1, rows):             # (a counter stands in the room)
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
        # (whether the tile b, under a, is part of the same thing: their own colours, not
        # the wall's or the floor's round them)
        if a <= wall_last < b:
            return _reaches(info, g, tx, b, bands, "top")
        return _family(_own_counts(info, g, tx, a), _own_counts(info, g, tx, b)) and _touching(info, g, tx, a, b)
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
            if shared:
                thing["foot_on_band"] = True          # (its last tile the counter's back edge)
            if f is None and kind != "on-wall":
                thing["cutout"] = _is_cutout(info, g, tx, a, b, bands)
            found.append(thing)
            # (a thing drawn on a counter's top, its own pixels tall and narrow on the
            # band's colour: a flower, a sign: standing on the top, at its front edge)
            if kind in ("free", "against-wall") and n_rows(a, b) >= 2 and not thing["cutout"] and not shared:
                top_band = next((r for r in range(a, b + 1) if (tx, r) in bands), None)
                if top_band is not None and top_band < b and _is_cutout(info, g, tx, top_band, top_band, bands, outlines=True):
                    found.append({"x": tx, "rows": [top_band, top_band], "kind": "on-band", "cutout": True, "under": len(found) - 1})
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
    """A thing's own pixels in a tile: all but the background's: the floor's shades; on the
    wall's rows, the plain wall's own pixel at that height (its bands run across, so a
    thing's dark outline is its own where the wall at that height is cream); a counter's
    colour and edges under what stands on it."""
    sprite = info["sprite"]
    ox, oy = g["ox"], g["oy"]
    floor_colours = {tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in info["floor_colours"]}
    wall_last = info["wall_first"] + info["wall_rows"] - 1
    plain = info.get("made_up_wall", {}).get(ty) if ty <= wall_last else None
    band = bands.get((tx, ty)) if (kind == "on-band" and bands) else None
    edges = _outline_pixels(sprite, g, tx, ty) if kind == "on-band" else set()
    own = set()
    y0 = oy + ty * PX
    x0 = ox + tx * PX
    match = _wall_match(info, g, tx, ty) if plain else None
    for y in range(y0, y0 + PX):
        wall_pixel = plain[y - y0] if plain else None
        for x in range(x0, x0 + PX):
            c = sprite.colour(x, y)
            if c is None or (x, y) in edges:
                continue
            if plain is None and _alike(c, floor_colours):
                continue
            if c == wall_pixel or (band is not None and c == band):
                continue
            if plain is not None and (x, y) in _decorations(info, g):
                continue                              # (the wall's own decoration: a scallop)
            if match is not None and any(c == sprite.colour(ox + wx * PX + (x - x0), y) for wx in _wall_tiles(info, g, ty)):
                continue                              # (the wall's own pattern there: its scallops)
            own.add((x, y))
    if kind == "on-band":
        # (the counter's side edges down the tile's outer columns are the counter's)
        for x in (x0, x0 + 1, x0 + PX - 2, x0 + PX - 1):
            run = [y for y in range(y0, y0 + PX) if (x, y) in own]
            if len(run) >= 10:
                own -= {(x, y) for y in run}
    return own


def _touching(info, g, tx, a, b):
    """Whether a tile's own pixels touch those of the tile under it across their edge."""
    above = _own_pixels(info, g, tx, a, {}, "thing")
    below = _own_pixels(info, g, tx, b, {}, "thing")
    y_last = g["oy"] + a * PX + PX - 1
    edge = {x for (x, y) in above if y == y_last}
    return any(y == y_last + 1 and (x in edge or x - 1 in edge or x + 1 in edge) for (x, y) in below)


def _own_counts(info, g, tx, ty):
    """A tile's own colours, counted: what's drawn there but the floor and the wall."""
    cache = info.setdefault("_own_counts", {})
    if (tx, ty) not in cache:
        sprite = info["sprite"]
        cache[(tx, ty)] = Counter(sprite.colour(x, y) for (x, y) in _own_pixels(info, g, tx, ty, {}, "thing"))
    return cache[(tx, ty)]


def _wall_tiles(info, g, ty):
    """The columns of a wall row's plain tiles (whole, labelled the wall)."""
    cache = info.setdefault("_wall_tiles", {})
    if ty not in cache:
        sprite = info["sprite"]
        y0 = g["oy"] + ty * PX
        cache[ty] = [cx for cx in range(1, g["cols"] - 1) if info["labels"][ty][cx] == "wall"
                     and float(sprite.drawn[y0:y0 + PX, g["ox"] + cx * PX:g["ox"] + cx * PX + PX].mean()) >= 0.99]
    return cache[ty]


def _wall_match(info, g, tx, ty):
    """The plain wall tile of the row most like a thing's tile, pixel for pixel (the
    wall's pattern changes from tile to tile: a scallop, a band); None if the row has none."""
    cache = info.setdefault("_wall_match", {})
    if (tx, ty) in cache:
        return cache[(tx, ty)]
    sprite = info["sprite"]
    ox, oy = g["ox"], g["oy"]
    x0, y0 = ox + tx * PX, oy + ty * PX
    best, score = None, -1
    for cx in range(g["cols"]):
        if info["labels"][ty][cx] != "wall" or cx == tx:
            continue
        c0 = ox + cx * PX
        if float(sprite.drawn[y0:y0 + PX, c0:c0 + PX].mean()) < 0.99:
            continue                                  # (a corner cut by the void)
        s = sum(1 for j in range(PX) for i in range(PX) if sprite.colour(x0 + i, y0 + j) == sprite.colour(c0 + i, y0 + j))
        if s > score:
            best, score = (cx, ty), s
    cache[(tx, ty)] = best
    return best


def _decorations(info, g):
    """The wall's own decorations on its rows (a scallop): the pixels that aren't the plain
    wall at their height, in small patches made only of the wall's own light colours (the
    plain wall's and the room's corners', where the art draws them too), touching nothing
    else that isn't the wall."""
    if "_decor" in info:
        return info["_decor"]
    sprite = info["sprite"]
    first = info["wall_first"]
    made = info.get("made_up_wall", {})
    y0 = g["oy"] + first * PX
    y1 = g["oy"] + (first + info["wall_rows"]) * PX
    palette = {c for row in made.values() for c in row if c is not None and not _dark(c)}
    for cx in (g["ox"], g["ox"] + (g["cols"] - 1) * PX):
        for y in range(y0, y1):
            for x in range(cx, cx + PX):
                c = sprite.colour(x, y)
                if c is not None and not _dark(c):
                    palette.add(c)
    odd = set()
    for y in range(y0, y1):
        plain = made.get((y - g["oy"]) // PX)
        for x in range(g["ox"] + PX, g["ox"] + (g["cols"] - 1) * PX):
            c = sprite.colour(x, y)
            if c is not None and plain and c != plain[(y - g["oy"]) % PX]:
                odd.add((x, y))
    walled = {p for p in odd if sprite.colour(*p) in palette}
    decor = set()
    left = set(walled)
    while left:
        start = left.pop()
        patch = {start}
        todo = [start]
        while todo:
            x, y = todo.pop()
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if q in left:
                    left.discard(q)
                    patch.add(q)
                    todo.append(q)
        touches = any((x + dx, y + dy) in odd and (x + dx, y + dy) not in walled
                      for (x, y) in patch for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        if len(patch) <= 40 and not touches:
            decor |= patch
    info["_decor"] = decor
    return decor


def _is_wall_pixel(info, g, x, y):
    """Whether a pixel on the wall's rows is the wall's own: the plain wall's at its height,
    or any plain wall tile's of its row at that spot (a scallop)."""
    sprite = info["sprite"]
    first = info["wall_first"]
    wall_last = first + info["wall_rows"] - 1
    ty = (y - g["oy"]) // PX
    if not first <= ty <= wall_last:
        return False
    c = sprite.colour(x, y)
    plain = info.get("made_up_wall", {}).get(ty)
    j = (y - g["oy"]) % PX
    return (bool(plain) and plain[j] == c) or (x, y) in _decorations(info, g)


def _into_wall(info, g, own, reach=16):
    """A thing's pixels on the wall's rows past its tiles: the drawn pixels touching it that
    aren't the wall's own (a map's frame up into the row above, a plant's leaves), as far
    as `reach` beyond what it had."""
    sprite = info["sprite"]
    first = info["wall_first"]
    wall_top = g["oy"] + first * PX
    wall_end = g["oy"] + (first + info["wall_rows"]) * PX
    xs = [p[0] for p in own]
    ys = [p[1] for p in own]
    lo_x, hi_x, lo_y = min(xs) - reach, max(xs) + reach, min(ys) - reach
    taken = info.setdefault("_taken", set()) - set(own)
    out = set()
    todo = list(own)
    seen = set(own)
    while todo:
        x, y = todo.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                q = (x + dx, y + dy)
                if q in seen:
                    continue
                seen.add(q)
                qx, qy = q
                if not (lo_x <= qx <= hi_x and max(lo_y, wall_top) <= qy < wall_end) or q in taken:
                    continue
                if not (g["ox"] + PX <= qx < g["ox"] + (g["cols"] - 1) * PX):
                    continue                          # (not the room's corners: their art is the wall's)
                c = sprite.colour(qx, qy)
                if c is None or _is_wall_pixel(info, g, qx, qy):
                    continue
                out.add(q)
                todo.append(q)
    info["_taken"] |= out
    return out


def _holes(info, own, bands, keep=False):
    """The holes in a thing: drawn pixels with its own pixels to their left and right and
    above and below them (a dark line of a tray the wall's line matched, a slit the counter's
    edge matched, the counter seen through a machine's lip). Each with the colour it takes:
    its own, or, where it's the background's (the counter's, the floor's, the wall's), the
    nearest body colour of the thing in its row."""
    sprite = info["sprite"]
    rows, cols = {}, {}
    for (x, y) in own:
        rows.setdefault(y, []).append(x)
        cols.setdefault(x, []).append(y)
    floor_colours = {tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in info["floor_colours"]}
    wall_colours = {tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in info["wall_colours"]}
    background = set(bands.values()) | wall_colours
    top = min(rows)
    out = {}
    for y, xs in rows.items():
        lo, hi = min(xs), max(xs)
        for x in range(lo + 1, hi):
            if (x, y) in own or x not in cols:
                continue
            c = sprite.colour(x, y)
            ys = cols[x]
            top_edge = y <= top + 1 and (x, y + 1) in own   # (a gap in its top edge, closed from below)
            if c is None or not (min(ys) < y or top_edge):
                continue                              # (the thing's own above it, and to its sides)
            if not keep and (c in background or _alike(c, floor_colours)):
                near = sorted((abs(xx - x), xx) for xx in xs if not _dark(sprite.colour(xx, y)))
                c = sprite.colour(near[0][1], y) if near else c
            out[(x, y)] = c
    return out


def _drop_shadow(sprite, own):
    """The shadow the GBA draws on the floor beside a thing: neutral grey pixels outside its
    dark outline, to the right of it in a row or below it in a column."""
    def shade(c):
        h, s, v = hsv(c)
        return s < 0.12 and 0.3 < v < 0.65 and not _dark(c)
    rows, cols = {}, {}
    for (x, y) in own:
        rows.setdefault(y, []).append(x)
        cols.setdefault(x, []).append(y)
    out = set()
    for y, xs in rows.items():
        darks = [x for x in xs if _dark(sprite.colour(x, y))]
        past = [x for x in xs if darks and x > max(darks)]
        if past and all(shade(sprite.colour(x, y)) for x in past):
            out |= {(x, y) for x in past}
    for x, ys in cols.items():
        darks = [y for y in ys if _dark(sprite.colour(x, y))]
        past = [y for y in ys if darks and y > max(darks)]
        if past and all(shade(sprite.colour(x, y)) for y in past):
            out |= {(x, y) for y in past}
    # (and the grey touching it with no outline to its right or below: the shadow's corner,
    # where neither its row nor its column has an outline to be past)
    grown = True
    while grown:
        grown = False
        for (x, y) in own - out:
            if not shade(sprite.colour(x, y)):
                continue
            if not any((x + dx, y + dy) in out for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                continue
            if any(_dark(sprite.colour(xx, y)) for xx in rows[y] if xx > x) or any(_dark(sprite.colour(x, yy)) for yy in cols[x] if yy > y):
                continue
            out.add((x, y))
            grown = True
    return out


def _main_pieces(pixels, share=0.2):
    """The connected pieces of a set of pixels (8-connected) at least `share` the size of
    the biggest: a flower without the slivers of the counter's edge beside it."""
    left = set(pixels)
    pieces = []
    while left:
        start = left.pop()
        piece = {start}
        todo = [start]
        while todo:
            x, y = todo.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (x + dx, y + dy)
                    if q in left:
                        left.discard(q)
                        piece.add(q)
                        todo.append(q)
        pieces.append(piece)
    biggest = max(len(p) for p in pieces)
    out = set()
    for p in pieces:
        if len(p) >= share * biggest:
            out |= p
    return out


def objects(info, g, things):
    """The things as objects: runs side by side with the same rows and kind are one (a
    counter, a table); each with its tiles, its own pixels, and the box round them
    ("bbox": [x0, y0, x1, y1] in the picture's pixels, y1 and x1 past the last)."""
    bands = info.get("bands", {})
    by_run = {}
    for k, t in enumerate(things):
        by_run[k] = {"kind": t["kind"], "rows": tuple(t["rows"]), "x": t["x"], "runs": [k], "cutout": t.get("cutout", False),
                     "under": t.get("under"), "foot_on_band": t.get("foot_on_band", False)}
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
                        and _family(_own_counts(info, g, x, o["rows"][1]), _own_counts(info, g, x + 1, o["rows"][1]))), None)
            if nxt is None:
                break
            group.append(nxt)
            taken.add(nxt)
            x += 1
        groups.append(group)
    # (a counter's pieces, its wings and its front, one object: the runs of the band's own
    # colour that touch, standing in the room)
    band_colours = set(bands.values())

    def main_of(k):
        t = things[k]
        c = Counter()
        for ty in range(t["rows"][0], t["rows"][1] + 1):
            for col, n in info["counts"][(t["x"], ty)].items():
                if not _dark(col):
                    c[col] += n
        return c.most_common(1)[0][0] if c else None
    counter_runs = [k for k in by_run if not by_run[k]["cutout"] and not by_run[k]["foot_on_band"]
                    and by_run[k]["kind"] in ("free", "against-wall")
                    and any((by_run[k]["x"], ty) in bands for ty in range(by_run[k]["rows"][0], by_run[k]["rows"][1] + 1))]
    merged = []
    left = set(counter_runs)
    while left:
        seed = left.pop()
        part = {seed}
        todo = [seed]
        while todo:
            a_ = todo.pop()
            ta = things[a_]
            for b_ in list(left):
                tb = things[b_]
                if abs(ta["x"] - tb["x"]) <= 1 and ta["rows"][0] <= tb["rows"][1] + 1 and tb["rows"][0] <= ta["rows"][1] + 1:
                    left.discard(b_)
                    part.add(b_)
                    todo.append(b_)
        if len(part) > 1:
            merged.append(sorted(part))
    in_merged = {k for part in merged for k in part}
    groups = [gr for gr in groups if not any(k in in_merged for k in gr)] + merged
    for part in merged:
        for k in part:
            by_run[k]["kind"] = "free"
    claimed = set()
    for k in by_run:
        t = things[k]
        for ty in range(t["rows"][0], t["rows"][1] + 1):
            claimed |= _own_pixels(info, g, t["x"], ty, bands, "thing")
    info["_taken"] = claimed
    out = []
    for group in groups:
        first = dict(by_run[group[0]])
        first["rows"] = (min(by_run[k]["rows"][0] for k in group), max(by_run[k]["rows"][1] for k in group))
        a, b = first["rows"]
        tiles = [(by_run[k]["x"], ty) for k in group for ty in range(by_run[k]["rows"][0], by_run[k]["rows"][1] + 1)]
        own = set()
        block = set()
        for tx, ty in tiles:
            on_counter = first["foot_on_band"] and ty == b
            own |= _own_pixels(info, g, tx, ty, bands, "on-band" if on_counter else first["kind"])
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
        shadow = set()
        if first["cutout"] and own:
            shadow = _drop_shadow(sprite, own)
            own = _main_pieces(own - shadow)
            # (on to the tile below through its pixels that touch: a flower's pot reaching
            # down over the counter's next row)
            below = [(tx, b + 1) for tx in {t[0] for t in tiles} if b + 1 < len(info["labels"])]
            more = set()
            for btx, bty in below:
                more |= _own_pixels(info, g, btx, bty, bands, "on-band" if first["kind"] == "on-band" else first["kind"])
            shadow |= _drop_shadow(sprite, own | more)
            more -= shadow
            grown = True
            while grown and more:
                grown = False
                for p in list(more):
                    if any((p[0] + dx, p[1] + dy) in own for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                        own.add(p)
                        more.discard(p)
                        grown = True
        elif own:
            shadow = _drop_shadow(sprite, own)
            own -= shadow
        art = set(own)                                # (as drawn: what its fingerprint is of)
        if own and any(ty <= info["wall_first"] + info["wall_rows"] - 1 for _, ty in tiles) or \
                (own and any(ty == info["wall_first"] + info["wall_rows"] for _, ty in tiles) and first["kind"] in ("against-wall", "on-wall")):
            own |= _into_wall(info, g, own)
        fill = {}
        if own:
            fill = _holes(info, own, bands, keep=first["cutout"])
            own |= set(fill)
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
        out.append({"art": art, "fill": fill, "shadow": shadow - own, "foot_on_band": first["foot_on_band"],
                    "kind": kind, "rows": [a, b], "columns": [min(by_run[k]["x"] for k in group), max(by_run[k]["x"] for k in group)],
                    "tiles": tiles, "own": own, "bbox": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
                    "cutout": first["cutout"], "under": first["under"], "runs": group})
    return escalators(info, g, out)


def _front_rows(info, obj):
    """How many of a thing's own rows are its front, as the GBA draws it: from its foot up
    to where its top's own colour takes over (the commonest colour of its upper half,
    outlines aside): a counter's darker edge and its kickplate, a cushion's band."""
    sprite = info["sprite"]
    x0, y0, x1, y1 = obj["bbox"]
    upper = Counter()
    for (x, y) in obj["own"]:
        if y < (y0 + y1) / 2.0:
            c = sprite.colour(x, y)
            if c is not None and not _dark(c):
                upper[c] += 1
    if not upper:
        return (y1 - y0) // 2
    top_colour = upper.most_common(1)[0][0]
    n = 0
    for y in range(y1 - 1, y0 - 1, -1):
        row = Counter(sprite.colour(x, y) for x in range(x0, x1) if (x, y) in obj["own"])
        light = Counter({c: m for c, m in row.items() if c is not None and not _dark(c)})
        if light and light.get(top_colour, 0) * 2 >= sum(light.values()):
            break
        n += 1
    return n if n < (y1 - y0) else (y1 - y0) // 2

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


# --- the room in true voxels: a cube a pixel ---

def _lid_line(sprite, o):
    """The row of the first dark line drawn right across a thing below its top's outline:
    what's above it is its lid, lying; None if there's none."""
    x0, y0, x1, y1 = o["bbox"]
    started = False
    for v in range(y0, y1):
        px = [sprite.colour(x, v) for x in range(x0, x1) if (x, v) in o["own"]]
        px = [c for c in px if c is not None]
        if not px:
            continue
        dark = sum(1 for c in px if _dark(c))
        if not started:
            started = dark * 10 < len(px) * 6           # (past the outline along its top)
            continue
        if dark * 10 >= len(px) * 6 and len(px) * 10 >= (x1 - x0) * 6:
            return v
    return None


def fingerprint(sprite, o, g=None):
    """A thing's fingerprint: its tiles' pictures as drawn, from its first tile (as LGFR3D's
    prints are a tile's picture), so it holds however its pixels are told from what's
    behind them, and in every room drawn with the same tiles. `g`: the room's grid (its
    offset on the picture); the grid at the picture's corner if not given."""
    import hashlib
    ox, oy = (g["ox"], g["oy"]) if g else (0, 0)
    tiles = sorted(set(map(tuple, o["tiles"])))
    tx0 = min(t[0] for t in tiles)
    ty0 = min(t[1] for t in tiles)
    h = hashlib.sha1()
    for tx, ty in tiles:
        h.update(b"%d,%d:" % (tx - tx0, ty - ty0))
        h.update(sprite.rgba[oy + ty * PX:oy + ty * PX + PX, ox + tx * PX:ox + tx * PX + PX].tobytes())
    return h.hexdigest()[:12]


def load_fixes(paths=(), spec=None):
    """The fixes that apply: the library shipped with the tool (fixes/rooms.json), any files
    given, and the spec's "objects" ({fingerprint or "tx,ty": fix}), later ones winning."""
    import json
    import os
    fixes = {}
    lib = os.path.join(os.path.dirname(__file__), "fixes", "rooms.json")
    for path in [lib] + list(paths):
        if path and os.path.exists(path):
            for key, fix in json.load(open(path)).items():
                if not key.startswith("_"):
                    fixes[key] = fix
    for key, fix in ((spec or {}).get("objects") or {}).items():
        fixes[key] = fix
    return fixes


def fix_for(sprite, o, fixes, g=None):
    return fixes.get(fingerprint(sprite, o, g)) or fixes.get("%d,%d" % (o["columns"][0], o["rows"][0])) or {}


def fix_recess(o, fix):
    """A fix's "recess": {(x, v): depth} of a thing's own pixels, its rows and columns from
    the thing's box's corner."""
    out = {}
    x0, y0 = o["bbox"][0], o["bbox"][1]
    for r in fix.get("recess", []):
        for (x, v) in o["own"]:
            if r["cols"][0] <= x - x0 <= r["cols"][1] and r["rows"][0] <= v - y0 <= r["rows"][1]:
                out[(x, v)] = int(r.get("depth", 2))
    return out


def fix_raised(sprite, o, fix):
    """A fix's "raise" (one or a list): its pixels of those colours, within its rows and
    columns."""
    out = set()
    x0, y0 = o["bbox"][0], o["bbox"][1]
    rs = fix.get("raise") or []
    for r in ([rs] if isinstance(rs, dict) else rs):
        cols_ = {tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in r.get("colours", [])}
        rows_ = r.get("rows", [-10 ** 6, 10 ** 6])
        span = r.get("cols", [-10 ** 6, 10 ** 6])
        out |= {p for p in o["own"] if sprite.colour(*p) in cols_ and rows_[0] <= p[1] - y0 <= rows_[1]
                and span[0] <= p[0] - x0 <= span[1]}
    return out


def fix_dropped(o, fix):
    """A fix's "drop": its own pixels in those rows and columns."""
    x0, y0 = o["bbox"][0], o["bbox"][1]
    every = [-10 ** 6, 10 ** 6]
    out = set()
    for d in fix.get("drop", []):
        out |= {p for p in o["own"] if d.get("cols", every)[0] <= p[0] - x0 <= d.get("cols", every)[1]
                and d.get("rows", every)[0] <= p[1] - y0 <= d.get("rows", every)[1]}
    return out


def fix_included(sprite, o, fix):
    """A fix's "include": the drawn pixels in those rows and columns, its own too."""
    x0, y0 = o["bbox"][0], o["bbox"][1]
    out = set()
    for d in fix.get("include", []):
        out |= {(x0 + i, y0 + j) for j in range(d["rows"][0], d["rows"][1] + 1)
                for i in range(d["cols"][0], d["cols"][1] + 1) if sprite.colour(x0 + i, y0 + j) is not None}
    return out


def build_voxels(sprite, g, info, things, front_height="auto", low_height=8.0, wall_thick=4, spec=None, thick=2, fixes=None, round_depth=0.5, outlines=True):
    """The room as cubes (voxel.Grid), owned by "floor", "wall", "side-left", "side-right"
    and "thing-N" (N an index into objects()). See voxel.py for how a sprite is unprojected.
    `fixes` ({fingerprint or "tx,ty": fix}, see load_fixes) set a thing's shape by hand:
    "profile" ([["up", n], ["back", n], ..., [kind, "rest"]] up from its foot), "recess"
    ([{"rows": [a, b], "cols": [c, d], "depth": k}] from its box's corner), "raise"
    ({"colours": ["rrggbb"]}, or a list), "shape" ("box", "round", "slab", "escalator"),
    "front" (rows), "back_to_wall" (true or false), "keep" (closed holes keep the picture's
    colours), "outline" (false: none). `outlines`: every thing outlined round its sides and
    back (voxel.outline)."""
    from . import voxel
    spec = dict(spec or {})
    fixes = fixes if fixes is not None else load_fixes(spec=spec)
    ox, oy, cols, rows = g["ox"], g["oy"], g["cols"], g["rows"]
    labels = info["labels"]
    first = info["wall_first"]
    wall_last = first + info["wall_rows"] - 1
    wall_h = info["wall_rows"] * PX
    wall_top = oy + first * PX
    wall_z = oy + (wall_last + 1) * PX
    made = info.get("made_up_wall", {})
    floor_colours = {tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in info["floor_colours"]}
    objs = objects(info, g, things)
    grid = voxel.Grid()
    covered = set()
    shaded = set()                         # (a thing's drop shadow: on the wall, the thing casts it)
    for o in objs:
        if o["kind"] != "flat":
            covered |= o["own"]
            shaded |= o.get("shadow", set())

    def wall_colour(r_rel):
        ty = first + r_rel // PX
        row = made.get(ty)
        c = row[r_rel % PX] if row else None
        return c or (200, 190, 170)

    # the floor
    floor_tiles = [tuple(t) for t in info.get("floor_tiles", [])]

    def under(tx, ty):
        x0, y0 = ox + tx * PX, oy + ty * PX
        best, score = None, -1
        for cand in floor_tiles:
            cx, cy = ox + cand[0] * PX, oy + cand[1] * PX
            s = sum(1 for j in range(PX) for i in range(PX)
                    if sprite.colour(x0 + i, y0 + j) is not None and _alike(sprite.colour(x0 + i, y0 + j), floor_colours)
                    and sprite.colour(cx + i, cy + j) == sprite.colour(x0 + i, y0 + j))
            if s > score:
                best, score = cand, s
        return best
    for ty in range(wall_last + 1, rows):
        for tx in range(cols):
            if labels[ty][tx] not in ("floor", "thing", "wall_side"):
                continue
            x0, y0 = ox + tx * PX, oy + ty * PX
            alt = None
            for j in range(PX):
                for i in range(PX):
                    x, v = x0 + i, y0 + j
                    c = sprite.colour(x, v)
                    if c is None:
                        continue
                    if (x, v) in covered:
                        alt = alt or under(tx, ty)
                        c = sprite.colour(ox + alt[0] * PX + i, oy + alt[1] * PX + j) if alt else c
                    grid.set(x, -1, v, c, "floor")
    # the back wall, stepping round the corners
    shift = {}
    for x in range(ox, ox + cols * PX):
        ys = [v for v in range(wall_top, wall_z + PX) if 0 <= v < sprite.H and sprite.colour(x, v) is not None]
        if not ys or info["wall_rows"] == 0:
            continue
        tx = (x - ox) // PX
        if not any(labels[ty][tx] in ("wall", "thing") for ty in range(first, wall_last + 1)) and ys[0] >= wall_z:
            continue
        shift[x] = max(0, ys[0] - wall_top)
    for x, d in shift.items():
        z_face = wall_z - 1 + d
        for y in range(wall_h):
            r = z_face - y
            r_rel = r - d - wall_top
            c = sprite.colour(x, r) if 0 <= r < sprite.H else None
            if c is None or (x, r) in covered or (x, r) in shaded:
                c = wall_colour(r_rel)
            for k in range(int(wall_thick)):
                grid.set(x, y, z_face - k, c if k == 0 or y == wall_h - 1 else wall_colour(r_rel), "wall")
    for side, tx in (("side-left", 0), ("side-right", cols - 1)):
        tys = [ty for ty in range(wall_last + 1, rows) if labels[ty][tx] == "wall_side"]
        if not tys:
            continue
        x0 = ox + tx * PX
        xs = range(x0, x0 + int(wall_thick)) if tx == 0 else range(x0 + PX - int(wall_thick), x0 + PX)
        edge = x0 if tx == 0 else x0 + PX - 1
        z_start = wall_z + shift.get(edge, 0)
        for ty in tys:
            for z in range(max(z_start, oy + ty * PX), oy + ty * PX + PX):
                for x in xs:
                    for y in range(wall_h):
                        grid.set(x, y, z, wall_colour(wall_h - 1 - y), side)

    # the things
    heights = {}

    def front_h(o, fix=None):
        fix = fix or {}
        n_rows = o["rows"][1] - o["rows"][0] + 1
        if "front" in fix:
            h = float(fix["front"])
        elif "front_height" in spec:
            h = float(spec["front_height"])
        elif front_height == "auto":
            h = float(_front_rows(info, o))
            if h <= 1:
                h = float(low_height if n_rows == 1 else PX)
        else:
            h = float(front_height if n_rows > 1 else low_height)
        return int(round(max(1.0, min(h, float(o["bbox"][3] - o["bbox"][1])))))

    def under_object(o):
        if o["kind"] == "on-band":
            return next((u for u in range(len(objs)) if o["under"] in objs[u]["runs"]), None)
        if o.get("foot_on_band"):
            return next((u for u in range(len(objs)) if objs[u]["kind"] == "free" and
                         any((cx, o["rows"][1]) in objs[u]["tiles"] for cx in range(o["columns"][0], o["columns"][1] + 1))), None)
        return None
    # (what stands on a counter isn't the counter's: its top there is its own colour)
    paint = {}
    for k, o in enumerate(objs):
        u = under_object(o)
        if u is None:
            continue
        band = Counter(info["bands"].get(t) for t in objs[u]["tiles"] if t in info["bands"]).most_common(1)
        colour = band[0][0] if band else None
        for p in o["own"]:
            if p in objs[u]["own"] and colour is not None:
                paint.setdefault(u, {})[p] = colour

    recess_of = fix_recess
    raised_of = lambda o, fix: fix_raised(sprite, o, fix)
    dropped = fix_dropped
    included = lambda o, fix: fix_included(sprite, o, fix)

    standing_on = {u for u in (under_object(o) for o in objs) if u is not None}
    order = sorted(range(len(objs)), key=lambda k: (k in standing_on, objs[k]["kind"] == "free"))
    under_cubes = {}                                    # (counter: the pixels something stands on)
    turned = {}                                         # (a round thing: where it's outlined)
    for k in order:
        o = objs[k]
        fix = fix_for(sprite, o, fixes, g)
        kind = o["kind"]
        owner = "thing-%d" % k
        shape = fix.get("shape") or ("round" if o["cutout"] else "box")
        if kind == "flat":
            continue
        if kind == "on-wall" and not o["cutout"]:
            # (a TV, a map: a box standing proud of the wall, moved up as far as it stands
            # out, so its picture stays where the GBA draws it; as deep as it's tall a fifth)
            x0_, y0_, x1_, y1_ = o["bbox"]
            d = int(fix.get("depth", max(1, min(4, round((y1_ - y0_) / 5.0)))))
            for (x, r) in o["own"]:
                c = o["fill"].get((x, r)) or sprite.colour(x, r)
                z_face = wall_z - 1 + shift.get(x, 0)
                y = z_face - r
                for j in range(1, d + 1):
                    grid.set(x, y + d, z_face + j, c, owner)
            continue
        own_set = (set(o["own"]) | included(o, fix)) - dropped(o, fix)
        u = under_object(o)
        yb = 0
        if u is not None:
            yb = heights.get(u) or front_h(objs[u], fix_for(sprite, objs[u], fixes, g))
        part = voxel.Grid()
        if shape == "escalator" or kind == "escalator":
            shape = "escalator"
            build_escalator(part, sprite, o, fix, owner)
        elif shape == "plant":
            build_plant(part, sprite, o, own_set, fix, owner, recess=recess_of(o, fix), raised=raised_of(o, fix))
            if u is not None:
                under_cubes.setdefault(u, set()).update(own_set)
        elif shape in ("round", "slab"):
            foot = max(v for _, v in own_set)
            z_foot = foot + 1 + yb
            if shape == "slab":
                voxel.unproject_slab(part, sprite, own_set, yb, owner, thick=int(thick))
            else:
                xs = [x for x, _ in own_set]
                bottom = [x for x, v in own_set if v >= foot - 2]
                r_foot = (max(bottom) - min(bottom) + 1) / 2.0 if bottom else 2.0
                r_all = (max(xs) - min(xs) + 1) / 2.0 * float(fix.get("round_depth", round_depth))
                z_axis = max(z_foot - 1 - r_foot * float(fix.get("round_depth", round_depth)), wall_z + r_all)
                voxel.lathe(part, sprite, own_set, yb, z_axis, owner, depth=float(fix.get("round_depth", round_depth)))
                turned[k] = voxel.lathe_outlined(sprite, own_set, yb)
            if u is not None:
                under_cubes.setdefault(u, set()).update(own_set)
        else:
            if "profile" in fix or "profiles" in fix:
                kinds = voxel.segments_kinds(fix.get("profile", [["up", "rest"]]), o["bbox"][3])
                if "profiles" in fix:
                    kinds = voxel.columns_kinds(fix["profiles"], kinds, o["bbox"][0], o["bbox"][3])
            elif kind == "against-wall":
                line = _lid_line(sprite, o)
                kinds = voxel.box_kinds(lambda x, top, zf, line=line, yb=yb: min(zf - top, max(1, (zf - line) if line is not None else 0, wall_z - top - yb)), own_set)
            elif kind == "tall":
                kinds = voxel.box_kinds(lambda x, top, zf: max(1, zf - top - PX), own_set)
            else:
                h = front_h(o, fix)
                heights[k] = h
                kinds = voxel.box_kinds(lambda x, top, zf, h=h: h, own_set)
            to_wall = fix.get("back_to_wall", kind == "against-wall")
            back_of = (lambda x, yb=yb: wall_z - yb) if to_wall else None
            stretch = float(fix.get("stretch", 1.0))
            if fix.get("square"):
                # (its top as deep as it's wide: the rows the middle column lies, stretched)
                x0_, _, x1_, y1_ = o["bbox"]
                xm = (x0_ + x1_) // 2
                lying = sum(1 for v in range(o["bbox"][1], y1_) if (xm, v) in own_set and kinds(xm, v) == "back")
                stretch = max(1.0, (x1_ - x0_) / float(lying)) if lying else 1.0
            shown = voxel.unproject_profile(part, sprite, own_set, kinds, back_of=back_of, owner=owner,
                                            recess=recess_of(o, fix), raised=raised_of(o, fix),
                                            colours=None if fix.get("keep") else o.get("fill"),
                                            stretch=stretch)
            if u is not None:
                under_cubes.setdefault(u, set()).update(shown)
            if k in paint:
                # (the counter's top under what stands on it: its own colour; where it lies
                # flat on the counter (a mat), the counter wears it)
                for p, colour in paint[k].items():
                    if p not in under_cubes.get(k, set()):
                        continue
                    x, v = p
                    for y in range(0, 64):
                        z = v + y + 1                  # (the cube whose top shows row v)
                        if (x, y, z) in part.cubes and (x, y + 1, z) not in part.cubes:
                            part.set(x, y, z, colour, owner)
                            break
        for (x, y, z), c in part.cubes.items():
            if shape != "escalator" and z + yb < wall_z:
                continue                              # (nothing behind the wall's face)
            if shape in ("round", "slab", "escalator"):
                grid.set(x, y, z, c, owner)
            else:
                grid.set(x, y + yb, z + yb, c, owner)
    # (every thing outlined from the sides and the back as the GBA outlines it from the
    # front: an escalator and what lies flat aside)
    hex_ = lambda h: tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    for k, o in enumerate(objs):
        fix = fix_for(sprite, o, fixes, g)
        if o["kind"] in ("flat", "escalator") or fix.get("shape") == "escalator":
            continue
        if "round" in fix:
            voxel.round_corners(grid, "thing-%d" % k, int(fix["round"]))
        if "sides" in fix:
            # (its sides painted as the fix says: a cabinet's plain sides, a panel inset)
            sides = fix["sides"]
            for which, side in (((0, 1), sides), ((0,), sides.get("left")), ((1,), sides.get("right"))):
                if side and "colour" in side:
                    voxel.paint_sides(grid, "thing-%d" % k, hex_(side["colour"]),
                                      hex_(side["panel"]) if "panel" in side else None, which)
        if outlines and fix.get("outline", True):
            voxel.outline(grid, "thing-%d" % k, voxel._darkest(sprite, o["own"]),
                          keep_shown=k not in turned, where=turned.get(k))
    return grid, objs


def _treads(sprite, x0, y0):
    """Whether a tile is drawn as treads: four or more rows of short light dashes on dark,
    each between dark rows (an escalator's steps, a staircase's)."""
    n = 0
    for j in range(PX):
        cs = [sprite.colour(x0 + i, y0 + j) for i in range(PX)]
        if any(c is None for c in cs):
            continue
        light = [not _dark(c) and hsv(c)[1] < 0.2 for c in cs]
        runs_ = []
        run = 0
        for f in light + [False]:
            if f:
                run += 1
            elif run:
                runs_.append(run)
                run = 0
        if len(runs_) >= 2 and all(2 <= r <= 9 for r in runs_) and sum(light) <= 12:
            above = [sprite.colour(x0 + i, y0 + j - 1) for i in range(PX)] if j > 0 else []
            if above and sum(1 for c in above if c is not None and _dark(c)) >= 10:
                n += 1
    return n >= 4


def escalators(info, g, objs):
    """The things drawn as an escalator, one object each: the things whose tiles are treads,
    and those touching them (its balustrades), merged; their kind "escalator"."""
    sprite = info["sprite"]
    tread = {k for k, o in enumerate(objs) if not o["cutout"]
             and any(_treads(sprite, g["ox"] + tx * PX, g["oy"] + ty * PX) for tx, ty in o["tiles"])}
    if not tread:
        return objs
    merged = set(tread)
    grew = True
    while grew:
        grew = False
        tiles = {t for k in merged for t in objs[k]["tiles"]}
        for k, o in enumerate(objs):
            if k in merged or o["cutout"] or o["kind"] not in ("free", "tall"):
                continue
            if any(abs(tx - ax) + abs(ty - ay) == 1 for tx, ty in o["tiles"] for ax, ay in tiles):
                merged.add(k)
                grew = True
    # (one escalator per touching group)
    groups = []
    left = set(merged)
    while left:
        seed = left.pop()
        part = {seed}
        todo = [seed]
        while todo:
            a = todo.pop()
            for b in list(left):
                if any(abs(tx - ax) + abs(ty - ay) <= 1 for tx, ty in objs[a]["tiles"] for ax, ay in objs[b]["tiles"]):
                    left.discard(b)
                    part.add(b)
                    todo.append(b)
        groups.append(sorted(part))
    out = [o for k, o in enumerate(objs) if k not in merged]
    for part in groups:
        own = set().union(*(objs[k]["own"] for k in part))
        tiles = [t for k in part for t in objs[k]["tiles"]]
        xs = [p[0] for p in own]
        ys = [p[1] for p in own]
        out.append({"art": own, "fill": {}, "shadow": set(), "foot_on_band": False, "kind": "escalator", "rows": [min(t[1] for t in tiles), max(t[1] for t in tiles)],
                    "columns": [min(t[0] for t in tiles), max(t[0] for t in tiles)], "tiles": tiles, "own": own,
                    "bbox": [min(xs), min(ys), max(xs) + 1, max(ys) + 1], "cutout": False, "under": None,
                    "runs": [r for k in part for r in objs[k]["runs"]]})
    return out


def build_plant(grid, sprite, o, own, fix, owner, recess=None, raised=None):
    """A potted plant, in parts the fix names by rows and columns from the thing's box's
    corner, each where the GBA draws it:
    - its pot from the foot, by the fix's "profile"; its "soil" (a colour, over "rows" and
      "cols") sunk by its "recess"; "pot_depth": its top reaching back that far (a pot whose
      top the leaves hide), a "rim" (a colour) round the soil;
    - its trunk ("trunk": the "cols" and "rows" it's drawn in, its "stripes" the rows whose
      colours repeat up it), a square column standing in the soil, outlined at its corners;
    - its crown ("crown": its "rows", a "profile" and "profiles" up from its foot as a
      thing's): the rows that lie fronds or leaves, level (voxel.canopy), over the trunk's
      middle, or with no trunk on the pot; the rows that stand leaves hanging, tips curling,
      or a flower's head, "thick" cubes deep; "mirror": its leaves in front of what stands
      laid behind it too."""
    from . import voxel
    x0, y0 = o["bbox"][0], o["bbox"][1]
    rel = lambda p, r, c: r[0] <= p[1] - y0 <= r[1] and c[0] <= p[0] - x0 <= c[1]
    every = [-10 ** 6, 10 ** 6]
    hex_ = lambda h: tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    trunk = fix.get("trunk", {})
    t_rows, t_cols = trunk.get("rows", every), trunk.get("cols", every)
    crown = fix.get("crown", {})
    if isinstance(crown, list):
        crown = {"rows": crown}
    crown_px = {p for p in own if rel(p, crown.get("rows", every), every)}
    trunk_px = {p for p in own if "trunk" in fix and rel(p, t_rows, t_cols)} - crown_px
    soil = fix.get("soil")
    soil_px = {p for p in own if soil and "rows" in soil and rel(p, soil["rows"], soil["cols"])} - crown_px
    pot_px = (own - crown_px - trunk_px) | soil_px
    colours = {p: hex_(soil["colour"]) for p in soil_px}
    # the pot
    kinds = voxel.segments_kinds(fix.get("profile", [["up", "rest"]]), o["bbox"][3])
    foot = max(v for _, v in pot_px) if pot_px else o["bbox"][3] - 1
    back_of = (lambda x, z=foot + 1 - int(fix["pot_depth"]): z) if "pot_depth" in fix else None
    pot = voxel.Grid()
    shown = voxel.unproject_profile(pot, sprite, pot_px, kinds, back_of=back_of, owner=owner,
                                    recess=recess, raised=raised, colours=colours)
    if "pot_depth" in fix and pot.cubes:
        # (its top as the picture doesn't show it: the rim round it, the soil inside)
        top = {}
        for (x, y, z) in pot.cubes:
            top[(x, z)] = max(top.get((x, z), -1), y)
        seen = {(x,) + cell for (x, _), cell in shown.items()}      # (the picture's own)
        rim = hex_(fix["rim"]) if "rim" in fix else None
        dirt = hex_(soil["colour"]) if soil else None
        for (x, z), y in top.items():
            if (x, y, z) in seen:
                continue
            edge = any((x + dx, z + dz) not in top for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            c = rim if edge else dirt
            if c is not None:
                pot.cubes[(x, y, z)] = c
    for key, c in pot.cubes.items():
        grid.set(*key, c, owner)
    ground = [shown[p][0] + 1 for p in (soil_px or pot_px) if p in shown]
    y_soil = max(ground) if ground else 0
    # the crown, its fronds level
    bottom = max(v for _, v in crown_px) + 1 if crown_px else y0
    kinds = voxel.segments_kinds(crown.get("profile", [["back", "rest"]]), bottom)
    if "profiles" in crown:
        kinds = voxel.columns_kinds(crown["profiles"], kinds, x0, bottom)
    cells = voxel.canopy(crown_px, kinds, bottom)
    if trunk_px:
        tx0, tx1 = min(x for x, _ in trunk_px), max(x for x, _ in trunk_px)
        w = tx1 - tx0 + 1
        z_front = max(v for _, v in trunk_px) + y_soil
        z_axis = z_front - (w - 1) / 2.0
        over = [z for (x, _), (y, z) in cells.items() if tx0 <= x <= tx1 and y == 0] or [z for _, z in cells.values()]
        lift = int(math.floor(z_axis - sum(over) / float(len(over)) + 0.5)) if over else 0
    else:
        lift = y_soil                                   # (its leaves lying on the pot)
    thick = int(crown.get("thick", 1))
    placed = {}
    for (x, v), (y, z) in cells.items():
        c = sprite.colour(x, v)
        if c is None:
            continue
        standing = kinds(x, v) == "up"
        for j in range(thick if standing else 1):
            placed.setdefault((x, y + lift, z + lift - j), c)
    if crown.get("mirror"):
        stand = [z for (x, y, z) in placed if y > lift]
        if stand:
            mid = (min(stand) + max(stand)) / 2.0
            for (x, y, z), c in list(placed.items()):
                if y == lift and z > mid:
                    placed.setdefault((x, y, int(round(2 * mid - z))), c)
    for key, c in placed.items():
        grid.set(*key, c, owner)
    if not trunk_px:
        return
    # the trunk: from the soil up to the fronds, a square column outlined at its corners
    stripes = trunk.get("stripes", t_rows)
    mid = (tx0 + tx1) // 2
    cycle = [sprite.colour(mid, y0 + r) for r in range(stripes[0], stripes[1] + 1)]
    cycle = [c for c in cycle if c is not None] or [voxel._darkest(sprite, trunk_px)]
    dark = voxel._darkest(sprite, trunk_px)
    for y in range(y_soil, lift):
        c = cycle[(z_front - y - (y0 + stripes[0])) % len(cycle)]
        for x in range(tx0, tx1 + 1):
            for z in range(z_front - w + 1, z_front + 1):
                corner = x in (tx0, tx1) and z in (z_front - w + 1, z_front)
                grid.set(x, y, z, dark if corner else c, owner)


def build_escalator(grid, sprite, o, fix, owner):
    """An escalator: steps between two glass balustrades, a handrail along the top of each,
    rising to the side its handrails rise to in the picture. The handrails are its most
    saturated colour; in each column the upper is the far one, the lower the near one; the
    near balustrade's foot is the escalator's front, how far its handrail is above that its
    height; where the handrails climb in the picture, it climbs as much."""
    own = o["own"]
    x0, y0, x1, y1 = o["bbox"]
    sat = Counter(sprite.colour(*p) for p in own if hsv(sprite.colour(*p))[1] > 0.5 and not _dark(sprite.colour(*p)))
    if not sat:
        return
    rail = sat.most_common(1)[0][0]
    rails = {}
    for (x, v) in own:
        if sprite.colour(x, v) == rail:
            rails.setdefault(x, []).append(v)
    xs = sorted(rails)
    near, far = {}, {}
    for x in xs:
        vs = sorted(rails[x])
        mid = (vs[0] + vs[-1]) / 2.0
        upper = [v for v in vs if v <= mid]
        lower = [v for v in vs if v > mid]
        if upper and lower:
            far[x], near[x] = min(upper), min(lower)
    if not near:
        return
    cols = sorted(near)
    # (the flat end: where the near handrail is lowest in the picture; the climb from there)
    flat_row = max(near.values())
    rises_left = sum(near[x] for x in cols[:len(cols) // 2]) < sum(near[x] for x in cols[len(cols) // 2:])
    z_near = y1                                        # (the near balustrade's foot)
    height = max(2, z_near - 1 - flat_row)             # (its handrail above its foot)
    flat_far = max(far.values())
    z_far = flat_far + height + 1

    def rise(x):
        if x in near:
            return max(0, flat_row - near[x])
        nearest = min(cols, key=lambda c: abs(c - x))
        return max(0, flat_row - near[nearest])
    glass = Counter(sprite.colour(*p) for p in own if not _dark(sprite.colour(*p)) and 0.1 < hsv(sprite.colour(*p))[1] < 0.5
                    and 120 <= hsv(sprite.colour(*p))[0] <= 200)
    glass = glass.most_common(1)[0][0] if glass else (190, 220, 210)
    lights = Counter(sprite.colour(*p) for p in own if not _dark(sprite.colour(*p)) and hsv(sprite.colour(*p))[1] < 0.15)
    tread = lights.most_common(1)[0][0] if lights else (150, 150, 160)
    from .voxel import _darkest as darkest_of
    dark = darkest_of(sprite, own)
    # (the grate's rhythm: which rows of the steps are its light dashes, odd or even)
    rows_with = Counter(v % 2 for (x, v) in own if sprite.colour(x, v) == tread)
    parity = rows_with.most_common(1)[0][0] if rows_with else 0
    end = max(cols) if rises_left else min(cols)
    for x in range(x0, x1):
        if x not in near and not (min(cols) <= x <= max(cols)):
            continue
        r = rise(x)
        # the steps between the balustrades, solid under: each tread's top the grate the
        # art draws there (a cube's top shows on row z - y - 1); where a balustrade is drawn
        # over it, the grate's stripes carried on, every other row
        for z in range(z_far + 1, z_near - 2):
            for y in range(0, r + 1):
                c = dark
                if y == r:
                    row = z - y - 1
                    art = sprite.colour(x, row) if (x, row) in own else None
                    if art is not None and art in (tread, dark):
                        c = art
                    else:
                        c = tread if row % 2 == parity else dark
                grid.set(x, y, z, c, owner)
        # the balustrades: glass, its handrail along the top, its edges dark
        for zb in ((z_far - 1, z_far), (z_near - 2, z_near - 1)):
            for z in zb:
                for y in range(0, r + height + 1):
                    if y >= r + height:
                        c = rail
                    elif y == r + height - 1 or x == end:
                        c = dark
                    else:
                        c = glass
                    grid.set(x, y, z, c, owner)
