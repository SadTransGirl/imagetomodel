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
            "plain_wall": list(plain_wall) if plain_wall else None, "counts": counts, "forced": forced}


def _family(a, b):
    """Whether two tiles, one over the other, look like one thing: their commonest colours
    (outlines aside) shared."""
    def top(c):
        return [col for col, _ in c.most_common() if not _dark(col)][:3]
    ta, tb = top(a), top(b)
    return bool(set(ta) & set(tb))


def _main_colour(c):
    for col, _ in c.most_common():
        if not _dark(col):
            return col
    return None


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

    def band_colour(tx, ty):
        return bands.get((tx, ty))

    def joined(tx, a, b):
        # (whether the tile b, under a, is part of the same thing)
        if a <= wall_last < b:
            return True
        return _family(counts[(tx, a)], counts[(tx, b)])
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
            found.append({"x": tx, "rows": [a, b], "kind": kind})
            ty = b if shared else b + 1
    return found


def build(sprite, g, info, things, front_height=16.0, low_height=8.0, wall_thick=4.0, spec=None):
    """The room's mesh, in the picture's pixels (x across, y up, z down the picture); each
    polygon named for the object it belongs to ("floor", "wall", "side-left", "thing-3")."""
    spec = dict(spec or {})
    ox, oy, cols, rows = g["ox"], g["oy"], g["cols"], g["rows"]
    labels = info["labels"]
    first = info["wall_first"]
    wall_last = first + info["wall_rows"] - 1
    wall_h = info["wall_rows"] * PX
    wall_z = oy + (wall_last + 1) * PX                 # (the back wall's face)
    mesh = Mesh()

    def px(tx, ty):
        return ox + tx * PX, oy + ty * PX

    def plain_uv(y_lo, y_hi):
        """The plain wall tile's middle column, from its foot up: for the walls' other faces."""
        if info["plain_wall"]:
            wx, wy = px(*info["plain_wall"])
            u = wx + PX / 2.0
            return [(u, wy + PX), (u, wy + PX), (u, wy), (u, wy)]
        return [(ox + PX / 2.0, oy + PX)] * 4
    # the floor: flat, each tile its picture (under a thing's back, too)
    for ty in range(rows):
        for tx in range(cols):
            lab = labels[ty][tx]
            if lab in ("floor", "thing", "wall_side"):
                x0, z0 = px(tx, ty)
                mesh.face([(x0, 0, z0), (x0, 0, z0 + PX), (x0 + PX, 0, z0 + PX), (x0 + PX, 0, z0)], name="floor")
    # the back wall: a slab the wall's rows tall and deep, its face the rows a pixel a
    # pixel (a thing hanging on it painted there), its top and ends plain
    wall_cols = [tx for tx in range(cols) if any(labels[ty][tx] in ("wall", "thing") for ty in range(first, wall_last + 1))]
    if wall_cols and info["wall_rows"] > 0:
        for tx in wall_cols:
            x0, _ = px(tx, 0)
            z_back = oy + first * PX
            mesh.face([(x0, 0, wall_z), (x0 + PX, 0, wall_z), (x0 + PX, wall_h, wall_z), (x0, wall_h, wall_z)], name="wall")
            lid_uv = (x0 + PX / 2.0, oy + first * PX + 0.5)
            mesh.face([(x0, wall_h, z_back), (x0, wall_h, wall_z), (x0 + PX, wall_h, wall_z), (x0 + PX, wall_h, z_back)], lid_uv, name="wall")
        xa, _ = px(min(wall_cols), 0)
        xb, _ = px(max(wall_cols) + 1, 0)
        z_back = oy + first * PX
        mesh.face([(xa, 0, z_back), (xa, 0, wall_z), (xa, wall_h, wall_z), (xa, wall_h, z_back)], plain_uv(0, wall_h), name="wall")
        mesh.face([(xb, 0, wall_z), (xb, 0, z_back), (xb, wall_h, z_back), (xb, wall_h, wall_z)], plain_uv(0, wall_h), name="wall")
        mesh.face([(xb, 0, z_back), (xa, 0, z_back), (xa, wall_h, z_back), (xb, wall_h, z_back)], plain_uv(0, wall_h), name="wall")
    # the walls down the sides: the tile's picture flat (the GBA's view of the wall from
    # above), a wall standing along the tile's inner edge, as tall as the back wall, its
    # faces plain wall, its top a lighter plain
    for side_name, tx in (("side-left", 0), ("side-right", cols - 1)):
        tys = [ty for ty in range(rows) if labels[ty][tx] == "wall_side"]
        if not tys:
            continue
        inner = tx == 0
        x0, _ = px(tx, 0)
        xi = x0 + PX - wall_thick if inner else x0
        xo = x0 + PX if inner else x0 + wall_thick
        y_top = wall_h
        ty0 = tys[0]
        while ty0 <= tys[-1]:
            ty1 = ty0
            while ty1 + 1 in tys:
                ty1 += 1
            _, z0 = px(tx, ty0)
            _, z1 = px(tx, ty1 + 1)
            face_uv = plain_uv(0, y_top)
            # (the face towards the room, plain wall; the ends and the top the picture where
            # the GBA draws them, so the GBA's view keeps its pixels, as the port's do)
            if inner:
                mesh.face([(xo, 0, z0), (xo, 0, z1), (xo, y_top, z1), (xo, y_top, z0)], face_uv, name=side_name)
            else:
                mesh.face([(xi, 0, z1), (xi, 0, z0), (xi, y_top, z0), (xi, y_top, z1)], face_uv, name=side_name)
            mesh.face([(xi, 0, z1), (xo, 0, z1), (xo, y_top, z1), (xi, y_top, z1)], name=side_name)
            mesh.face([(xo, 0, z0), (xi, 0, z0), (xi, y_top, z0), (xo, y_top, z0)], face_uv, name=side_name)
            mesh.face([(xi, y_top, z0), (xi, y_top, z1), (xo, y_top, z1), (xo, y_top, z0)], name=side_name)
            ty0 = ty1 + 1
    # the things
    for k, t in enumerate(things):
        tx, (a, b) = t["x"], t["rows"]
        kind = t["kind"]
        name = "thing-%d" % k
        x0, za = px(tx, a)
        zb = oy + (b + 1) * PX                        # (its foot)
        n = b - a + 1
        if kind == "on-wall":
            continue                                   # (painted on the wall's face already)
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
        # the back plain, the front's edge column
        mesh.face([(x0, 0, z_front), (x0 + PX, 0, z_front), (x0 + PX, h, z_front), (x0, h, z_front)], name=name)
        if z_front - z_back > 1e-6:
            mesh.face([(x0, h, z_back), (x0, h, z_front), (x0 + PX, h, z_front), (x0 + PX, h, z_back)], name=name)
        # (the sides and back wear a pixel just in from the front's edge: its body's colour,
        # not its outline's)
        edge_l = (x0 + 1.5, zb - h / 2.0)
        edge_r = (x0 + PX - 1.5, zb - h / 2.0)
        left_free = tx == 0 or not any(o["x"] == tx - 1 and o["rows"][1] == b and o["kind"] == kind for o in things)
        right_free = tx == cols - 1 or not any(o["x"] == tx + 1 and o["rows"][1] == b and o["kind"] == kind for o in things)
        if left_free:
            mesh.face([(x0, 0, z_back), (x0, 0, z_front), (x0, h, z_front), (x0, h, z_back)], edge_l, name=name)
        if right_free:
            mesh.face([(x0 + PX, 0, z_front), (x0 + PX, 0, z_back), (x0 + PX, h, z_back), (x0 + PX, h, z_front)], edge_r, name=name)
        if kind in ("free", "tall"):
            mesh.face([(x0 + PX, 0, z_back), (x0, 0, z_back), (x0, h, z_back), (x0 + PX, h, z_back)], ((x0 + PX / 2.0), zb - h / 2.0), name=name)
    return mesh


def cutouts(sprite, g, info, things):
    """The pixels to cut from the canvas (none yet: a thing at the wall's foot is one with the
    wall's face behind it)."""
    return []


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
            kinds[(t["x"], ty)] = {"on-wall": "h", "against-wall": "A", "free": "b", "tall": "B"}[t["kind"]]
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
