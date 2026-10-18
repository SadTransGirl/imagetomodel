"""What a building is, read from its sprite: where its wall stands, the band along the
roof's foot, the roof's top panel and the slopes at its sides, its door and windows. Each
measure can be set by hand in a spec (a JSON object with any of the keys `measure`
returns), which wins over what was read."""
from collections import Counter

import numpy as np

from .picture import hsv

GLASS_HUE = (185.0, 245.0)


def is_glass(c):
    h, s, v = hsv(c)
    return GLASS_HUE[0] <= h <= GLASS_HUE[1] and s > 0.3 and v > 0.55


def is_dark(c):
    return hsv(c)[2] < 0.5


def _median(values, default):
    values = sorted(values)
    return values[len(values) // 2] if values else default


def _components(mask):
    """The 4-connected regions of a boolean array: a list of (x0, y0, x1, y1) boxes."""
    H, W = mask.shape
    seen = np.zeros_like(mask)
    boxes = []
    for y in range(H):
        for x in range(W):
            if not mask[y, x] or seen[y, x]:
                continue
            todo = [(x, y)]
            seen[y, x] = True
            x0 = x1 = x
            y0 = y1 = y
            while todo:
                cx, cy = todo.pop()
                x0, x1, y0, y1 = min(x0, cx), max(x1, cx), min(y0, cy), max(y1, cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < W and 0 <= ny < H and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        todo.append((nx, ny))
            boxes.append([x0, y0, x1 + 1, y1 + 1])
    return boxes


def _merge(boxes, gap):
    """Boxes within `gap` pixels of each other joined (a window's two panes)."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def measure(sprite, spec=None):
    """The building's measures (sprite rows and columns, in pixels):
      top            the first row drawn across most of the sprite
      wall_top       the row the standing wall starts at (its height is H - wall_top)
      band           rows of the roof's front edge standing above the wall (its dark band)
      panel          [xl, xr]: the columns of the roof's top panel
      top_centre     the row of the top panel's back edge
      left, right    each side's slope: {"edge": its outer column, "top": the row of its
                     back corner, "eave": the row of its front corner (the band's top),
                     "drop": how far below the top panel the eave is (pixels)}, or None
      wall           [left, right]: the columns the wall spans
      door           [x0, y0, x1, y1] or None
      windows        [[x0, y0, x1, y1], ...]
      colours        wall, panel: the colours read as the wall's and the top panel's
    """
    spec = dict(spec or {})
    W, H = sprite.W, sprite.H
    m = {"W": W, "H": H}

    def drawn_count(y):
        return int(sprite.drawn[y].sum())

    top = 0
    while top < H - 1 and drawn_count(top) * 2 < W:
        top += 1
    m["top"] = int(spec.get("top", top))

    # the wall's colours: the light ones of the bottom third found right across it (in
    # three of its four quarters: a door is in one place, the wall everywhere), glass aside
    counts = Counter()
    quarters = {}
    for y in range(H * 2 // 3, H):
        counts += sprite.row_counts(y)
        for q in range(4):
            for c in sprite.row_counts(y, W * q // 4, W * (q + 1) // 4):
                quarters.setdefault(c, set()).add(q)
    total = sum(counts.values()) or 1
    wall_colours = {c for c, n in counts.items()
                    if n / total >= 0.02 and len(quarters.get(c, ())) >= 3 and hsv(c)[2] >= 0.45 and not is_glass(c)}
    xc0, xc1 = int(W * 0.35), int(W * 0.65) + 1

    def dominant(y):
        return sprite.dominant(y, xc0, xc1)

    # the standing wall: up from the bottom, a row is the wall's while any of a few columns
    # across it (between its windows and door) is a wall colour; dark rows (outlines,
    # frames) are passed over; it ends at a row of the roof's colours right across
    probes = [int(W * f) for f in (0.15, 0.3, 0.5, 0.7, 0.85)]

    def probe(y):
        cs = [sprite.colour(x, y) for x in probes]
        if any(c in wall_colours for c in cs):
            return "wall"
        if all(c is None or is_dark(c) or is_glass(c) for c in cs):
            return "dark"
        return "other"
    wall_top = H - 1
    y = H - 1
    while y > m["top"]:
        kind = probe(y)
        if kind == "wall":
            wall_top = y
        elif kind == "other":
            break
        y -= 1
    # (better: the wall ends under the lowest dark line drawn right across it with the
    # roof's colours above it; a trim of other colours at the wall's top is the wall's)
    def dark_line(y):
        cs = [sprite.colour(x, y) for x in range(W)]
        return sum(1 for c in cs if c is not None and is_dark(c)) * 10 >= W * 6

    def roof_like(y):
        cs = [sprite.colour(x, y) for x in probes]
        return all(c is None or (c not in wall_colours and not is_glass(c) and (is_dark(c) or hsv(c)[1] >= 0.3)) for c in cs) \
            and any(c is not None for c in cs)
    y = H - 1
    while y > m["top"] and dark_line(y):
        y -= 1                                        # (the outline along the foot)
    while y > m["top"]:
        if dark_line(y):
            above = y - 1
            while above > m["top"] and dark_line(above):
                above -= 1
            if above <= m["top"] or roof_like(above):
                wall_top = y + 1
                break
        y -= 1
    m["wall_top"] = int(spec.get("wall_top", wall_top))
    wall_top = m["wall_top"]

    # the top panel's colours: the common ones of the roof's middle, and its main one
    panel_counts = Counter()
    for y in range(m["top"], max(m["top"] + 1, wall_top - 8)):
        panel_counts += sprite.row_counts(y, xc0, xc1)
    ptotal = sum(panel_counts.values()) or 1
    panel_colours = {c for c, n in panel_counts.items() if n / ptotal >= 0.05}
    panel_main = panel_counts.most_common(1)[0][0] if panel_counts else None

    # the band: the rows above the wall that aren't the panel's, up to 8
    band = 0
    y = wall_top - 1
    while y > m["top"] and band < 8:
        c = dominant(y)
        if c in panel_colours:
            break
        band += 1
        y -= 1
    m["band"] = int(spec.get("band", band))
    band = m["band"]
    eave_centre = wall_top - band

    # the top panel's columns: the run of its main colour through the middle, row by row
    lefts, rights = [], []
    for y in range(m["top"], eave_centre):
        xm = W // 2
        if sprite.colour(xm, y) != panel_main:
            continue
        x0 = xm
        while x0 > 0 and sprite.colour(x0 - 1, y) == panel_main:
            x0 -= 1
        x1 = xm + 1
        while x1 < W and sprite.colour(x1, y) == panel_main:
            x1 += 1
        lefts.append(x0)
        rights.append(x1)
    xl, xr = _median(lefts, 0), _median(rights, W)
    m["panel"] = [int(v) for v in spec.get("panel", [xl, xr])]
    xl, xr = m["panel"]
    top_centre = m["top"]
    for y in range(m["top"], eave_centre):
        if int(sprite.drawn[y, xl:xr].sum()) * 2 >= (xr - xl):
            top_centre = y
            break
    m["top_centre"] = int(spec.get("top_centre", top_centre))
    top_centre = m["top_centre"]

    # each side's slope: the roof past the panel, its outer column, its corners' rows
    def side(outer, inner, step):
        cols = range(outer, inner, step)
        edge = None
        for x in cols:
            span = sprite.column_drawn(x)
            if span and span[0] < eave_centre:
                edge = x
                break
        if edge is None or abs(inner - edge) < 2:
            return None
        span = sprite.column_drawn(edge)
        top_side = span[0]
        eave = min(top_side + (eave_centre - top_centre), span[1] - band)
        return {"edge": int(edge), "top": int(top_side), "eave": int(eave), "drop": int(top_side - top_centre)}
    m["left"] = spec.get("left", side(0, xl, 1))
    m["right"] = spec.get("right", side(W - 1, xr - 1, -1))
    if m["right"] is not None and "edge" in m["right"]:
        m["right"]["edge"] = int(m["right"]["edge"]) + (0 if "right" in spec else 1)   # (the column past the edge)

    # the wall's columns: those drawn down most of its rows
    def wall_column(x):
        return int(sprite.drawn[wall_top:H, x].sum()) * 4 >= (H - wall_top) * 3
    tl = 0
    while tl < W // 2 and not wall_column(tl):
        tl += 1
    tr = W
    while tr > W // 2 and not wall_column(tr - 1):
        tr -= 1
    m["wall"] = [int(v) for v in spec.get("wall", [tl, tr])]
    tl, tr = m["wall"]

    # the door: a region of other colours at the wall's foot; the windows: the glass
    wall_rgb = sprite.rgba[:, :, :3]
    other = np.zeros((H, W), dtype=bool)
    glass = np.zeros((H, W), dtype=bool)
    for y in range(wall_top, H):
        for x in range(tl, tr):
            c = sprite.colour(x, y)
            if c is None:
                continue
            if is_glass(c):
                glass[y, x] = True
            elif c not in wall_colours and not is_dark(c):
                other[y, x] = True
    door = None
    foot = H - 1
    while foot > wall_top and sprite.row_counts(foot, tl, tr) and is_dark(sprite.dominant(foot, tl, tr)):
        foot -= 1
    for box in sorted(_components(other), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        w, h = box[2] - box[0], box[3] - box[1]
        if w >= 5 and h >= 6 and box[3] >= foot - 3:
            door = [max(tl, box[0] - 1), max(wall_top, box[1] - 1), min(tr, box[2] + 1), min(H, box[3] + 1)]
            # (up to its outline: dark rows above it, and a plain row under a dark one)
            def dark_row(y):
                cs = [sprite.colour(x, y) for x in range(box[0], box[2])]
                return all(c is not None and is_dark(c) for c in cs)
            def plain_row(y):
                cs = {sprite.colour(x, y) for x in range(box[0], box[2])}
                return len(cs) == 1 and None not in cs
            y = door[1]
            for _ in range(4):
                if y - 1 >= wall_top and dark_row(y - 1):
                    y -= 1
                elif y - 2 >= wall_top and plain_row(y - 1) and dark_row(y - 2):
                    y -= 2
                else:
                    break
            door[1] = y
            break
    m["door"] = spec.get("door", door)
    windows = []
    for box in _merge(_components(glass), 3):
        w, h = box[2] - box[0], box[3] - box[1]
        if w < 3 or h < 3:
            continue
        grown = [max(tl, box[0] - 1), max(wall_top, box[1] - 1), min(tr, box[2] + 1), min(H, box[3] + 1)]
        d = m["door"]
        if d and grown[0] >= d[0] and grown[1] >= d[1] and grown[2] <= d[2] and grown[3] <= d[3]:
            continue                                  # (the door's own window)
        windows.append(grown)
    m["windows"] = spec.get("windows", sorted(windows, key=lambda b: (b[1], b[0])))
    m["colours"] = {"wall": ["%02x%02x%02x" % c for c in sorted(wall_colours, key=lambda c: -counts[c])],
                    "panel": ["%02x%02x%02x" % c for c in sorted(panel_colours, key=lambda c: -panel_counts[c])]}
    return m
