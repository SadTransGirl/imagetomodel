"""A building's mesh from its measures: the wall standing from its rows (its door and
windows set back), the roof's top panel lying over it as deep as the sprite draws it, a
slope down to each side's eave, the band along the roof's foot standing as its front edge
and following the slopes' front edges down at the corners; the sides and the back, which
the sprite never drew, a plain strip of the wall and the band's colour."""
from collections import Counter

from .geometry import Mesh
from .picture import hsv
from .measure import is_glass


def plain_column(sprite, x_range, y0, y1, wall_colours):
    """The wall's column with the fewest changes of colour (for its sides): mostly the wall's
    own colour (its commonest light one, not an outline's grey), no glass, no door."""
    counts = Counter()
    for y in range(y0, y1):
        counts += sprite.row_counts(y)
    main = next((c for c, _ in counts.most_common() if c in wall_colours and hsv(c)[2] >= 0.45), None)
    best, best_x = None, None
    for x in x_range:
        col = [sprite.colour(x, y) for y in range(y0, y1)]
        if any(c is None for c in col):
            continue
        common = Counter(col).most_common(1)[0][0]
        if common != main or hsv(common)[2] < 0.4:
            continue
        if sum(1 for c in col if is_glass(c)) >= 2:
            continue
        changes = sum(1 for a, b in zip(col, col[1:]) if a != b)
        if best is None or changes < best:
            best, best_x = changes, x
    if best_x is None:
        best_x = (x_range.start + x_range.stop) // 2
    return best_x + 0.5


def _holes_in(rects, ya, yb):
    """The rectangles spanning a slab of rows, as x intervals, sorted."""
    return sorted((r[0], r[2]) for r in rects if r[1] <= ya and r[3] >= yb)


def build(sprite, m, inset=1.0, door_inset=1.0, relief=True, depth=None):
    """The mesh. `inset`: how far the windows are set back (pixels), `door_inset` the door;
    `depth`: the building's depth in pixels (the roof's picture stretched over it), else
    as deep as the sprite draws its roof."""
    W, H = sprite.W, sprite.H
    mesh = Mesh()
    F = float(H)                                      # (the front: the sprite's bottom)
    wall_top = m["wall_top"]
    hw = F - wall_top                                 # (the wall's height at the middle)
    band = float(m["band"])
    Yt = hw + band                                    # (the top panel's height)
    xl, xr = (float(v) for v in m["panel"])
    tl, tr = (float(v) for v in m["wall"])
    zb = m["top_centre"] + Yt                         # (the roof's back edge)
    if depth is not None:
        zb = F - depth
    roof_v0, roof_v1 = m["top_centre"], wall_top - band
    wall_colours = {tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in m["colours"]["wall"]}

    def roof_uv(x, z):
        """The top panel's picture over a deeper building: its rows spread along it."""
        return (x, roof_v0 + (z - zb) / (F - zb) * (roof_v1 - roof_v0))
    stretched = depth is not None and abs((F - zb) - (roof_v1 - roof_v0)) > 1e-6

    # a point of the band (its middle row) for the faces that wear its colour
    band_uv = ((xl + xr) / 2.0, wall_top - band / 2.0 if band else wall_top - 0.5)

    # the sides' slopes: each a quad from the top panel's edge down to its eave
    sides = {}
    for key, inner, outer_default in (("left", xl, 0.0), ("right", xr, float(W))):
        s = m.get(key)
        if not s:
            sides[key] = None
            continue
        y_e = Yt - s["drop"]
        zb_s = s["top"] + y_e if depth is None else zb
        zf_s = s["eave"] + y_e if depth is None else F - (roof_v1 - s["eave"]) * (F - zb) / (roof_v1 - roof_v0) if stretched else s["eave"] + y_e
        sides[key] = {"x": float(s["edge"]), "y": y_e, "zb": zb_s, "zf": zf_s, "inner": inner}

    def side_uv(key, x, z):
        s = sides[key]
        if not stretched:
            return None
        v0, v1 = m[key]["top"], m[key]["eave"]
        return (x, v0 + (z - s["zb"]) / (s["zf"] - s["zb"]) * (v1 - v0))

    # the wall's top edge: the band's foot at the middle, down the slopes' fascia at the sides
    def wall_top_y(x):
        if sides["left"] and x < xl:
            s = sides["left"]
            t = (x - s["x"]) / (xl - s["x"]) if xl > s["x"] else 1.0
            return (s["y"] - band) + t * (hw - (s["y"] - band))
        if sides["right"] and x > xr:
            s = sides["right"]
            t = (s["x"] - x) / (s["x"] - xr) if s["x"] > xr else 1.0
            return (s["y"] - band) + t * (hw - (s["y"] - band))
        return hw

    # --- the front wall, with its door and windows cut out and set back ---
    holes = []
    if relief:
        for r in m.get("windows", []):
            holes.append((r, inset))
        if m.get("door"):
            holes.append((m["door"], door_inset))
    rects = [r for r, _ in holes]
    y_corner_l, y_corner_r = wall_top_y(tl), wall_top_y(tr)
    ys = {0.0, hw, y_corner_l, y_corner_r}
    for r in rects:
        ys |= {F - r[3], F - r[1]}
    ys = sorted(y for y in ys if 0.0 <= y <= hw)

    def left_at(y):
        # (the wall's left edge at a height: straight, then along the fascia to the panel)
        if y <= y_corner_l or not sides["left"]:
            return tl
        return tl + (xl - tl) * (y - y_corner_l) / (hw - y_corner_l) if hw > y_corner_l else xl

    def right_at(y):
        if y <= y_corner_r or not sides["right"]:
            return tr
        return tr - (tr - xr) * (y - y_corner_r) / (hw - y_corner_r) if hw > y_corner_r else xr

    for ya, yb in zip(ys, ys[1:]):
        if yb - ya < 1e-6:
            continue
        va, vb = F - ya, F - yb                       # (the slab's rows: va below vb)
        edges = [(left_at, None)]
        for x0, x1 in _holes_in(rects, vb, va):
            edges.append((None, float(x0)))
            edges.append((None, float(x1)))
        edges.append((right_at, None))
        for k in range(0, len(edges), 2):
            a, b = edges[k], edges[k + 1]
            ax_lo = a[0](ya) if a[0] else a[1]
            ax_hi = a[0](yb) if a[0] else a[1]
            bx_lo = b[0](ya) if b[0] else b[1]
            bx_hi = b[0](yb) if b[0] else b[1]
            if bx_lo - ax_lo < 1e-6 and bx_hi - ax_hi < 1e-6:
                continue
            pts = [(ax_lo, ya, F), (max(bx_lo, ax_lo), ya, F), (max(bx_hi, ax_hi), yb, F), (ax_hi, yb, F)]
            mesh.face(pts, name="wall")
    # the holes: each set back along the GBA's line of sight (so its picture stays put),
    # its reveals the frame's colour
    for (x0, v0, x1, v1), d in holes:
        yb_, yt_ = F - v1, F - v0                     # (its bottom and top heights)
        mesh.face([(x0, yb_ - d, F - d), (x1, yb_ - d, F - d), (x1, yt_ - d, F - d), (x0, yt_ - d, F - d)], name="relief")
        frame = sprite.colour(x0, (v0 + v1) // 2) or sprite.colour(x0 + 1, (v0 + v1) // 2)
        edge_uv = (x0 + 0.5, (v0 + v1) / 2.0) if frame else band_uv
        top_uv = ((x0 + x1) / 2.0, v0 + 0.5)
        mesh.face([(x0, yb_, F), (x0, yb_ - d, F - d), (x0, yt_ - d, F - d), (x0, yt_, F)], edge_uv, name="relief")
        mesh.face([(x1, yb_ - d, F - d), (x1, yb_, F), (x1, yt_, F), (x1, yt_ - d, F - d)], edge_uv, name="relief")
        mesh.face([(x0, yt_, F), (x0, yt_ - d, F - d), (x1, yt_ - d, F - d), (x1, yt_, F)], top_uv, name="relief")
        if yb_ > 0.5:
            mesh.face([(x0, yb_ - d, F - d), (x0, yb_, F), (x1, yb_, F), (x1, yb_ - d, F - d)], ((x0 + x1) / 2.0, v1 - 0.5), name="relief")

    # --- the roof ---
    # the top panel, and the band standing at its front edge
    mesh.face([(xl, Yt, zb), (xl, Yt, F), (xr, Yt, F), (xr, Yt, zb)],
              [roof_uv(xl, zb), roof_uv(xl, F), roof_uv(xr, F), roof_uv(xr, zb)] if stretched else None, name="roof")
    if band > 0:
        mesh.face([(xl, hw, F), (xr, hw, F), (xr, Yt, F), (xl, Yt, F)], name="roof")
    for key in ("left", "right"):
        s = sides[key]
        if not s:
            continue
        x_in, x_out = s["inner"], s["x"]
        # the slope, from the panel's edge out and down to the eave
        slope = [(x_in, Yt, zb), (x_in, Yt, F), (x_out, s["y"], s["zf"]), (x_out, s["y"], s["zb"])]
        if key == "right":
            slope = slope[::-1]
        uv = [side_uv(key, p[0], p[2]) for p in slope] if stretched else None
        if stretched:
            uv = [roof_uv(p[0], p[2]) if p[0] == x_in else side_uv(key, p[0], p[2]) for p in slope]
        mesh.face(slope, uv, name="roof")
        if band > 0:
            # the fascia under the slope's front edge, and the eave's outer edge and back
            fascia = [(x_out, s["y"] - band, s["zf"]), (x_in, hw, F), (x_in, Yt, F), (x_out, s["y"], s["zf"])]
            if key == "right":
                fascia = fascia[::-1]
            mesh.face(fascia, name="roof")
            outer = [(x_out, s["y"] - band, s["zb"]), (x_out, s["y"] - band, s["zf"]), (x_out, s["y"], s["zf"]), (x_out, s["y"], s["zb"])]
            if key == "right":
                outer = outer[::-1]
            mesh.face(outer, band_uv, name="roof")
    # the back: the band's colour, from the eaves' back corners up to the panel's back edge
    left_pts = [(sides["left"]["x"], sides["left"]["y"] - band, sides["left"]["zb"]), (sides["left"]["x"], sides["left"]["y"], sides["left"]["zb"])] \
        if sides["left"] else [(xl, hw, zb)]
    right_pts = [(sides["right"]["x"], sides["right"]["y"] - band, sides["right"]["zb"]), (sides["right"]["x"], sides["right"]["y"], sides["right"]["zb"])] \
        if sides["right"] else [(xr, hw, zb)]
    back = [right_pts[0], left_pts[0]] + left_pts[1:] + [(xl, Yt, zb), (xr, Yt, zb)] + right_pts[1:][::-1]
    if band > 0 or sides["left"] or sides["right"]:
        for j in range(1, len(back) - 1):
            mesh.face([back[0], back[j], back[j + 1]], band_uv, name="roof")

    # --- the sides and the back of the wall: a plain strip of it ---
    y_side_l, y_side_r = y_corner_l, y_corner_r
    left_col = plain_column(sprite, range(int(tl) + 1, max(int(tl) + 2, int(W * 0.3))), wall_top, H, wall_colours)
    right_col = plain_column(sprite, range(int(W * 0.7), max(int(W * 0.7) + 1, int(tr) - 1)), wall_top, H, wall_colours)
    wall_zb = zb
    mesh.face([(tl, 0, wall_zb), (tl, 0, F), (tl, y_side_l, F), (tl, y_side_l, wall_zb)],
              [(left_col, F), (left_col, F), (left_col, F - y_side_l), (left_col, F - y_side_l)], name="wall")
    mesh.face([(tr, 0, F), (tr, 0, wall_zb), (tr, y_side_r, wall_zb), (tr, y_side_r, F)],
              [(right_col, F), (right_col, F), (right_col, F - y_side_r), (right_col, F - y_side_r)], name="wall")
    y_back = min(y_side_l, y_side_r)
    mesh.face([(tr, 0, wall_zb), (tl, 0, wall_zb), (tl, y_back, wall_zb), (tr, y_back, wall_zb)],
              [(left_col, F), (left_col, F), (left_col, F - y_back), (left_col, F - y_back)], name="wall")
    # a lid under the roof, between the wall's top and the panel, the band's colour
    # (a little behind the front, where the door and windows are set back into)
    y_lid = min(y_corner_l, y_corner_r)
    z_lid = F - max(1.0, inset, door_inset) - 0.5
    mesh.face([(tl, y_lid, wall_zb), (tr, y_lid, wall_zb), (tr, y_lid, z_lid), (tl, y_lid, z_lid)], band_uv, name="wall")
    return mesh
