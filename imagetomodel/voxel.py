"""True voxels: one cube a GBA pixel, as made by hand in MagicaVoxel. A sprite is
unprojected, not extruded: the GBA draws a cube's front face on one row and its top face on
the row above, so the bottom rows of each column of a thing are its front, standing, and
the rows above them its top, lying back. Rounded corners, rims and outlines come out as the
art draws them, and the model seen from the GBA's angle gives the picture back but for the
row along each front's top edge (a cube has one colour; its top and front show two rows).

Coordinates are the room's pixels: x across (east), y up from the floor (the floor's cubes
at y = -1), z down the picture (south, towards the camera). A cube (x, y, z) shows its front
face on row z - y and its top face on row z - y - 1."""
import math
import struct
from collections import Counter

from .geometry import Mesh, normal
from .picture import PX, hsv, luma


def _dark(c):
    return hsv(c)[2] < 0.45


class Grid:
    """Cubes: {(x, y, z): (r, g, b)}, and which object each belongs to."""

    def __init__(self):
        self.cubes = {}
        self.owner = {}

    def set(self, x, y, z, colour, owner=""):
        self.cubes[(x, y, z)] = tuple(int(v) for v in colour[:3])
        self.owner[(x, y, z)] = owner

    def __len__(self):
        return len(self.cubes)

    def bounds(self):
        xs = [k[0] for k in self.cubes]
        ys = [k[1] for k in self.cubes]
        zs = [k[2] for k in self.cubes]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def only(self, owner):
        out = Grid()
        for k, c in self.cubes.items():
            if self.owner[k] == owner:
                out.cubes[k] = c
                out.owner[k] = owner
        return out

    def moved(self, dx, dy, dz):
        out = Grid()
        for (x, y, z), c in self.cubes.items():
            out.cubes[(x + dx, y + dy, z + dz)] = c
            out.owner[(x + dx, y + dy, z + dz)] = self.owner[(x, y, z)]
        return out


# --- unprojecting a thing ---

def _columns(own):
    cols = {}
    for (x, v) in own:
        cols.setdefault(x, []).append(v)
    return {x: sorted(vs) for x, vs in cols.items()}


def _row_colours(sprite, own, rows_of):
    """For each height y: the body colour seen from the left and from the right (the first
    colour inside the outline), and the commonest body colour across. `rows_of(x, y)` is
    the picture's row of column x at height y."""
    out = {}
    xs = sorted({x for x, _ in own})
    ys = set()
    for x in xs:
        for y in range(0, 256):
            v = rows_of(x, y)
            if v is None:
                break
            ys.add(y)
    for y in ys:
        line = []
        for x in xs:
            v = rows_of(x, y)
            if v is not None and (x, v) in own:
                line.append(sprite.colour(x, v))
        line = [c for c in line if c is not None]
        if not line:
            continue

        def body(seq):
            for c in seq[1:]:
                if not _dark(c):
                    return c
            return seq[0]
        mode = Counter(c for c in line if not _dark(c)).most_common(1)
        out[y] = {"left": body(line), "right": body(line[::-1]), "mid": mode[0][0] if mode else line[0]}
    return out


def _darkest(sprite, own):
    best = None
    for p in own:
        c = sprite.colour(*p)
        if c is not None and (best is None or luma(c) < luma(best)):
            best = c
    return best or (40, 40, 60)


def unproject_box(grid, sprite, own, height_of, back_of=None, owner="", y_base=0):
    """A thing's cubes from its own pixels. For each column x (its rows top(x) to the foot
    zf(x)): `height_of(x, top, zf)` its front's height h; its front the bottom h rows,
    standing at z = zf - 1; its top the rows above, lying at y = h - 1 (row v at z = v + h);
    `back_of(x)` a z its top reaches back to past what's drawn (a thing against the wall).
    Solid between. The cubes the GBA sees wear the picture; the sides the row's body
    colour, the back the row's commonest, the outline along the back edge."""
    cols = _columns(own)
    outline = _darkest(sprite, own)
    plan = {}
    for x, vs in cols.items():
        top, zf = vs[0], vs[-1] + 1
        h = max(1, min(int(height_of(x, top, zf)), zf - top))
        z0 = min(top + h, zf - 1)
        if back_of is not None:
            z0 = min(z0, back_of(x))
        plan[x] = (top, zf, h, z0)

    def front_row(x, y):
        if x not in plan:
            return None
        top, zf, h, z0 = plan[x]
        return zf - 1 - y if 0 <= y < h else None
    rows = _row_colours(sprite, own, front_row)
    xs = sorted(plan)
    for x in xs:
        top, zf, h, z0 = plan[x]
        for z in range(z0, zf):
            for y in range(h):
                if z == zf - 1:
                    v = zf - 1 - y                     # (its front)
                    c = sprite.colour(x, v) if (x, v) in own else None
                elif y == h - 1:
                    v = z - h                          # (its top as drawn)
                    if v < top:
                        # (reaching back past what's drawn, to the wall: the lid's body, its
                        # outline along the back edge)
                        body = top + 1 if (x, top + 1) in own and top + 1 < zf - h else top
                        v = top if z == z0 else body
                    c = sprite.colour(x, v) if (x, v) in own else None
                else:
                    c = None
                if c is None:
                    r = rows.get(y)
                    left = x == xs[0] or (x - 1) not in plan or not (plan[x - 1][3] <= z < plan[x - 1][1]) or y >= plan[x - 1][2]
                    right = x == xs[-1] or (x + 1) not in plan or not (plan[x + 1][3] <= z < plan[x + 1][1]) or y >= plan[x + 1][2]
                    if z == z0 and (left or right or y == 0):
                        c = outline                    # (the back edge's outline)
                    elif r is None:
                        c = outline
                    elif left:
                        c = r["left"]
                    elif right:
                        c = r["right"]
                    else:
                        c = r["mid"]
                grid.set(x, y + y_base, z, c, owner)
    return plan


def unproject_slab(grid, sprite, own, y_base, owner="", thick=2):
    """A thing standing on its own (a plant, a flower): each pixel a cube, its bottom row
    at height `y_base`, standing at z = foot + y_base; thickened backwards only as far as
    the cubes in front of it hide the extra cubes' tops from the GBA's angle."""
    if not own:
        return
    foot = max(v for _, v in own)
    z = foot + y_base
    for (x, v) in own:
        c = sprite.colour(x, v)
        if c is None:
            continue
        y = z - v
        grid.set(x, y, z, c, owner)
        # (a cube k behind shows its top on row v - k - 1: covered while the column goes on
        # up that far)
        k = 1
        while k <= thick and (x, v - k - 1) in own and all((x, v - j) in own for j in range(1, k + 2)):
            grid.set(x, y, z - k, c, owner)
            k += 1


# --- meshing ---

def mesh_of(grid, mesh=None, strip_v=0, strip_w=None, name=None):
    """The grid's visible faces as quads, merged greedily where neighbouring faces have the
    same colour (and the same owner), each wearing its colour from the mesh's palette.
    Polygons are named by their owner (or `name`)."""
    if mesh is None:
        mesh = Mesh(strip_v=strip_v, strip_w=strip_w)
    cubes = grid.cubes
    dirs = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
    for axis, sign in dirs:
        ua, va = [a for a in range(3) if a != axis]
        slices = {}
        for k, c in cubes.items():
            n = list(k)
            n[axis] += sign
            if tuple(n) in cubes:
                continue
            key = (c, grid.owner.get(k, ""))
            slices.setdefault(k[axis], {})[(k[ua], k[va])] = key
        for plane, cells in slices.items():
            seen = set()
            for (u, v) in sorted(cells):
                if (u, v) in seen:
                    continue
                key = cells[(u, v)]
                u1 = u + 1
                while (u1, v) in cells and (u1, v) not in seen and cells[(u1, v)] == key:
                    u1 += 1
                v1 = v + 1
                while all((uu, v1) in cells and (uu, v1) not in seen and cells[(uu, v1)] == key for uu in range(u, u1)):
                    v1 += 1
                for uu in range(u, u1):
                    for vv in range(v, v1):
                        seen.add((uu, vv))
                p = plane + (1 if sign > 0 else 0)

                def at(a, b):
                    pt = [0, 0, 0]
                    pt[axis] = p
                    pt[ua] = a
                    pt[va] = b
                    return tuple(pt)
                quad = [at(u, v), at(u1, v), at(u1, v1), at(u, v1)]
                nrm = normal(quad)
                if nrm[axis] * sign < 0:
                    quad = quad[::-1]
                colour, owner = key
                mesh.face(quad, mesh.plain(colour), name=name or owner or "voxels")
    return mesh


# --- MagicaVoxel ---

def write_vox(path, grid):
    """A MagicaVoxel .vox (version 150): x across, the depth with the front at the low end,
    z up; the palette the grid's colours (the nearest kept past 255). A grid larger than
    256 cubes a side doesn't fit and raises."""
    (x0, y0, z0), (x1, y1, z1) = grid.bounds()
    sx, sy, sz = x1 - x0 + 1, z1 - z0 + 1, y1 - y0 + 1
    if max(sx, sy, sz) > 256:
        raise ValueError("%dx%dx%d cubes: MagicaVoxel holds 256 a side" % (sx, sy, sz))
    counts = Counter(grid.cubes.values())
    palette = [c for c, _ in counts.most_common(255)]
    index = {c: i + 1 for i, c in enumerate(palette)}

    def nearest(c):
        if c in index:
            return index[c]
        best = min(palette, key=lambda p: sum((p[k] - c[k]) ** 2 for k in range(3)))
        index[c] = index[best]
        return index[best]
    data = bytearray()
    for (x, y, z), c in grid.cubes.items():
        data += bytes((x - x0, z1 - z, y - y0, nearest(c)))
    n = len(grid.cubes)

    def chunk(tag, content, children=b""):
        return tag + struct.pack("<ii", len(content), len(children)) + content + children
    size = chunk(b"SIZE", struct.pack("<iii", sx, sy, sz))
    xyzi = chunk(b"XYZI", struct.pack("<i", n) + bytes(data))
    rgba = bytearray()
    for i in range(256):
        c = palette[i] if i < len(palette) else (0, 0, 0)
        rgba += bytes((c[0], c[1], c[2], 255 if i < len(palette) else 0))
    main = chunk(b"MAIN", b"", size + xyzi + chunk(b"RGBA", bytes(rgba)))
    with open(path, "wb") as f:
        f.write(b"VOX " + struct.pack("<i", 150) + main)


def read_vox(path):
    """A .vox written by write_vox (or MagicaVoxel, one model): (size, [(x, y, z, i)], palette)."""
    raw = open(path, "rb").read()
    assert raw[:4] == b"VOX "
    off = 8
    size, cubes, palette = None, [], None

    def walk(off, end):
        nonlocal size, cubes, palette
        while off < end:
            tag = raw[off:off + 4]
            n, m = struct.unpack_from("<ii", raw, off + 4)
            body = off + 12
            if tag == b"SIZE":
                size = struct.unpack_from("<iii", raw, body)
            elif tag == b"XYZI":
                count, = struct.unpack_from("<i", raw, body)
                cubes = [tuple(raw[body + 4 + 4 * k:body + 8 + 4 * k]) for k in range(count)]
            elif tag == b"RGBA":
                palette = [tuple(raw[body + 4 * k:body + 4 * k + 4]) for k in range(256)]
            walk(body + n, body + n + m)
            off = body + n + m
    walk(off, len(raw))
    return size, cubes, palette


# --- a profile: each row of a thing standing or lying ---

def unproject_profile(grid, sprite, own, kind_of, back_of=None, owner="", recess=None, raised=None, colours=None):
    """A thing's cubes, row by row up each column from its foot: `kind_of(x, v)` says
    whether row v stands ("up": a cube on top of the last, its front face the pixel) or lies
    ("back": a cube behind the last, its top face the pixel). A box is its front rows up and
    its top rows back; a machine can be a lip up, a tray back, a panel up. The path keeps
    each pixel where the GBA draws it (a cube's front shows on row z - y, its top on the row
    above), but for the cube at each turn from up to back, which shows two rows.
    `back_of(x)`: a z its top reaches back to past what's drawn (to the wall).
    `recess`: {(x, v): k} pixels set k cubes back into it (a shelf's inside).
    `raised`: pixels standing a cube proud (a ball on a tray, a key on a keyboard).
    `colours`: {(x, v): colour} for pixels that wear another than the picture's (a hole
    closed with the thing's own colour).
    Returns {(x, v): (y, z)}, the cube each pixel is shown on."""
    recess = recess or {}
    raised = raised or set()
    cols = _columns(own)
    outline = _darkest(sprite, own)
    shown = {}
    placed = {}                                        # (x, y, z) -> colour, before filling
    heights = {}                                       # (x, z) -> height of the solid there
    hollow = set()
    by_height = {}                                     # y -> Counter of standing colours
    upright = {}                                       # (x, y) -> a standing pixel's colour
    proud = set()                                      # (the raised cubes: the picture's too)
    for x, vs in cols.items():
        top, zf = vs[0], vs[-1] + 1
        y, z = 0, zf
        last = None
        for v in range(zf - 1, top - 1, -1):
            c = ((colours or {}).get((x, v)) or sprite.colour(x, v)) if (x, v) in own else None
            kind = kind_of(x, v)
            if kind == "up":
                k = recess.get((x, v), 0)
                cell = (y, z - 1 - k)
                for j in range(k):
                    hollow.add((x, y, z - 1 - j))
                if c is not None:
                    placed[(x,) + cell] = c
                    by_height.setdefault(y, Counter())[c] += 0 if _dark(c) else 1
                    upright[(x, y)] = c
                    if (x, v) in raised:
                        placed[(x, y, z - k)] = c
                        proud.add((x, y, z - k))
                shown[(x, v)] = cell
                heights[(x, cell[1])] = max(heights.get((x, cell[1]), 0), y + 1)
                y += 1
                last = "up"
            else:
                if y == 0:
                    z -= 1                             # (lying on the floor: the floor's)
                    continue
                # (set k cubes down into it along the line of sight, a tray sunk in its
                # rim: its picture where the GBA draws it, its floor run on forward under
                # the gap to the rim)
                k = min(recess.get((x, v), 0), y - 1)
                cell = (y - 1 - k, z - 1 - k)
                if c is not None and not (last == "up" and (x,) + cell in placed):
                    placed[(x,) + cell] = c
                    for j in range(k):
                        placed.setdefault((x, y - 1 - k, z - 1 - j), c)
                if c is not None and (x, v) in raised:
                    placed[(x, y - k, z - 1 - k)] = c
                    proud.add((x, y - k, z - 1 - k))
                    heights[(x, z - 1 - k)] = max(heights.get((x, z - 1 - k), 0), y + 1 - k)
                shown[(x, v)] = cell
                for j in range(k + 1):
                    heights[(x, z - 1 - j)] = max(heights.get((x, z - 1 - j), 0), y - k)
                z -= 1
                last = "back"
        # (reaching back to the wall past what's drawn: the top's body, outlined at the back)
        if back_of is not None and y > 0:
            body = sprite.colour(x, top + 1) if (x, top + 1) in own else sprite.colour(x, top)
            while z > back_of(x):
                placed[(x, y - 1, z - 1)] = outline if z - 1 == back_of(x) else (body or outline)
                heights[(x, z - 1)] = max(heights.get((x, z - 1), 0), y)
                z -= 1
    # (nothing past the wall it reaches back to: what's drawn past it is behind it)
    if back_of is not None:
        placed = {k: c for k, c in placed.items() if k[2] >= back_of(k[0])}
        heights = {k: h for k, h in heights.items() if k[1] >= back_of(k[0])}
        shown = {p: cell for p, cell in shown.items() if cell[1] >= back_of(p[0])}
    # the solid under the surface: each row's body colour, its back the thing's body colour
    # (outlined round its edge after, by outline())
    mids = {yy: (cnt.most_common(1)[0][0] if cnt and cnt.most_common(1)[0][1] > 0 else outline) for yy, cnt in by_height.items()}
    every = sum(by_height.values(), Counter())
    body = every.most_common(1)[0][0] if every else outline
    zmin = {}
    for (x, zz) in heights:
        zmin[x] = min(zmin.get(x, zz), zz)
    filled = set()
    for (x, zz), h in heights.items():
        for yy in range(h):
            if (x, yy, zz) in placed or (x, yy, zz) in hollow:
                continue
            placed[(x, yy, zz)] = body if zz == zmin[x] else mids.get(yy, outline)
            filled.add((x, yy, zz))
    # (its sides at each height the colour its front's edge is there, the first inside its
    # outline from that side: a panel plain where the front's middle is a screen, a band
    # running on round the corner where the front is banded)
    edge = {}
    for (x, yy), c in upright.items():
        edge.setdefault(yy, []).append((x, c))
    side = {}
    for yy, row in edge.items():
        row.sort()
        light = [c for _, c in row if not _dark(c)]
        if light:
            side[yy] = (light[0], light[-1])
    ends = {}
    for (x, yy, zz) in placed:
        lo, hi = ends.get((yy, zz), (x, x))
        ends[(yy, zz)] = (min(lo, x), max(hi, x))
    for (x, yy, zz) in filled:
        if yy in side and zz != zmin[x]:
            lo, hi = ends[(yy, zz)]
            if x == lo:
                placed[(x, yy, zz)] = side[yy][0]
            elif x == hi:
                placed[(x, yy, zz)] = side[yy][1]
    # (the solid's cubes the GBA's view shows after all, a sunk tray's back wall standing
    # where its last rows are drawn: the picture's colours there)
    cells = {(x,) + cell for (x, _), cell in shown.items()} | proud
    for (x, r), key in gba_view(placed).items():
        if key not in cells and (x, r) in own:
            c = ((colours or {}).get((x, r)) or sprite.colour(x, r))
            if c is not None:
                placed[key] = c
    for (x, yy, zz), c in placed.items():
        grid.set(x, yy, zz, c, owner)
    return shown


def gba_view(cubes):
    """{pixel: cube} for the GBA's view: the cube whose face (a front on row z - y, a top
    on the row above) is nearest the camera at each pixel."""
    best = {}
    for (x, y, z) in cubes:
        faces = []
        if (x, y, z + 1) not in cubes:
            faces.append((x, z - y))
        if (x, y + 1, z) not in cubes:
            faces.append((x, z - y - 1))
        for p in faces:
            if p not in best or y + z > best[p][0]:
                best[p] = (y + z, (x, y, z))
    return {p: key for p, (_, key) in best.items()}


def gba_shown(cubes):
    """The cubes the GBA's view shows a face of."""
    return set(gba_view(cubes).values())


def paint_sides(grid, owner, colour, panel=None):
    """A thing's left and right faces, as the GBA's view doesn't show them, one colour, and
    `panel` (given) inset a cube from each face's edge: a cabinet's side panel."""
    cubes = {k for k, o in grid.owner.items() if o == owner}
    keep = gba_shown(cubes)
    ends = {}
    for (x, y, z) in cubes:
        lo, hi = ends.get((y, z), (x, x))
        ends[(y, z)] = (min(lo, x), max(hi, x))
    for pick in (0, 1):
        face = {(y, z): ends[(y, z)][pick] for (y, z) in ends}
        for (y, z), x in face.items():
            if (x, y, z) in keep:
                continue
            inner = all(face.get((y + dy, z + dz)) == x for dy, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            grid.set(x, y, z, panel if panel is not None and inner else colour, owner)


def outline(grid, owner, colour, keep_shown=True, where=None):
    """A thing outlined as the GBA outlines a sprite, from the sides and the back too: the
    cubes on the edge of its silhouette seen from the left, the right, the back and the
    front (nearest that side each) wear its outline colour, but for the ones the GBA's view
    shows, which keep the picture's (`keep_shown`; a round thing, turned from its picture
    without its outline, is outlined all round). `where(key, (dx, dy))`: whether the
    edge out that way (dx: left or right, dy: its top or foot) is drawn outlined there
    (on a round thing, where the picture outlines it). Returns how many it coloured."""
    cubes = {k for k, o in grid.owner.items() if o == owner}
    if not cubes:
        return 0
    keep = gba_shown(cubes) if keep_shown else set()
    edge = set()
    # (each side: the plane it's seen on, the axis it looks along, which end is nearest,
    # and which way out of its silhouette counts: from the left and the right only its top
    # and its foot, its upright edges being the front's and the back's, which on a round
    # thing are its sides, not its middle)
    every = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for plane, along, near, ways in (((2, 1), 0, min, every[2:]), ((2, 1), 0, max, every[2:]),
                                     ((0, 1), 2, min, every), ((0, 1), 2, max, every)):
        nearest = {}
        for key in cubes:
            cell = (key[plane[0]], key[plane[1]])
            if cell not in nearest or near(key[along], nearest[cell][along]) == key[along]:
                nearest[cell] = key
        # (its silhouette's outer edge: next to the outside, reached round it, not a gap
        # inside it)
        a_s = [a for a, _ in nearest]
        b_s = [b for _, b in nearest]
        lo_a, hi_a, lo_b, hi_b = min(a_s) - 1, max(a_s) + 1, min(b_s) - 1, max(b_s) + 1
        outside = set()
        todo = [(lo_a, lo_b)]
        while todo:
            q = todo.pop()
            if q in outside or q in nearest or not (lo_a <= q[0] <= hi_a and lo_b <= q[1] <= hi_b):
                continue
            outside.add(q)
            todo += [(q[0] + 1, q[1]), (q[0] - 1, q[1]), (q[0], q[1] + 1), (q[0], q[1] - 1)]
        for (a, b), key in nearest.items():
            out = [(da, db) for da, db in ways if (a + da, b + db) in outside]
            if out and (where is None or any(where(key, (da if plane[0] == 0 else 0, db)) for da, db in out)):
                edge.add(key)
    n = 0
    for key in edge - keep:
        grid.set(*key, colour, owner)
        n += 1
    return n


def box_kinds(height_of, own):
    """A box's profile: in each column its bottom `height_of(x, top, zf)` rows up, the
    rest back."""
    cols = _columns(own)
    plan = {x: (vs[0], vs[-1] + 1) for x, vs in cols.items()}

    def kind_of(x, v):
        top, zf = plan[x]
        return "up" if v >= zf - max(1, int(height_of(x, top, zf))) else "back"
    return kind_of


def columns_kinds(entries, default, x0, bottom):
    """A profile per range of columns: [{"cols": [a, b], "profile": segments}, ...] from a
    thing's box's left edge `x0`; the others `default`."""
    parts = [((e["cols"][0] + x0, e["cols"][1] + x0), segments_kinds(e["profile"], bottom)) for e in entries]

    def kind_of(x, v):
        for (a, b), k in parts:
            if a <= x <= b:
                return k(x, v)
        return default(x, v)
    return kind_of


def segments_kinds(segments, bottom):
    """A profile from segments up from a thing's bottom row: [["up", 5], ["back", 14],
    ["up", "rest"]]; a row past the last segment is the last segment's kind."""
    rows = []
    for kind, n in segments:
        if n == "rest":
            rows.append((kind, None))
            break
        rows.extend([(kind, 1)] * int(n))

    def kind_of(x, v):
        i = bottom - 1 - v
        if i < 0:
            return rows[0][0] if rows else "up"
        if i < len(rows):
            return rows[i][0]
        return rows[-1][0] if rows else "back"
    return kind_of


# --- a round thing: the sprite turned like a lathe ---

def _outline_like(c):
    """A sprite's outline colour: dark, or a dark neutral grey."""
    h, s, val = hsv(c)
    return _dark(c) or (s < 0.12 and val < 0.6)


def canopy(own, kind_of, bottom):
    """A plant's crown: fronds lying level, leaves hanging from them. `kind_of(x, v)` says
    of each row, up from `bottom` (the row under the crown), whether it stands ("up": a
    leaf hanging, a tip curling) or lies ("back": a frond, reaching away). Each column is
    then moved along the line of sight, its picture where the GBA draws it, till its fronds
    lie level with the others'. Returns {(x, v): (y, z)}, before it's placed: the fronds at
    height 0, to be moved along the line of sight as a whole (y + k, z + k)."""
    cols = _columns(own)
    out = {}
    for x, vs in cols.items():
        ups = 0
        frond = None
        cells = {}
        for v in range(bottom - 1, vs[0] - 1, -1):
            if kind_of(x, v) == "up":
                cell = (ups, v + ups)
                ups += 1
            else:
                cell = (ups - 1, v + ups)
                if frond is None:
                    frond = ups - 1
            if (x, v) in own and cell not in cells.values():
                cells[(x, v)] = cell
        k = -(frond if frond is not None else ups - 1)
        for p, (y, z) in cells.items():
            out[p] = (y + k, z + k)
    return out


def lathe_outlined(sprite, own, y_base):
    """Where a round thing's picture outlines it, for outline(): a row whose end pixel is
    an outline round its sides; its top (its foot) where the pixel above (below) is an
    outline drawn across, not a row's end (the side of a pot under a wider head)."""
    foot = max(v for _, v in own)
    rows = {}
    for (x, v) in own:
        rows.setdefault(v, []).append(x)

    def drawn(x, v):
        c = sprite.colour(x, v)
        return c is not None and _outline_like(c)

    def where(key, way):
        x, y, _ = key
        v = foot - (y - y_base)
        dx, dy = way
        if dx:
            xs = rows.get(v)
            return bool(xs) and drawn(min(xs) if dx < 0 else max(xs), v)
        n = v - 1 if dy > 0 else v + 1
        xs = rows.get(n)
        return (x, n) in own and drawn(x, n) and min(xs) < x < max(xs)
    return where


def lathe(grid, sprite, own, y_base, z_axis, owner="", axis_x=None, depth=1.0):
    """A round thing (a plant, a flower, a pot): each row of its pixels a slice round a
    vertical axis at (axis_x, z_axis), each pixel as deep as the slice is round there (a
    circle through the row's outermost pixels), wearing its own colour through. Its bottom
    row stands at height `y_base`. Round, so seen from the GBA's angle its edges sit a little
    higher than drawn."""
    if not own:
        return
    xs = [x for x, _ in own]
    cx = axis_x if axis_x is not None else (min(xs) + max(xs)) / 2.0
    foot = max(v for _, v in own)
    # (the outline round the silhouette is the sprite's way of drawing an edge; turned, the
    # edge is the shape's own: left out where an inner colour is beside it)
    outline_like = _outline_like
    inner = set()
    for (x, v) in own:
        c = sprite.colour(x, v)
        edge = any((x + dx, v + dy) not in own for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if c is not None and outline_like(c) and edge:
            continue
        inner.add((x, v))
    # (an outline inside it takes the colour beside it)
    colour_of = {}
    for (x, v) in inner:
        c = sprite.colour(x, v)
        if c is not None and outline_like(c):
            near = sorted((abs(xx - x), xx) for (xx, vv) in inner if vv == v and not outline_like(sprite.colour(xx, vv)))
            if near:
                c = sprite.colour(near[0][1], v)
        colour_of[(x, v)] = c
    rows = {}
    for (x, v) in own:
        rows.setdefault(v, [])
    for (x, v) in inner:
        rows[v].append(x)
    for v in list(rows):
        if not rows[v]:
            rows[v] = [x for (x, vv) in own if vv == v]
    for v, row in rows.items():
        r = max(abs(x - cx) for x in row) + 0.5
        y = y_base + (foot - v)
        for x in row:
            c = colour_of.get((x, v)) or sprite.colour(x, v)
            if c is None:
                continue
            dx = x - cx
            t = depth * (max(0.0, r * r - dx * dx)) ** 0.5     # (`depth`: how deep to how wide)
            for z in range(int(round(z_axis - t)), int(round(z_axis + t)) + 1):
                grid.set(x, y, z, c, owner)
