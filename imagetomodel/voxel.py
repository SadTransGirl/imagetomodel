"""True voxels: one cube a GBA pixel, as made by hand in MagicaVoxel. A sprite is
unprojected, not extruded: the GBA draws a cube's front face on one row and its top face on
the row above, so the bottom rows of each column of a thing are its front, standing, and
the rows above them its top, lying back. Rounded corners, rims and outlines come out as the
art draws them, and the model seen from the GBA's angle gives the picture back but for the
row along each front's top edge (a cube has one colour; its top and front show two rows).

Coordinates are the room's pixels: x across (east), y up from the floor (the floor's cubes
at y = -1), z down the picture (south, towards the camera). A cube (x, y, z) shows its front
face on row z - y and its top face on row z - y - 1."""
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
