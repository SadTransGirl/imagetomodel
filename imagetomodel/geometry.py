"""Triangles wearing the sprite. Coordinates are the port's, in pixels: x across (east), y
up from the ground, z down the map (south, towards the camera), from the block's top left
corner on the ground. The GBA draws a pixel up as a row and a pixel back as a row, so its
picture of a point is (x, z - y): a face whose texture coordinates are its corners'
pictures wears the sprite exactly as the GBA drew it, seen from the GBA's angle."""


def project(p):
    """Where the GBA draws a point: (column, row) of the sprite."""
    return (p[0], p[2] - p[1])


class Mesh:
    """Polygons (3 or more corners, flat, anticlockwise seen from outside), each corner with
    a texture coordinate in the sprite's pixels (u across, v down from its top)."""

    def __init__(self, strip_v=None, strip_w=None):
        self.polys = []            # [(name, [(x, y, z), ...], [(u, v), ...])]
        self.palette = []          # plain colours painted in a strip under the sprite
        self.patches = []          # patches painted there too: ((x, y) in the strip, rows)
        self.strip_v = strip_v     # (the strip's first row: the sprite's height)
        self.strip_w = strip_w     # (the strip's width: the sprite's)
        self._shelf_x = self._shelf_y = self._shelf_h = 0

    def patch(self, pixels):
        """A texture coordinate origin for a patch of pixels (a list of rows of (r, g, b)
        or None, all rows as long) painted in the strip under the sprite: a tile made up
        (a plain wall), a side painted for a thing. Patches are packed in shelves."""
        rows = [[tuple(int(v) for v in p[:3]) if p is not None else None for p in row] for row in pixels]
        h = len(rows)
        w = len(rows[0]) if rows else 0
        width = self.strip_w or 256
        if self._shelf_x + w > width and self._shelf_x > 0:
            self._shelf_y += self._shelf_h
            self._shelf_x, self._shelf_h = 0, 0
        at = (self._shelf_x, self._shelf_y)
        self.patches.append((at, rows))
        self._shelf_x += w
        self._shelf_h = max(self._shelf_h, h)
        return (float(at[0]), float((self.strip_v or 0) + 1 + at[1]))

    def plain(self, colour):
        """A texture coordinate of a plain colour: a pixel painted in the strip under the
        sprite (the kit's painting strip), for the faces the GBA never drew."""
        colour = tuple(int(c) for c in colour[:3])
        if colour not in self.palette:
            self.palette.append(colour)
        i = self.palette.index(colour)
        w = self.strip_w or 256
        # (row after row of the strip when there are more colours than it is wide: a
        # voxel model's palette; patches are for meshes with few colours)
        return (i % w + 0.5, (self.strip_v or 0) + i // w + 0.5)

    def face(self, points, uv=None, name=""):
        """A polygon. `uv` None: each corner wears where the GBA draws it (the faithful
        faces); a single (u, v): the whole face that one pixel's colour (the sides the GBA
        never drew); a list: a texture coordinate a corner."""
        pts = [tuple(float(c) for c in p) for p in points]
        if len(pts) < 3:
            return
        if uv is None:
            tex = [project(p) for p in pts]
        elif isinstance(uv, tuple) and len(uv) == 2 and not isinstance(uv[0], (tuple, list)):
            tex = [(float(uv[0]), float(uv[1]))] * len(pts)
        else:
            tex = [(float(t[0]), float(t[1])) for t in uv]
        self.polys.append((name, pts, tex))

    def triangles(self):
        """(name, [p0, p1, p2], [t0, t1, t2]) for each triangle, the polygons fanned."""
        for name, pts, tex in self.polys:
            for j in range(1, len(pts) - 1):
                yield name, [pts[0], pts[j], pts[j + 1]], [tex[0], tex[j], tex[j + 1]]

    def bounds(self):
        xs = [p[0] for _, pts, _ in self.polys for p in pts]
        ys = [p[1] for _, pts, _ in self.polys for p in pts]
        zs = [p[2] for _, pts, _ in self.polys for p in pts]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def __len__(self):
        return sum(len(pts) - 2 for _, pts, _ in self.polys)


def strip_pixels(mesh, width):
    """What the mesh paints in the strip under the sprite: {(x, y): (r, g, b)}, y from 0
    (the strip's first row), x across its width."""
    out = {}
    w = getattr(mesh, "strip_w", None) or width
    for i, c in enumerate(getattr(mesh, "palette", []) or []):
        out[(i % w, i // w)] = c
    for (x0, y0), rows in getattr(mesh, "patches", []) or []:
        for j, row in enumerate(rows):
            for i, c in enumerate(row):
                if c is not None:
                    out[(x0 + i, 1 + y0 + j)] = c
    return out


def rebased(mesh, dx, dz, old_strip_v, new_strip_v, new_strip_w=None):
    """A copy of the mesh moved by (-dx, 0, -dz), its sprite texture coordinates with it,
    its strip (palette and patches) kept, under a sprite `new_strip_v` tall."""
    out = Mesh(strip_v=new_strip_v, strip_w=new_strip_w or mesh.strip_w)
    out.palette = list(mesh.palette)
    out.patches = list(mesh.patches)
    out._shelf_x, out._shelf_y, out._shelf_h = mesh._shelf_x, mesh._shelf_y, mesh._shelf_h
    for name, pts, tex in mesh.polys:
        moved = [(p[0] - dx, p[1], p[2] - dz) for p in pts]
        uvs = [(t[0], t[1] - old_strip_v + new_strip_v) if t[1] >= old_strip_v else (t[0] - dx, t[1] - dz) for t in tex]
        out.polys.append((name, moved, uvs))
    return out


def normal(pts):
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = pts[0], pts[1], pts[2]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    n = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    length = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
    return (0.0, 0.0, 0.0) if length == 0 else (n[0] / length, n[1] / length, n[2] / length)
