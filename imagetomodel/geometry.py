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

    def __init__(self):
        self.polys = []            # [(name, [(x, y, z), ...], [(u, v), ...])]

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


def normal(pts):
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = pts[0], pts[1], pts[2]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    n = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    length = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
    return (0.0, 0.0, 0.0) if length == 0 else (n[0] / length, n[1] / length, n[2] / length)
