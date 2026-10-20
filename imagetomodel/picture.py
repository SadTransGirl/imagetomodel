"""A pixel-art sprite as the modeller reads it: at its native resolution (an upscaled
picture is brought back down by the size of its pixel blocks), with what's drawn (its
clear pixels, and the ground round it if asked, not drawn)."""
import math
from collections import Counter

import numpy as np
from PIL import Image

PX = 16                      # (pixels a tile, as the GBA's)


def detect_scale(rgba, limit=64):
    """How many times the picture was scaled up: the largest block size its pixels are
    constant over (1 when it's at its native size)."""
    h, w = rgba.shape[:2]
    best = 1
    for s in range(2, min(limit, w, h) + 1):
        if w % s or h % s:
            continue
        blocks = rgba.reshape(h // s, s, w // s, s, 4)
        first = blocks[:, :1, :, :1, :]
        if (blocks == first).all():
            best = s
    return best


def hsv(c):
    r, g, b = c[0] / 255.0, c[1] / 255.0, c[2] / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    v = mx
    d = mx - mn
    s = 0.0 if mx == 0 else d / mx
    if d == 0:
        h = 0.0
    elif mx == r:
        h = ((g - b) / d) % 6
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return (h * 60.0) % 360.0, s, v


def luma(c):
    return (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255.0


class Sprite:
    """A sprite at its native size: `rgba` (rows, columns, 4), `drawn` (rows, columns) where
    a pixel is part of the thing, `W` and `H` in pixels, `scale` how many times the file
    was bigger."""

    def __init__(self, image, scale=None, ground=None):
        rgba = np.array(image.convert("RGBA"))
        if scale is None:
            scale = detect_scale(rgba)
        if scale > 1:
            rgba = rgba[::scale, ::scale]
        self.scale = scale
        self.rgba = np.ascontiguousarray(rgba)
        self.H, self.W = rgba.shape[:2]
        self.drawn = self.rgba[:, :, 3] > 127
        if ground:
            self.clear_ground(ground)
        self.palette = Counter(map(tuple, self.rgba[self.drawn][:, :3].tolist()))

    @classmethod
    def load(cls, path, scale=None, ground=None):
        return cls(Image.open(path), scale=scale, ground=ground)

    def clear_ground(self, colours):
        """The ground round the sprite cleared: pixels of these colours reached from the
        picture's edge (a crop from a map: the grass round a house), not the same colours
        inside it."""
        colours = {tuple(c) for c in colours}
        H, W = self.H, self.W
        ground = np.zeros((H, W), dtype=bool)
        for y in range(H):
            for x in range(W):
                if tuple(self.rgba[y, x, :3].tolist()) in colours or not self.drawn[y, x]:
                    ground[y, x] = True
        seen = np.zeros((H, W), dtype=bool)
        todo = [(x, y) for x in range(W) for y in (0, H - 1)] + [(x, y) for y in range(H) for x in (0, W - 1)]
        while todo:
            x, y = todo.pop()
            if not (0 <= x < W and 0 <= y < H) or seen[y, x] or not ground[y, x]:
                continue
            seen[y, x] = True
            todo += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        self.drawn &= ~seen
        self.rgba[seen] = 0

    def colour(self, x, y):
        """A pixel's colour (r, g, b), or None where nothing is drawn."""
        if 0 <= x < self.W and 0 <= y < self.H and self.drawn[y, x]:
            return tuple(int(v) for v in self.rgba[y, x, :3])
        return None

    def row_counts(self, y, x0=0, x1=None):
        """A row's drawn colours, counted, between two columns."""
        x1 = self.W if x1 is None else x1
        counts = Counter()
        for x in range(max(0, x0), min(self.W, x1)):
            c = self.colour(x, y)
            if c is not None:
                counts[c] += 1
        return counts

    def dominant(self, y, x0=0, x1=None):
        counts = self.row_counts(y, x0, x1)
        return counts.most_common(1)[0][0] if counts else None

    def column_drawn(self, x):
        """The rows a column is drawn in: (first, last + 1), or None."""
        ys = np.nonzero(self.drawn[:, x])[0]
        if len(ys) == 0:
            return None
        return int(ys[0]), int(ys[-1]) + 1

    def row_drawn(self, y):
        xs = np.nonzero(self.drawn[y])[0]
        if len(xs) == 0:
            return None
        return int(xs[0]), int(xs[-1]) + 1

    def image(self):
        return Image.fromarray(self.rgba, "RGBA")

    def crop(self, x0, y0, w, h):
        """A sprite of a part of this one (pixels), at its native size."""
        out = Sprite.__new__(Sprite)
        out.scale = self.scale
        out.rgba = np.ascontiguousarray(self.rgba[y0:y0 + h, x0:x0 + w])
        out.H, out.W = out.rgba.shape[:2]
        out.drawn = self.drawn[y0:y0 + h, x0:x0 + w].copy()
        out.palette = Counter(map(tuple, out.rgba[out.drawn][:, :3].tolist()))
        return out

    @property
    def cols(self):
        return math.ceil(self.W / PX)

    @property
    def rows(self):
        return math.ceil(self.H / PX)
