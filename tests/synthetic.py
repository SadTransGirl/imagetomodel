"""A building sprite drawn by hand, in the GBA's manner, to test the modeller on: a flat
top panel with slopes at its sides, a dark band along the roof's foot, a wall of two rows
with windows and a door, all outlined."""
import numpy as np
from PIL import Image

PANEL, LEFT, RIGHT, LINE = (239, 148, 115), (255, 206, 140), (206, 82, 74), (222, 107, 99)
DARK, RED, OUTLINE = (90, 90, 115), (173, 33, 41), (66, 74, 107)
WALL, WALL2, GLASS, GLASS2, SILL, DOOR = (206, 222, 239), (173, 189, 206), (99, 165, 222), (156, 214, 255), (206, 181, 107), (189, 99, 74)


def house(W=80, H=80, top=8, wall_top=43, band=3, xl=12, xr=67, drop=6, scale=1):
    im = np.zeros((H, W, 4), dtype=np.uint8)

    def put(x, y, c):
        if 0 <= x < W and 0 <= y < H:
            im[y, x] = (*c, 255)
    eave = wall_top - band
    # the top panel, with lines across it every 8 rows; its outline at the top
    for y in range(top, eave):
        for x in range(xl, xr):
            put(x, y, OUTLINE if y == top else (LINE if (y - top) % 8 == 5 else PANEL))
    # the slopes: each a parallelogram, its back edge rising to the panel's, its front
    # edge dropping from the panel's front corner to the eave corner
    for x in range(0, xl):
        t = (xl - x) / float(xl)
        y0 = top + int(round(drop * t))
        y1 = eave + int(round(drop * t))
        for y in range(y0, y1):
            put(x, y, OUTLINE if x == 0 or y == y0 else LEFT)
        for y in range(y1, y1 + band):
            put(x, y, [DARK, RED, OUTLINE][min(band - 1, y - y1)])
    for x in range(xr, W):
        t = (x - xr + 1) / float(W - xr)
        y0 = top + int(round(drop * t))
        y1 = eave + int(round(drop * t))
        for y in range(y0, y1):
            put(x, y, OUTLINE if x == W - 1 or y == y0 else RIGHT)
        for y in range(y1, y1 + band):
            put(x, y, [DARK, RED, OUTLINE][min(band - 1, y - y1)])
    # the band along the panel's foot
    for x in range(xl, xr):
        for k in range(band):
            put(x, eave + k, [DARK, RED, OUTLINE][min(band - 1, k)])
    # the wall, a pixel in from each side, outlined, a lighter line every 8 rows
    for y in range(wall_top, H):
        for x in range(1, W - 1):
            edge = x in (1, W - 2) or y == H - 1
            put(x, y, OUTLINE if edge else (WALL2 if (y - wall_top) % 8 == 7 else WALL))
    # (the wall under the slopes' bands, where they come down past wall_top)
    for x in range(0, xl):
        t = (xl - x) / float(xl)
        y1 = eave + int(round(drop * t)) + band
        for y in range(wall_top, y1):
            if 1 <= x < W - 1:
                put(x, y, OUTLINE if x == 1 else WALL)

    def window(x0, y0, w=10, h=8):
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + w):
                frame = x in (x0, x0 + w - 1) or y in (y0, y0 + h - 1)
                put(x, y, OUTLINE if frame else (SILL if y == y0 + h - 2 else (GLASS if y < y0 + h // 2 else GLASS2)))
    window(16, wall_top + 5)
    window(40, wall_top + 5)
    window(40, wall_top + 21)
    # the door, at the wall's foot
    for y in range(H - 16, H - 1):
        for x in range(15, 31):
            frame = x in (15, 30) or y == H - 16
            put(x, y, OUTLINE if frame else DOOR)
    img = Image.fromarray(im, "RGBA")
    if scale > 1:
        img = img.resize((W * scale, H * scale), Image.NEAREST)
    return img
