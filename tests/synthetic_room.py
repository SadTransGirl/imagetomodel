"""A room drawn by hand, in the GBA's manner, to test the room mode on: the black beyond
it, a back wall of two rows with a poster hanging on it, a wall's end at each side, a
floor of light and shadowed tiles, a counter across the middle with a machine standing on
its back edge against the wall, a plant at the wall's foot, a stool on the floor, a mat
on the void's edge."""
import numpy as np
from PIL import Image

BLACK = (0, 0, 0)
WALL_TOP, WALL_FACE, WALL_LINE = (255, 255, 255), (247, 239, 165), (231, 140, 99)
OUTLINE, GREY, DARK = (82, 82, 107), (181, 181, 165), (123, 123, 123)
FLOOR, FLOOR_SQ, SHADOW, SHADOW_SQ = (247, 247, 206), (231, 222, 156), (231, 222, 156), (198, 181, 140)
END = (189, 99, 74)
COUNTER, COUNTER_FRONT, SCREEN, PLANT, POT, STOOL, MAT = (255, 132, 115), (206, 82, 66), (156, 189, 214), (107, 198, 107), (189, 99, 74), (255, 247, 132), (231, 82, 66)


def room(cols=12, rows=8, scale=1):
    W, H = cols * 16, rows * 16
    im = np.zeros((H, W, 4), dtype=np.uint8)
    im[:, :] = (*BLACK, 255)

    def put(x, y, c):
        if 0 <= x < W and 0 <= y < H:
            im[y, x] = (*c, 255)

    def tile_fill(tx, ty, f):
        for y in range(16):
            for x in range(16):
                put(tx * 16 + x, ty * 16 + y, f(x, y))
    # the floor: rows 2.. but the last, light; a shadow under the counter's row
    for ty in range(2, rows - 1):
        for tx in range(cols):
            sq = (tx * 16 // 8 + ty * 16 // 8)
            dark = ty == 4
            tile_fill(tx, ty, lambda x, y, dark=dark: ((SHADOW_SQ if dark else FLOOR_SQ) if (x // 8 + y // 8) % 2 == 0 and 2 <= x % 8 <= 5 and 2 <= y % 8 <= 5 else (SHADOW if dark else FLOOR)))
    # the back wall: rows 0 and 1: a white top band, cream face with salmon lines
    for tx in range(cols):
        tile_fill(tx, 0, lambda x, y: WALL_TOP if y < 4 else (WALL_LINE if y in (5, 6, 8, 9) else WALL_FACE))
        tile_fill(tx, 1, lambda x, y: OUTLINE if y >= 13 else (WALL_LINE if y in (8, 9) else WALL_FACE))
    # the wall's ends: a brown band down each side's row 2
    for tx in (0, cols - 1):
        tile_fill(tx, 2, lambda x, y: END if y < 10 else (OUTLINE if y == 10 else FLOOR))
    # a poster hanging on the wall (two tiles wide, on row 0 and 1)
    for y in range(6, 26):
        for x in range(cols * 8 - 14, cols * 8 + 14):
            frame = x in (cols * 8 - 14, cols * 8 + 13) or y in (6, 25)
            put(x, y, OUTLINE if frame else SCREEN)
    # the counter: rows 3 and 4 from column 2 to 8: its top light, its front dark with a grey foot
    for tx in range(2, 9):
        tile_fill(tx, 3, lambda x, y: OUTLINE if y == 0 else COUNTER)
        tile_fill(tx, 4, lambda x, y: COUNTER if y < 3 else (OUTLINE if y == 3 else (COUNTER_FRONT if y < 11 else (OUTLINE if y == 11 else GREY))))
    # a machine standing on the counter's back edge, against the wall: tiles (4, 1..3)
    for y in range(20, 58):
        for x in range(66, 78):
            frame = x in (66, 77) or y in (20, 57)
            put(x, y, OUTLINE if frame else (SCREEN if y < 36 else GREY))
    # a plant at the wall's foot: tile (1, 1) and its pot in (1, 2)
    for y in range(18, 30):
        for x in range(18, 30):
            if (x - 24) ** 2 + (y - 24) ** 2 <= 30:
                put(x, y, PLANT)
    for y in range(30, 42):
        for x in range(20, 28):
            put(x, y, OUTLINE if y in (30, 41) or x in (20, 27) else POT)
    # a stool on the floor: tile (9, 6)
    for y in range(6 * 16 + 4, 6 * 16 + 14):
        for x in range(9 * 16 + 3, 9 * 16 + 13):
            put(x, y, OUTLINE if y in (6 * 16 + 4, 6 * 16 + 13) or x in (9 * 16 + 3, 9 * 16 + 12) else STOOL)
    # the mat on the void's edge: the last row, the middle column
    for y in range((rows - 1) * 16, (rows - 1) * 16 + 8):
        for x in range(cols * 8 - 8, cols * 8 + 8):
            put(x, y, OUTLINE if y == (rows - 1) * 16 or x in (cols * 8 - 8, cols * 8 + 7) else MAT)
    img = Image.fromarray(im, "RGBA")
    if scale > 1:
        img = img.resize((W * scale, H * scale), Image.NEAREST)
    return img
