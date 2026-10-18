# imagetomodel

A 3D model from a piece of pixel art, faithful to it, for [LGFR3D](https://github.com/SadTransGirl/LGFR3D).

Give it a building as the GBA draws one (at any scale: an 8× upscale is brought back
to its pixels) and it measures the picture, builds the building in 3D and writes it in the
forms LGFR3D takes: the game's own model JSON, and a `.glb`/`.obj` in the modelling kit's
conventions that `tools/kit.py import` brings back.

```sh
pip install -e .
imagetomodel building house.png -o out
```

```
house: 80x80 (scale 8), wall from row 43, band 3, panel [12, 67], left edge 0 rows 14-45, ...
faithful: 99.9% of the sprite's pixels exact, 99.9% covered, 0 drawn past it, colour distance 0.000
```

## What "faithful" means here

The GBA draws every map from one angle: a pixel up shows as a row, and a pixel back shows
as a row. LGFR3D's own test of its 3D (`--faithful`) renders the world from that angle and
scores it against the GBA's picture. This tool builds for that test: every face the GBA
would see takes its texture coordinates from where the GBA draws its corners, so the model,
seen from the GBA's angle, gives the sprite back pixel for pixel. The faces the GBA never
drew (the sides, the back, the undersides) wear a plain strip of the wall or the band's
colour, and the shape between is the tool's reading of the art:

- **the wall** stands from its rows, as tall as they are, a pixel in from the roof where the
  art draws it so;
- **the door and the windows** are set back into it (`--inset`, `--door-inset`), along the
  GBA's line of sight, so their pictures stay exactly where they were;
- **the roof's top panel** lies over the wall as deep as the sprite draws it (its rows),
  a slope down to each side's eave where the art shades one (the side panels, their back
  edges rising to the top panel's, their front edges dropping to the corners);
- **the band** along the roof's foot (its dark edge) stands as its front edge and follows
  the slopes' front edges down at the corners.

Each output folder has `NAME-faithful.png`: the sprite, the model from the GBA's angle, and
what differs (red), with the score printed. Other views (`--views three-quarter,side,
front,back,top`) show the shape.

## What's written

| file | what |
|---|---|
| `NAME.lgfr.json` | the model as LGFR3D keeps one (`godot/models/`): a `mesh` piece in pixels, wearing a canvas with the sprite painted on it |
| `NAME.glb` | the model for Blender, Blockbench, Godot, and for `tools/kit.py import` |
| `NAME.obj` + `.mtl` + `.png` | the same for Blender (File → Import → Wavefront), its polygons kept whole |
| `NAME-measure.json` | what was read from the sprite: a spec to correct and pass back |
| `NAME-faithful.png`, `NAME-VIEW.png` | the pictures |

## Putting it in the game

A model stands in the game wherever its tiles' pictures are; it knows them by their
fingerprints (`prints`), which only the game has (its catalogue: `godot --path godot --
--catalog DIR`). Three ways:

1. **The sprite is the game's own art** (a house from the tileset, upscaled or not): give
   the catalogue and the thing's id, and the model wears the game's picture and stands
   where it is:
   `imagetomodel building house.png -o out --catalog build/catalog-towns --thing 5x5-1a2b3c4d --game-art`
2. **New art for a building the game has** (a redraw): the same, without `--game-art`:
   the sprite is painted over the game's picture, pixel by pixel, and the model stands
   where that building is.
3. **No catalogue to hand**: the `.json` is written with `prints` of 1 (a print no picture
   has: it stands nowhere, rather than everywhere) and a note; bring `NAME.glb` back
   instead with `python3 tools/kit.py import out/NAME.glb --catalog build/catalog-towns`,
   which finds its block by the picture, or pass `--thing ID` so the file carries it.

Models from this tool have priority 1, as the port's own building modeller's; the kit's
hand-made ones (2) win over them.

## Correcting a reading

`NAME-measure.json` holds every measure (rows and columns of the sprite, in pixels):
`wall_top`, `band`, `panel`, `top_centre`, `left`/`right` (each slope's `edge`, `top`,
`eave`, `drop`), `wall`, `door`, `windows`. Copy it, change what's wrong, and pass it with
`--spec`; a key given there wins over what's read. `--no-relief` keeps the wall flat;
`--depth N` makes the building N tiles deep with the roof's picture stretched over it (as
the port builds its deeper houses; the GBA view is then only as faithful as the stretch).
A sprite cropped from a map, with the ground round it, is cleared with
`--ground 70b858,88d068` (the ground's colours, hex).

## How it reads a sprite

`imagetomodel/measure.py`, in order: the first row drawn right across; the wall's colours
(the light ones found across the whole width of the bottom third, glass aside); the wall's
top, under the lowest dark line drawn across it with the roof's colours above; the band,
the rows above the wall that aren't the top panel's; the top panel's columns, the run of
its main colour through the middle; each slope, the roof past the panel, its back corner
the first row drawn at its outer column; the wall's columns; the door, a region of other
colours at the wall's foot, grown to its outline; the windows, the glass, grown to their
frames. Every step is a heuristic for art drawn in the GBA's manner; the spec is there
for what it gets wrong.

## Layout

```
imagetomodel/picture.py    the sprite at its native size (the upscale found and undone)
imagetomodel/measure.py    what the building is, read from the sprite
imagetomodel/building.py   the mesh from the measures
imagetomodel/geometry.py   faces with texture coordinates from the GBA's projection
imagetomodel/render.py     pictures of the model (a small rasterizer), the faithfulness score
imagetomodel/gltf.py       .glb and .obj in the kit's conventions
imagetomodel/lgfr.py       the model JSON as LGFR3D keeps it
imagetomodel/cli.py        imagetomodel building ...
tests/synthetic.py         a building drawn by hand, in the GBA's manner, for the tests
```

Python 3.10+, numpy and Pillow. `python3 -m pytest tests`.

No Nintendo art is in this repository: the tests draw their own building.

## Next

- rooms: a screenshot of a room cut on its 16-pixel grid, each tile labelled from its
  picture (floor, wall, counter, box, plant...) and built with the port's piece kinds;
- more roof styles read from the art: a ridge with two slopes (a gable), the eave that
  curves down over the wall (a Center's, a Mart's), a flat roof with a rim;
- things on the roof (a chimney, a skylight), porches set forward.
