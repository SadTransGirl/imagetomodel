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

## Rooms

```sh
imagetomodel room center-1f.png -o out
```

### True voxels (the default)

A room comes out as cubes, one a GBA pixel, as a room is made by hand in MagicaVoxel:
`NAME.vox` for the whole room and `NAME-things/thing-N/thing-N.vox` for each thing, to open
and work on in MagicaVoxel, with the port's model JSON, a `.glb` and an `.obj` meshed from
the cubes beside each (faces merged where they're one colour, each wearing its colour from
a palette painted in the canvas's strip).

A sprite is unprojected, not extruded. The GBA draws a cube's front face on one row and
its top face on the row above, so the bottom rows of each column of a thing are its front,
standing, and the rows above them its top, lying back: a rounded corner, a rim, an
outline come out as the art draws them, and the model seen from the GBA's angle gives the
picture back but for the row along each front's top edge (a cube has one colour, and
shows two rows there). How tall each front is comes from the art:

- **in the room** (a counter, a table, a cushion): the rows from its foot up to where its
  top's own colour takes over, so a counter is its darker edge and its kickplate;
- **against the wall** (a shelf, a PC): it reaches back to the wall; its lid the rows
  above the first dark line drawn right across it (the PC's grey cap, the shelf's green
  top), else as deep as the wall is far;
- **on a counter** (a machine whose foot is on the counter's back edge): standing on the
  counter, the counter's height up, its lid as deep as the wall is far (the healing
  machine's tray of balls lies on top);
- **standing on its own** (a plant, a flower): its pixels standing, thickened backwards
  (`--thick`) only as far as the GBA's view keeps.

The sides and the back the GBA never drew take each row's body colour (the first colour
inside its outline). Every thing is then **outlined** as the GBA outlines a sprite, from the
sides and the back too: the cubes on the edge of its silhouette seen from the left, the
right, the back and the front wear its outline colour, but for those the GBA's view shows,
which keep the picture's. A round thing is outlined only where its picture is (a plant's
leaves and pot, not a flower's petals); `"outline": false` in a fix turns it off. The back wall's face is the picture
exactly, stepping forward a cube a column where the art cuts the corner, so its bands run
round it; the floor is a layer of cubes under everything, the floor's own tile under each
thing. Things are cut from their background pixel by pixel: on the wall's rows against
the plain wall at that height (or any plain wall tile, the room's corners aside), on a
counter against its colour and its edges. The wall's own **decorations** (a scallop: a
small patch of the wall's and the corners' colours, touching nothing else) stay the
wall's, and what stands against the wall takes the pixels of its own that run on up into
the wall's rows (the top of a map, a shelf's top edge).

Some things are found and built their own way:

- **a counter** is one object, its front and its wings (the runs of things in a band of the
  same colour across the room, touching), so its U shape comes from how far back each
  column is drawn; what stands on it (a flower, a machine, a PC) is cut from its top,
  which wears its own colour there;
- **round things** (a plant, a flower: a thing standing on its own on a background) are
  turned like a lathe, each row a slice half as deep as the row is wide (`round_depth`
  in a fix sets it; 1 is fully round), the sprite's outline left out of the turning;
- **an escalator** (tiles drawn as treads, with the things touching them) is built as
  steps between two glass balustrades, a handrail along each, climbing where its handrails
  climb in the picture;
- **the shadow** the GBA draws beside a thing on the floor is the floor's; where it falls
  on the wall, the wall is built plain behind it (the thing, standing in 3D, casts it);
- **holes** in a thing (a pixel with the thing to its left, its right and above it) are
  closed, wearing the thing's own colour where they showed the counter or the wall
  through it: a machine's tray is whole, its lip a panel;
- **what hangs on the wall** (a TV, a map) stands proud of it as a box, as deep as a fifth
  of its height (`depth` in a fix sets it), moved up as far as it stands out so its picture
  stays where the GBA draws it.

### Fixes: things set by hand

A sprite can't always say what stands and what lies: a machine can be a lip standing, a
tray lying, a panel standing behind it. A **fix** says, by the thing's fingerprint (a hash
of its tiles' art as drawn, like LGFR3D's tile prints; each thing's is in
`NAME-labels.json`), so it holds for that thing in every room drawn with it, as LGFR3D's
own `building_fixes.json` does:

```json
{
  "ffcd5c6f8280": {
    "name": "Pokemon Center healing machine",
    "profile": [["up", 5], ["back", 14], ["up", 6], ["back", "rest"]],
    "recess": [{"rows": [9, 21], "cols": [8, 23], "depth": 2}],
    "raise": [{"colours": ["6bc66b"], "rows": [9, 22]}, {"colours": ["ff8473"], "rows": [23, 27]}],
    "keep": true
  }
}
```

- `profile`: up from the thing's foot, how many rows stand (`up`) and lie (`back`) in turn,
  `rest` the rows left; `profiles`: one for a range of columns (a keyboard lying beside a
  box);
- `recess`: rows and columns (from the thing's box's corner) set `depth` cubes in (a
  shelf's inside, a screen; where they lie, sunk along the line of sight: a tray in its
  rim, its picture where the GBA draws it);
- `raise`: pixels of these colours (within rows, and `cols`) a cube proud (a tray's balls,
  keys, a tab); one, or a list;
- `keep`: the holes closed in it keep the picture's colours (a tab the counter's colour,
  which would otherwise be taken for the counter);
- `drop`: pixels that aren't the thing's; `front`: its front's height; `depth`: how far
  a thing on the wall stands out; `shape`: `box`, `round`, `slab`, `escalator`, `plant`;
  `round_depth`: how deep a round thing is, against its width; `back_to_wall`.

A **plant** (a potted palm) is built in three parts, each where the GBA draws it: its pot
by its `profile`, its `soil` (a colour over rows and columns) sunk by a `recess`; its
`trunk` (the `cols` and `rows` it's drawn in, its `stripes` the rows whose colours repeat
up it) a square column standing in the soil, outlined at its corners, up to its crown; and
its `crown` (`rows`, with a `profile` and `profiles` up from its foot as a thing's): the
rows that lie are fronds, every column moved along the line of sight till they're level
over the trunk, the rows that stand leaves hanging from them or tips curling:

A plant with no trunk (a flower) rests its crown on its pot: `"thick"` makes what stands
in it (a flower's head) that many cubes deep, `"mirror"` lays its leaves in front of what
stands behind it too; `pot_depth` gives a pot whose top the leaves hide a square top, its
`rim` (a colour) round the `soil`'s colour. `include` adds pixels the background check took
(a white stem on a counter with white in it); `"outline": false` leaves a thing to its
picture's own outlines.

```json
"crown": {"rows": [0, 14], "profile": [["up", 6], ["back", "rest"]],
          "profiles": [{"cols": [6, 9], "profile": [["up", 2], ["back", "rest"]]}]},
"trunk": {"cols": [6, 9], "rows": [15, 20], "stripes": [16, 19]},
"soil": {"colour": "ad9442", "rows": [17, 21], "cols": [6, 9]}
```

The library shipped with the tool (`imagetomodel/fixes/rooms.json`) has the Pokemon
Center's shelf, PC, healing machine, nurse's PC, potted palm and flowers; `--fixes FILE` adds more, and a spec's
`"objects"` may key a fix by the thing's first tile (`"5,1"`) for one room.

`--style faces` builds the earlier way instead: boxes with their sides painted from the
art and their faces in relief.

### How a room is read

A screenshot of a room (the GBA's own 240×160 frame, or art drawn its way, at any scale)
is cut on its 16-pixel grid, wherever the grid lies on it (the screen scrolls with the
player: the offset at which the most tiles repeat is the grid's). Each tile is told from
its picture alone:

- **void**: the black beyond the room (cleared, so nothing is drawn there);
- **wall**: the rows at the top whose plain tile repeats across the room (the back wall),
  with what hangs on it painted on its face;
- **wall_side**: the wall's ends down the outermost columns, and a column's own repeated
  tile when it isn't the floor's (a wall down the side of the room);
- **floor**: the tile repeated most, and tiles of its colours or their shades (a shadow,
  a lamp's glow), and tiles half on the void's edge (a mat, a corner cut);
- **thing**: the rest.

The room is built as the port builds rooms, unsquashed (the game squashes rooms, and
models in them, as it builds them): the floor flat; the back wall a thin slab its rows
tall (`--wall-thick`, 4 px), its face the picture a pixel a pixel, its top the band's
colour; a thin wall down each side where the art has one, and across each corner the art
cuts on the diagonal; and each thing read column by column:

- a thing on the wall's rows with plain floor under it **hangs on the wall** (a poster, a
  map): painted there;
- a thing on the wall's rows with a thing (or a pot drawn on the floor) under it is one
  with it and **stands tall against the wall** (a shelf, a machine), a box from the wall's
  face to its foot;
- a thing coming down from the wall onto a band of things alike right across the room
  (a counter) stands with **its foot on the counter's back edge**, and the counter's
  tile starts the counter's own run;
- a thing in the room **stands free**: a box with its front standing at its foot
  (`--front-height`, 16 px; a thing of one tile `--low-height`, 8 px: a stool, a
  cushion) and its top lying back from its head;
- a thing standing on its own on a background (its own pixels at least as tall as wide,
  clear of the tile's sides, on the floor's or the wall's colours or a counter's) is a
  **cutout**, the GBA's picture standing with the background cut away (a plant, a flower
  or a sign on a counter's top), and what's behind it wears a plain tile instead.

Every thing is cut from its background (the floor's shades, the wall's colours, a
counter's colour under what stands on it) and its box is the size of its own pixels, so
the floor shows round a cushion and a table is as big as it is drawn. The faces the GBA
never drew are **painted from the thing's own art**, as a pixel artist extrudes a sprite,
into the canvas's strip:

- its **sides**: the front's edge profile (each row's body colour inside its outline)
  across the depth, the outline continuing along the back edge, so a counter's end keeps
  its red top band, its dark front and its grey foot;
- its **back**: a panel in the front's row colours, outlined at both ends;
- its **top**, standing against the wall: its top row's colours back over it, outlined
  (`--tops painted`; from the GBA's angle such a top covers the wall's band above it, as
  the port's own tops do; `--tops picture` wears the band there instead, exact from that
  angle). A free-standing thing's top is the art itself.

Each thing is also a model of its own, in `NAME-things/thing-N/`: `thing-N.lgfr.json` (as
the port keeps one, standing for its tiles), `thing-N.glb`, `thing-N.obj` with its `.png`
(its tiles' picture, cut from the background, with its painted faces in the strip), and
renders from three-quarter and from behind, to look at in Blender or Blockbench and to
bring back with the kit.

The GBA's picture can't tell a tall front from a deep top, so a reading is a guess, and
`NAME-labels.png` shows it: each tile's label (W wall, S side wall, T thing) and each
thing's kind (h hanging, A against the wall, b a box, B a box standing tall, c a cutout).
Correct it
with `--spec`: `{"wall_rows": 2, "tiles": {"9,6": "box", "7,2": "floor"}}`, where a tile
may be `void`, `floor`, `wall`, `wall_side`, `thing`, or `box` (a thing standing tall) and
`counter` (a thing standing free, low) to force its kind.

What's written is as for a building, and besides: `NAME-labels.json` (the grid, the wall's
rows, the labels, the things and the objects they make with their boxes), the per-thing
folders above, and the room's `.glb`/`.obj` with an object per thing (and `floor`, `wall`,
`side-left`, `side-right`), so that `tools/kit.py import` makes a model of each thing,
standing wherever its pictures are, in every room with it.

A thing seen from the GBA is used twice when another stands on it (a machine on a counter):
its tile is the counter's top and the machine's foot, both faithful; from other angles
the machine's lower rows show on the counter's top as well. The walls standing at the
room's sides and corners cover, from the GBA's angle, the rows above them (the GBA drew
them flat), so a room scores a little under a building.

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
imagetomodel/room.py       a room from a screenshot: the grid, each tile's label, the things, the mesh
imagetomodel/voxel.py      cubes: a sprite unprojected, meshed greedily, written as MagicaVoxel .vox
imagetomodel/cli.py        imagetomodel building ..., imagetomodel room ...
tests/synthetic.py         a building drawn by hand, in the GBA's manner, for the tests
tests/synthetic_room.py    a room, likewise
```

Python 3.10+, numpy and Pillow. `python3 -m pytest tests`.

No Nintendo art is in this repository: the tests draw their own building.

## Next

- rooms: stairs drawn as stairs (not an escalator), a person standing in the screenshot
  (masked out), water; more fixes for the things of other rooms;
- buildings in true voxels too;
- more roof styles read from the art: a ridge with two slopes (a gable), the eave that
  curves down over the wall (a Center's, a Mart's), a flat roof with a rim;
- things on the roof (a chimney, a skylight), porches set forward.
