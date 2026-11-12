# COSMA

Static geomodel builder for CO2 storage. Exports corner-point GRDECL grids
(geometry, porosity, permeability and any user-defined property) for dynamic
simulators. Licensed under Apache-2.0.

Status: **M1 (part 1)**. Synthetic faulted box -> GRDECL -> independent
readers, UTM-safe placement, model boundary definition, and importers for
surfaces and well data. Not yet: fault sticks, variograms, QC reports, GUI.

## Conventions

- Internal units: metres. Depth positive down.
- The grid lives in **local coordinates** (near 0, 0). A `GridFrame` places it
  on the map: origin of local (0, 0) and the azimuth of local +y (the J
  direction), clockwise from north. Azimuth 0 is I east, J north, K down
  (left-handed). Rotation preserves handedness.
- Export is local coordinates + `MAPUNITS`/`MAPAXES` by default. Absolute map
  coordinates are available (`coordinates="absolute"`) but warn: readers that
  store x, y as float32 round to 0.5 m at a northing of 4.7 million, which
  distorts cell volumes (measured through xtgeo: -1.6% to +3.7% at 10 m cells;
  local coordinates: 4e-4%).
- Pillar order in COORD: I fastest, then J. Six numbers per pillar
  (x, y, z at top, then x, y, z at bottom).
- ZCORN order: K slowest, then top/bottom face of the layer, then J, then
  front/back edge of the cell, then I, then west/east edge (fastest).
  This is the array shape `(nz, 2, ny, 2, nx, 2)` flattened in C order.
- Property arrays: shape `(nx, ny, nz)` in memory, written I fastest.

## Model boundary

Define the areal extent either manually or from a polygon; both give a
`GridSpec` (frame, column edges, active-column mask).

    from cosma.grid.boundary import (grid_spec_from_cells, grid_spec_from_rectangle,
                                     grid_spec_from_extent, grid_spec_from_polygon)
    from cosma.io.polygons import read_boundary

    spec = grid_spec_from_extent(xmin, xmax, ymin, ymax, dx=100, dy=100)          # manual
    spec = grid_spec_from_cells(ox, oy, azimuth, nx, ny, dx, dy)                  # manual
    poly = read_boundary("aoi.geojson", target_crs="EPSG:32616")                  # import
    spec = grid_spec_from_polygon(poly, 100, 100, azimuth="auto", margin=50)      # fit grid

- Polygon formats: CSV / whitespace XYZ / IRAP-RMS ASCII (999 separators),
  GeoJSON, ESRI shapefile (`.shp`, with `.prj` and `.dbf` when present) and
  KML / KMZ. `read_polygon_records` also returns each polygon's name and
  attributes; `read_boundary(which=...)` picks one polygon by position,
  `"largest"`, or name.
- **CRS policy: nothing is guessed.** Formats that carry a coordinate system
  (GeoJSON, KML, shapefile) require `target_crs`, the model CRS. A shapefile
  without `.prj` also needs `source_crs`; an explicit `source_crs` overrides
  the file's. Plain text files are assumed to be in the model CRS already.
- Guards: lon/lat values outside +-180/+-90 under a geographic CRS are
  rejected (a wrong `.prj` is the usual cause); a datum change (e.g. NAD83 to
  WGS84) warns with PROJ's stated accuracy, because without grid-shift files PROJ
  may apply a null shift while stating several metres of uncertainty.
- Shapefile rings are grouped into shells and holes by nesting, not by
  winding order, so files from writers that get the orientation wrong still
  load. KML is parsed with `defusedxml` (DTDs and entities refused) and KML/KMZ
  payloads are capped at 50 MB, including after decompression.
- Not supported yet: shapefile in a `.zip`, KML polygons as a closed `LineString`
  (rejected with a hint), polygon holes in text files.
- Invalid polygons (self-intersecting, too few vertices, zero area) are rejected,
  not repaired.
- `azimuth="auto"` minimises wasted columns (minimum-area rotated rectangle,
  folded into [-45, 45)). Pass a number to align to something else, e.g. a
  fault trend.
- `inside="center"` (default): a column is active if its centre is inside the
  polygon (a centre within 1 micron of the edge counts as inside).
  `inside="fraction"`: active if at least `min_fraction` of its area is inside,
  using exact intersection areas.
- `spec.area_ratio` reports active column area / polygon area: the cost of the
  staircase boundary.
- **Inactive columns are no-flow in a simulator.** A polygon cut does not make an
  open aquifer boundary; that is a stage 9 (boundary condition) job.

## Surfaces and wells (M1)

    from cosma.surfaces import read_surface
    from cosma.wells.io import (read_well_headers, read_deviation, read_las, read_tops, read_core, build_wells)

    top = read_surface("top.irap", xy_unit="m", z_unit="m", positive="down")
    hdr = read_well_headers("heads.csv", xy_unit="m", z_unit="m")      # kb = elevation above MSL
    tr  = read_deviation("dev.csv", hdr, z_unit="m", azimuth_reference="grid")
    tops = read_tops("tops.csv", tr, depth_type="md", z_unit="m")
    # (scripting API; the GUI path is cosma.imports.import_file with a Project, below)

- **Nothing is guessed.** Units are mandatory arguments; plain `ft` is refused (`ft_intl`
  or `ft_us`: 2 ppm apart, ~12 ft at State Plane coordinates). Tops/core say whether depth is
  `md`, `tvd` (below KB) or `tvdss`; azimuths say `grid` or `true` (with convergence);
  porosity says `frac` or `percent`; a well with no deviation survey must be declared with
  `Trajectory.vertical`. Internally: metres, depth positive down below MSL.
- Surfaces: IRAP ASCII/binary and ZMAP+ ASCII via xtgeo; rotation-aware bilinear sampling,
  NaN outside and next to undefined nodes. Grids with a reversed y axis (yflip -1) are
  refused. A surface whose values are all <= 0 with `positive="down"` warns (elevations?).
- Trajectories: minimum curvature, exact on circular arcs, interpolated along the arc.
  Depth to MD conversion is refused for horizontal or hooked wells.
- LAS 1.2/2.0 via lasio; depth must be MD; feet need an explicit unit.
- Not yet: scattered XYZ surfaces (gridding), CPS-3, fault sticks, variograms, LAS 3.0,
  multi-well LAS, QC reports.

## Import confirmation

Every importer is reached through `cosma.imports.import_file(kind, path, confirm, **context)`:
the file is inspected (no unit interpretation), the user confirms a dialog that lists what
was read from the file and the parameters to be used, and only then does the real reader
run. Kinds: `surface`, `polygon`, `well_headers`, `deviation`, `tops`, `core`, `las`.

- A parameter the file cannot state (units, depth type, porosity unit, CRS when absent)
  starts empty and **Import stays disabled until it is chosen**. Values taken from the file
  are tagged "from file", documented defaults "default - please check".
- `cosma.gui.import_dialog.confirm_with_dialog` is the Qt dialog (`pip install -e .[gui]`);
  scripts and tests use `cosma.imports.scripted({...})`, which goes through the same validation.
- **Well data is connected to wells in the dialog.** For headers, deviation, tops, core and
  LAS, each well named in the file gets a row: connect it to an existing well of the
  `Project`, or import it as a new well (name editable). Defaults: exact name -> that well;
  same name ignoring case/punctuation (`W1` ~ `W-1`) -> that well, marked "check"; otherwise
  new. Import stays disabled if a new name is taken, a name is empty, or two file wells
  would land on one well.
- **Data the well already has is flagged and never overwritten silently.** Once units and links
  are chosen, the dialog lists each collision (same top name; a log curve with the same
  mnemonic over an overlapping MD range; a core column overlapping in depth; an existing header
  or deviation survey) with what is there, what the file has, and whether they are identical.
  Each needs an explicit choice: **replace**, **skip** (keep existing) or **import with suffix**
  (not offered for header/survey; the suffix must not collide with an existing name). There is
  an "apply to all" shortcut. Replacing a curve/column removes the old one entirely, including
  its non-overlapping part. Log/core files are kept as separate sets; importing the same
  file name again never overwrites a set (`a`, `a#2`, ...).
- `cosma.project.Project` stores what was *received* (header, survey, picks as depth + type,
  core, logs per LAS file) and derives trajectories and pick positions on demand, so a
  corrected survey never leaves stale coordinates, and data may arrive in any order (picks
  for a well without header/survey are kept and flagged "unplaced"). Every import is atomic.
- The low-level readers (`read_surface`, `read_las`, ...) remain callable directly for
  scripting; the GUI must only call them through `import_file`.

## Project, type well and stratigraphic chart

- **New project** (`cosma.gui.new_project.new_project_workflow`) asks whether there is a type
  well. There is no default answer. If yes, that well is imported first (header, deviation
  survey - skippable, tops, each through the usual confirmation dialog), the **stratigraphic
  chart is built from its tops** (ordered by depth; the deepest top is the base of the lowest
  interval) and shown in the chart editor. If no, the chart is defined later: by hand
  (`Project.create_chart`) or from any well (`Project.build_chart(well)`).
- **Chart editor**: rename, reorder, add and remove intervals; set the contact at the top
  (conformable / unconformity). An interval is named after the pick at its top; renaming it
  renames that pick in every well and keeps the old spelling as an alias, so later imports
  using the old name still map. Removal is refused while data hangs on the interval; the whole
  edit applies or none of it.
- **Horizons are the attachment points.** "Base of A" and "Top of B" are one horizon: a depth
  grid attached once serves both, and a second one is a conflict (replace / skip / keep as
  alternative with a suffix; the first stays the one the model uses). Horizons are identified
  by id, so renaming or reordering never detaches data.
- **Imports after the chart exists**:
  - a **surface** must be attached to a horizon (surfaces cannot be imported before a chart
    exists). The dialog suggests where it sits from the file name (`Top_Seal` = top of Seal,
    `Base_Seal` = base of Seal = top of the next interval) and from depth (the surface sampled
    at wells that have a pick for the horizon; median difference shown). A suggestion is
    preselected only when it is clear (name match without contradicting depths, or a single
    horizon that fits); a name that points at a horizon the depths contradict gives a warning
    ("wrong horizon, datum or units?") and no preselection. The choice is always the user's.
  - **tops** of other wells: every pick name in the file is matched to a chart horizon (or
    ignored); unknown names must be mapped explicitly and are remembered as aliases. Picks
    out of the chart's order are warned about, not blocked (faults repeat or omit section).
  - Logs, core, headers and surveys span several intervals and are not attached to one;
    which interval a sample falls in follows from the tops. Variograms and per-layer maps will
    attach to intervals through the same mechanism (not built yet).

## Variograms

A variogram belongs to one **stratigraphic interval** and one **property** (e.g. `PHI` in `Res`); several can be kept
per pair, one of them the *primary* that the property step will use (same replace / skip / suffix rules as surfaces).

* **Model** (`cosma.variogram`): nugget plus up to three spherical / exponential / gaussian structures, GSLIB
  conventions - ranges are *practical* ranges, azimuth is the major axis in degrees clockwise from north, anisotropy
  is layer-parallel (no dip or plunge; files that ask for it are refused, not approximated).
* **Experimental** (`experimental_variogram`): Matheron semivariance. *Vertical* is computed inside each well and
  pooled (pairs never straddle two wells). *Major / minor* use only pairs from **different wells in the same layer**
  (`|dz| <=` the layer thickness you give), within the angular tolerance and optional bandwidth - pairs at different
  depths measure vertical, not horizontal, continuity. Empty lag bins are NaN, never 0.
* **Data** (`cosma.variogram_data.collect`): logs or core of one interval, positions from the deviation survey, depth
  as *depth below the interval top* (`top`) or stretched to the mean thickness (`proportional`). Wells missing a pick,
  trajectory or the property are listed with the reason; a property present in several sources of a well must have its
  source named.
* **Editor** (`gui/variogram_editor.py`, logic in `variogram_session.py`): experimental points sized by pair count and
  labelled with it, model curve over them, manual entry of every parameter, *Suggest azimuth* (direction of lowest
  semivariance; no suggestion when there is no clear anisotropy), *Auto-fit* (vertical gives nugget and sill, the
  horizontal ranges are then fitted with those held; a direction without enough pairs is left **unset**, not guessed).
  `model()` refuses to build while any range is unset.
* **Import** (`import_file("variogram", path, confirm, project=...)`): GSLIB parameter block or COSMA JSON; the
  confirmation dialog asks for the property, interval, range unit and whether the variogram was computed on raw or
  normal-score data. GSLIB cannot state these, so they start empty.
* **Facies (indicator) variograms.** A whole-number curve (facies code) can be turned into the 0/1 indicator of one code;
  the variogram is stored under the property name `CURVE[code]` (e.g. `FACIES[2]`), transform `indicator`, and its
  sill should be about p(1-p) (the editor warns if it is not, or if the facies is rare: < 30 samples, < 2 % or > 98 %).
  One variogram per facies; no transform can be applied to an indicator. Order-relation corrections between facies
  are not done.
* **Main window** (`python -m cosma.gui`, or the `cosma` launcher): project tree (wells; intervals with their surfaces
  and variograms), **Import** menu (surface, headers, deviation, tops, LAS, core, variogram), **Stratigraphy > Edit
  chart**, and **Variograms > New / Edit selected / Export selected** (double-click a variogram to edit it). A saved
  variogram remembers how it was made (curve, source, frame, facies code), so *Edit* reopens it with the same data
  and the saved model loaded; a variogram imported from a file asks for the data again. Export writes GSLIB or JSON.
* **Honest limits.** Horizontal variograms need several wells; with few wells they are poorly constrained (the editor
  warns) and the ranges should come from analogues. Vertical ranges from logs are at log resolution, not grid-layer
  resolution. Variograms are not yet saved to disk with the project (the project is in memory only).

## Per-interval property maps

A map is one 2D grid of a property for one stratigraphic interval, stored like surfaces and variograms (one *primary*
per interval and property, alternatives kept; replace / skip / suffix on conflict). It records where it came from:

* **imported** - from a surface file (IRAP/ZMap); the horizontal unit and the unit of the values are chosen, never
  assumed (`fraction`, `percent` - converted to a fraction -, `mD`, `m`, `other`). The dialog warns if a map declared
  as a fraction has values above 1, or one declared as percent has none above 1.
* **thickness** - base surface minus top surface (primary surfaces of the interval's two horizons); nodes where the
  surfaces cross are left undefined and counted.
* **kriged** - ordinary kriging of **layer averages** with the interval's variogram.

*Layer average* of a well = the property averaged between the interval's two picks, each sample weighted by the
vertical thickness it represents (a gap in the log is not bridged); for a facies it is the facies **proportion**. A
well covering too little of the interval (default 50 %, your choice) is skipped and listed; below 80 % it is flagged.

*Kriging* (`maps.krige`) uses the horizontal part of the variogram (nugget, structures, major/minor range, azimuth) and
all wells (global neighbourhood, up to 3000). It is exact at a well (variance 0 there, even with a nugget), the weights
sum to 1, and the kriging variance is kept with the map. Two wells at the same location are refused. It was checked
against an independent covariance-form solution, and on a simulated field leave-one-out kriging beat predicting the
mean (RMSE 0.366 against 0.402 - a modest gain, as expected for one noisy realization at 150 m spacing).

* **Use a variogram of layer averages** (Variograms > New > *Layer averages*): same support as the map. Such a variogram
  is stored as `NAME (layer average)`, horizontal only (its vertical range is set to the mean thickness and is not
  used). A variogram computed from raw log samples can still be chosen, but the dialog says why the map will then be
  smoother than it should be and its kriging variance on the wrong scale.
* **Options.** Grid: taken from the top or base surface of the interval, or around the wells with an increment and
  margin you type. Optional mask to the interval (needs both surfaces) and optional clipping to a physical range (the
  number of clipped nodes is reported). Unit is required.
* **Window.** Maps menu: compute (with preview of estimate and variance, wells marked), view, export as IRAP; the
  Import menu reads maps from files; maps show in the project tree.
* **Honest limits.** With one well the map is that value everywhere; with a handful of wells it is a smooth trend, not
  a prediction (the dialog says so). No trend / external drift or collocated co-kriging yet; no local neighbourhood
  search (not needed at these well counts); log-normal or other transforms are not applied to permeability maps.

## Fault sticks

A fault stick is a 3D polyline digitised along a fault on one seismic section; a fault is a set of sticks. Import:
Import > Fault sticks. Faults are stored in the project (`project.faults`, by name) and need no stratigraphic chart.

* **Files.** A delimited table (comma, semicolon, tab or spaces) with a header row: `x`, `y`, a depth column, and
  optionally `fault` and `stick` (common aliases are recognised; other columns are listed and ignored). Or a headerless
  `x y z` file whose sticks are separated by blank lines or `999 999 999` lines (one fault, named by you). I could not
  check this against real Petrel / Kingdom / CPS exports, so the first real file may need a header added.
* **Nothing is guessed.** The horizontal unit, the vertical unit and *depth (positive down)* versus *elevation
  (positive up)* must be chosen. Time-domain sticks are not accepted - depth-convert them first (a column called
  `twt`/`time` is flagged). Which points form a stick comes from the `stick` column or the separators; with neither,
  each fault is read as ONE polyline and the preview says so.
* **Dip.** The plane is fitted to all points (orthogonal regression): dip, dip direction, strike line, RMS misfit. With
  **one stick the dip is reported as undetermined**, never as a number: a stick lies in a seismic section, so the plane
  through it is the section, not the fault. Nearly collinear points are flagged as poorly constrained.
* **Vertical faults in v1.** COSMA v1 treats faults as vertical, so the grid builder needs the fault position at ONE
  depth. `trace_at(depth)` intersects every stick with that depth and orders the points along the fault. A stick that
  does not reach the depth, runs along it, or crosses it more than once is skipped and the reason is listed; a trace that
  folds back is flagged. The import warns when the dip is below 80 deg and says how far the fault moves sideways over its
  own depth range, and when the sticks lie outside the project's surfaces (units / CRS / sign mistakes).
* **Window.** A Faults node in the project tree; Faults > View (plan view of the sticks, plane fit, trace at a depth you
  type, skipped sticks in amber), Export all sticks as CSV (x, y, depth in metres, fault, stick), Remove.
* **Faults and property continuity.** Faults never act as barriers to properties. Every sample is positioned by its depth
  below its own well's top pick, wells are compared layer by layer, and kriging distances are plan-view, so data on both
  sides of a fault are used together, as for layers deposited before the faulting. This is checked, not assumed
  (`tests/test_fault_continuity.py`): moving every pick and log sample east of a fault down by a throw leaves the
  samples, layer averages, experimental variograms, kriged maps and variances unchanged to 1e-9, a step in the bounding
  surfaces leaves the thickness map flat and the interval mask intact, and the map shows no jump at the fault.
  A control shows the experiment would notice an implementation based on absolute depth.
  **What can still go wrong is a well that cuts the fault inside an interval:** the section is then cut out or repeated,
  so its picks no longer match the layers of the other wells. Such wells are listed with the crossing depth in the map
  and variogram warnings (and `fault_crossings` in the map record). COSMA reports them and does not drop them. Crossings
  follow the fault's trace at each depth, so dip and curvature are respected; past the fault tip there is no crossing.
* **The assumption behind this** is that the porosity and permeability of a layer were fixed before the faulting.
  Not true for syn-depositional (growth) faults, where the layer thickens across the fault, nor where compaction or
  cementation depends on burial depth and the fault displaced a layer into a different burial history, nor in the fault
  zone itself (cataclasis, cementation). COSMA does not model any of this yet; it also has no switch to turn the
  assumption off, and the surfaces are not gridded with the faults honoured.
* **Not done:** no fault throw, no transmissibility, no fault-conforming pillars, no listric/curved surfaces, no Petrel
  fault-polygon or `.flt` formats, and the trace is not yet used by any grid step (that is M3).

## Grid geometry (M3, first slice)

`cosma.grid.builder.build_grid(spec, horizons, contacts, layerings, min_thickness=0.1)` turns pillar-sampled
horizons (`sample_horizons` samples surfaces at the pillars) into a layered corner-point grid with vertical pillars.
Nothing is defaulted: every interval needs a `Layering` and a contact type.

- Layering per interval: `fixed_count` (n proportional layers), `fixed_thickness` (about dz, count set by the thickest
  column, thinner columns use fewer equal layers), `follow_top` / `follow_base` (layers of exactly dz from the top or
  base; the rest collapse at the opposite end).
- Contacts (applied after any throw): `erosional` (older horizons cannot rise above the surface), `onlap` (younger
  horizons cannot sink below it), `conformable` (nothing clipped; crossings are repaired to zero thickness and reported).
- Cells thinner than the minimum thickness (default 0.1 m) become inactive; the report gives their count and the
  geometric volume lost. Columns touching an undefined horizon are inactive.
- Horizons may be given per pillar or per cell corner `(ny, 2, nx, 2)`; per-corner lets cells on either side of a
  fault have different depths at their shared pillar.
- Not yet: fault intersections (interaction tool), buffer/aquifer boundaries, properties, project and GUI wiring.

### Defining, building and exporting the grid (Grid menu)

`Grid > Define, build and export grid...` holds the whole definition in the project (saved with it, format 4):
origin, azimuth, cells and cell sizes, minimum thickness, a layering choice per interval, and erosional or onlap for
every unconformity in the chart. Nothing has a default except the 0.1 m minimum thickness; "What is still missing?"
lists every open item, including fault trace depths and unconfirmed fault interactions, and Build refuses until
the list is empty. "Fit origin and cell counts to the top surface" proposes an extent (rounded up so the surface is
covered); "Suggest orientations" lists the azimuth along each fault's strike, along the mean fault strike and with the I
axis down the structural dip, each with the cells needed to cover the top surface compared with a north-aligned grid,
so the price of rotating is visible. Double-click one to use it. The build report lists layers per interval, thin cells
made inactive and the volume lost, unconformity clipping, per-fault faces, throw and snap distances, and fault ages;
Export writes the GRDECL (geometry only, properties come with the property milestone). Not yet: graded spacing.

### Model boundary (`cosma.grid.boundary_options`)

Import the boundary polygon (Import > Model boundary; confirmation popup, CRS never guessed) and tick any combination of
treatments in the Grid dialog. None ticked is a closed box: columns outside the polygon are inactive.
- **Inside rule**: a column is inside when its centre is, or when at least a chosen fraction of it is.
- **Extended buffer zone**: columns whose centre lies within a distance of the polygon are active too; exported as
  `FIPNUM` (1 = inside the polygon, 2 = buffer). A warning appears if the grid is too small for the distance.
- **Pore-volume multipliers**: a factor on the outer N rings of active columns, written as `MULTPV` to
  `<name>_grid_boundary.inc` (GRID section).
- **Aquifer**: Fetkovich (`AQUFETP`) or Carter-Tracy (`AQUCT`) with `AQUANCON` on the outer faces you tick (grid index
  directions I-, I+, J-, J+), in `<name>_solution_boundary.inc` (SOLUTION), plus the `AQUDIMS` that OPM demands in
  `<name>_runspec_boundary.inc` (RUNSPEC). Every parameter is yours; blank initial pressure means "equilibrate".
Deck text is OPM Flow / Eclipse syntax in METRIC units. Pore-volume multipliers together with an aquifer, or multipliers
with a buffer, are allowed but flagged. Validated in the real engine (`tests/test_opm_flow.py`): each treatment loads,
runs a CO2STORE case, and lowers the pressure build-up relative to a closed box. Caveat: an aquifer's datum depth
and pressure must be consistent with the deck's EQUIL, otherwise a small mismatch can dominate a small injection.
The buffer and multiplier columns carry no properties yet; the property step must populate them.

### Fault ages and interactions (`cosma.fault_interactions`, Faults menu)

Each fault needs an explicit trace depth (the dialog proposes mid-stick, which counts only when you press Find) and
the project holds a reach distance (how close a fault end must come to another fault to count as meeting it).
COSMA then finds every pair of traces that cross or where one ends within reach of the other. A fault that ends on
or near another is *suggested* to be the younger; for a crossing, geometry cannot say and no suggestion is made. You
answer each: either fault older, same age, or unrelated. The tool opens by itself after a fault import whenever two
or more faults are not yet settled. `Project.fault_age_order()` refuses until every fault has a depth and every
interaction an answer, and refuses contradictory answers. Answers survive re-checking only if the same pair still meets
in the same way near the same place. Project format is now 3 (v1 and v2 files still load).
Where the reach of two faulted faults overlaps, `corner_depths(..., age_order=...)` lets the younger fault claim the
corner. That is approximate: surfaces carry both steps, a corner in a block between two faults is sampled from the
younger fault's far side only, so depths there can be off by the older fault's throw (the report counts the corners).

### Faults in the geometry (`cosma.grid.fault_sampling`)

Vertical faults only, each taken at one explicit trace depth. Throw is never invented: it is the depth step the
imported surfaces show across the trace, measured by sampling each side at 1.5 and 3 node spacings from the trace and
extrapolating to it (so the surface's smeared step is not sampled). A fault is "negligible" when the median
|throw| / thickness of the thinner adjacent interval is below 10 % at every horizon; it then stays out of the geometry
(transmissibility only) and the report says so.

Snapping: a cell takes the side of the trace its centre is on, and faces between cells of opposite side are the fault
faces (nearest face, no tolerance, tips follow). Each cell corner takes the depth of its own cell's side, extrapolated
from the surface on that side. The report gives the number of faces and the largest distance between a face and the
true trace. Corners within reach of two faulted faults are flagged for the (not yet built) interaction tool.
Assumption: surface steps are where the stick trace is. If the fault sticks and the surfaces disagree, throw is
measured against the wrong line.

Validation: `tests/test_opm_flow.py` writes a built grid (dipping top, erosional contact, pinched cells) and has a real
OPM Flow engine (`pip install opm-simulators`) load it: OPM's cell volumes equal COSMA's, and a CO2STORE run completes.

### Fault transmissibility (`cosma.fault_transmissibility`, Faults menu)

Every fault needs an explicit choice before a grid can be built; nothing is preselected.

- **open**: no multiplier (1), stated by the user, not assumed.
- **fixed**: one multiplier m >= 0 on every connection of the fault (0 = sealing). Allowed for any fault.
- **calculated** (only for faults with visible throw; a negligible-throw fault is refused at build): method `slab`. The
  fault is a slab of thickness t and permeability kf between the two cells, in series with the matrix:
  `TM = 1 / (1 + t (2/kf - 1/ki - 1/kj) / (Li/ki + Lj/kj))`. kf is given by the user; t is given, or is a user ratio
  times the fault's median |throw| over the horizons. The multiplier is derived from series resistance (tested
  against a direct resistance calculation) and is the same quantity Manzocchi et al. 1999 express with fault thickness and permeability (their equation
  number was not checked). One multiplier is
  written per fault: the mean over its cell pairs weighted by their transmissibility without the fault.

Export writes `<name>_faults.inc` (`FAULTS` face records over all layers, then `MULTFLT`), to be included in the GRID
section after the grid. OPM applies the multiplier to juxtaposed connections across the throw as well as neighbours.
Fault names longer than 8 characters, or clashing once upper-cased, are written as FLT001... with a comment.
Calculated faults need cell properties that exist only after the property step. Until then the grid geometry still
exports; the fault include is **not** written (so no fault has a MULTFLT yet) and the export notes say so and which
properties are missing. Once the properties exist the same export writes the include.

- **calculated, method `sgr`** (SGR from VSHALE): the same slab, but kf comes from the shale gouge ratio of each cell
  pair through a relation `log10 kf = f(SGR, D)` (kf in mD, D = the face's throw in m; `cosma.fault_relations`).
  COSMA has **one built-in relation**, read from its source paper: `manzocchi1999` (Manzocchi, Walsh, Nell and
  Yielding 1999, Petroleum Geoscience 5, 53-63, Eq. 1: `-4*SGR - 0.25*log10(D)*(1 - SGR)**5`). It reproduces the
  paper's worked example (D = 20 m, SGR = 0.2: 0.12 mD median of a log-normal distribution spanning about two orders
  of magnitude); the authors say it is not reliable as SGR tends to 0. **Your own methods** are added in the dialog and
  stored in the project: a name, an expression, and a *required* source (it is your responsibility), plus an optional
  note. Expressions use numbers, `SGR`, `D`, `+ - * / **`, parentheses and `log10 ln exp sqrt abs min max`, evaluated by
  a small closed evaluator (no Python `eval`); an expression must be finite over a probe grid (SGR 0.05-0.95, D 1-100 m)
  or it is refused. *Test* shows kf over SGR and throw before you add it. Methods can be exported to, and imported from,
  a JSON file to move them between projects. A fault's choice keeps its **own copy** of the expression: editing or
  removing a stored method never changes a saved choice (the dialog shows such a choice as "saved copy").
  SGR (`cosma.fault_sgr`) needs the cell property **VSHALE** (fraction 0-1; percent is refused, never rescaled). For a
  face between two cells at layer k: throw T = difference of the two cell-centre depths at that layer; the slipped
  interval is the T of stratigraphy above the cell centre (the last layer included in part); layer thickness and VSHALE
  are the mean of the two columns; SGR = shale thickness in that window / T. Face-layers with T below the user's
  **minimum throw**, or whose window reaches above the model top, are left out of the weighted mean and counted in the
  export comments (with SGR min / weighted mean / max and the kf range). The slab thickness is a user ratio times the
  face's throw, or a constant.
- **VSHALE notice:** when faults are present and no well has a curve named `VSHALE` (any case; COSMA does not guess
  which curve is shale volume), a message says VSHALE is needed to calculate SGR; shown once per session after a
  fault or LAS import, and always in the fault transmissibility dialog. Open and fixed faults do not need it.
- The fault transmissibility dialog opens after a fault import until every fault has a choice.

Known limits: cell pairs are taken at equal layer index (exact for uniform permeability, an approximation when
permeability varies strongly across a throw); one multiplier per fault, not per face; the SGR window uses the mean of the
two walls' VSHALE (one stratigraphy per face) and the grid's own centre-depth throw; SGR is untested against a
commercial package; the SGR path is exercised only with synthetic VSHALE until the property step exists.

## Facies from logs (`cosma.facies`, Facies menu)

The hard data for facies simulation. Everything is explicit; a sample no rule matches stays undefined.

- **Scheme:** facies with whole-number codes >= 1 and unique names (code 0 means "no facies").
- **From logs, one of:** ordered *rules* (each: a facies and one or more conditions `curve < <= > >= value`, all of
  which must hold; the first matching rule wins), or a *code map* for a curve that already holds facies codes (a code
  with no mapping stays undefined and is counted in the report). Both use one log source per well; a source is named
  only where a well has several.
- **Flags:** depth intervals of a well (MD, TVD below KB or TVDSS) that are a given facies; they override the log. A
  well with flags but no log is sampled every `flag_step` metres.
- **To cells:** each log sample is located in the grid (bilinear cell tops and bottoms, rotated frames, deviated wells,
  inactive and collapsed cells skipped) and each cell takes the facies with the greatest *length of well path* in it,
  provided the classified length is at least `min_coverage` of the path in the cell. An exact tie leaves the cell
  undefined. Wells that disagree in one cell are a conflict, resolved only by the policy the user chose (`longest`
  path wins, or `drop`: no hard datum). `min_coverage` and the policy are saved with the setup and have no default.
- **Dialog:** Preview shows the log classification per well (share of well length per facies, undefined share,
  unmapped codes, skipped wells and why) and, when a grid can be built and the upscaling choices are made, the hard
  cells per facies. Save is possible only for entries that have been previewed; any later edit switches it off.
- Known limits: cell lookup needs vertical pillars (as COSMA builds them); facies are taken per cell, so thin beds
  below cell size are lost to the majority rule; nothing here simulates facies yet (that is the next step).

## Data checks (`cosma.datacheck`, Facies menu > Data checks...)

Answers "is there enough data to simulate from?" before any simulation. Every threshold is typed by the user; there are no
defaults, and the report only *suggests* a fallback - it never applies one.

* **Inventory:** hard cells, wells and well-path length per interval and facies, against minimum hard cells / wells.
* **Vertical variogram support:** per interval and facies, experimental vertical variogram from the wells' log samples
  (pairs within one well); the longest lag at which every bin holds the minimum number of pairs must reach the needed lag.
* **Horizontal variogram support:** cell-scale points from different wells in the same layer (within a vertical tolerance),
  all directions pooled; needs the minimum number of wells and pairs. One or two wells say so plainly.
* **Distributions at cell scale:** each chosen log curve is averaged into cells (arithmetic, geometric or harmonic, chosen per
  property), then described (mean, std, P10/50/90) next to the log-scale spread (variance ratio shows the support effect) and
  normal / lognormal fits with a Kolmogorov-Smirnov *distance* (not a p-value: samples are spatially correlated).
* **Porosity-permeability from core:** log10 k on phi at three levels - facies x interval, facies over all intervals, interval
  over all facies - each with pairs, slope +/- CI, R2 and scatter, passing only if it meets the minimum pairs (and R2 if set).
  The report lists, per interval and facies, which level would work first; if none does it says to enter the relation by hand.
  A core sample takes the facies of the nearest classified log sample within a tolerance.
* The dialog runs the checks on the current entries; Save is possible only after a run of exactly those entries.

Limits: the vertical check inherits the variogram tool's angular limit for deviated wells; horizontal points are cell-scale
and pooled over directions; the KS distance is not a goodness-of-fit test; facies per cell is the majority rule from
`cosma.facies`.

## Saving and loading projects

A project is one `.cosma` file (a zip): `manifest.json` (format, format version, SHA-256 of the other members),
`project.json` (names, settings, chart, variogram models) and `arrays.npz` (logs, surveys, core, surface grids, fault sticks).

* **No pickle.** Opening a project cannot run code; the file is readable with any zip tool and numpy. Object arrays are
  refused on save and on load.
* **Exact.** Floats, NaNs, curve order and the chart's id counter round-trip identically; trajectories and pick
  positions are *derived* after loading, as they always are, so nothing stale can be stored.
* **Safe writes.** The file is written to a temporary name beside the target and renamed; a failure leaves the previous
  file untouched. (The file is fsynced before the rename; that is not covered by a test.)
* **Strict loading.** A file from a newer COSMA is refused with a message, never guessed at. Checksums are verified, and
  every cross-reference is checked (surfaces and variograms must belong to chart horizons, primaries must exist, the
  type well must exist ...); a project is returned whole or not at all.
* **Window.** File > Open / Save / Save as (Ctrl+S); the title shows the file and `*` when there are unsaved changes
  (measured by content, not by tracking edits). New, Open and Quit ask Save / Discard / Cancel when work would be lost;
  choosing Save and then cancelling the file dialog counts as Cancel. `python -m cosma.gui project.cosma` opens a file.
* **Not stored yet:** the model boundary polygon (the project has no place for it yet) and, later, grids and exports.
  There is no autosave. Format version 2 added faults; version-1 files still load (there is nothing to migrate), and an older COSMA refuses a version-2 file instead of silently dropping the faults.

## Tests

    pip install -e .[validate,test]
    pytest

Readers used for validation: xtgeo (geometry, properties, MAPAXES), resdata
(geometry; float32 limited), and the OPM deck parser (syntax and numbers only,
no OPM grid is built). No simulator has been run.

## Vertical proportion curves (Facies menu)

A vertical proportion curve (VPC) gives the share of each facies at each position inside an interval. Nothing is
preselected: position mode, bin size or count, smoothing window, minimum wells per interval and minimum hard cells in every
bin's smoothing window are typed by you, and every interval gets an explicit source.

* **Position**: relative (0 at the top of the interval, 1 at its base) or true depth from the top (bin size in m).
* **From the wells**: proportions of the hard-facies cells, weighted by cell thickness, moving-average smoothed (odd window,
  truncated at the ends). An interval with too little data stops the build with the reason; nothing is borrowed from
  another interval.
* **My curve**: typed bin by facies table. Rows must sum to 1 (never rescaled).
* **Checks**: hard cells whose facies has zero proportion at their position are reported as conflicts; the interval mean from
  the curve is compared with the mean from the hard cells.
* Save is possible only after Compute on exactly the entries being saved.

Not yet: using curves in the facies simulation.

### Proportion maps from images (Facies menu > Import proportion map...)

Import a PNG / JPG / TIFF / GeoTIFF / screenshot of a facies map for one interval. Processing order: crop, rotate, flip,
scale (nearest neighbour, colours stay exact), mask polygon; then georeference with >= 3 control points (affine fit; the import
is refused when a residual exceeds the tolerance you type; with exactly 3 points the fit is exact and says nothing about the
points); then the legend.

* **Legend**: each colour (click it on the image or type R G B) is tied to a facies that already exists in the project. Several
  colours may map to one facies; two colours of different facies closer than twice the tolerance are refused. Pixels matching
  no colour are no data and the most common unmatched colours are listed.
* **Every facies needs a colour or an explicit "absent from the map"** (fraction 0 wherever the map has data).
* **Smoothing**: Gaussian on the class indicators, normalised by the smoothed valid mask; the radius is suggested as half the
  smallest model column in pixels and can be overwritten.
* **Checks**: fractions within 0-1 and summing to 1; the map is resampled (bilinear) to the active model columns and refused when
  it covers less than the share you typed.
* Save asks for confirmation and stores the resampled fractions and the settings, not the image.
* `proportions.combine_with_maps` multiplies a curve by a map, divides by the interval's global curve proportions and
  renormalises. Columns without map data (`curve` = keep the curve, or `stop`) and the largest share of cells where curve and map
  leave nothing are arguments, not defaults. It is an API function; it has no dialog yet.

## Facies simulation (`cosma.sis`, M4.4 engine)

Sequential indicator simulation per interval. Inputs: the proportions per cell (`proportions.layer_proportions`, or
`combine_with_maps` when a proportion map exists), the hard facies, one **indicator variogram per facies and interval**
(attach it to the interval in the variogram tool under the facies name, transform `indicator`; a missing one stops the run, nothing is
borrowed from another interval) and a seed.

* Each node: simple kriging of the indicator residuals around the local proportion, probabilities clipped to [0, 1] and
  renormalised (counted in the report), a facies with local proportion 0 stays at 0, the facies is drawn. Hard cells are exact.
* `SisSettings(max_neighbours, search_factor, levels)`: no defaults. Multigrid `levels` simulate coarse lattices first.
* Same seed = same realization (same numba build); the seed is per run, the interval is mixed in.
* **Known limits (tested, not fixed)**: with strongly continuous indicator variograms the kriging weights sum to more than 1,
  estimates leave [0, 1] in a large share of nodes, and clipping (a) raises the minority facies proportion (about +10% relative at
  p = 0.3, none at p = 0.5) and (b) weakens short-lag indicator correlation (about 85% of the model at lag 1-2, long lags are
  right). A realization is not forced to its target; the realization manager (M4.7) compares each with the target.
  Hard data far from a coarse-grid node are only seen at the finest level. Speed ~100k cells/s per core with 12 neighbours
  (about 100 s for 10 million cells), single-threaded. No GUI yet; no persistence of the settings yet.

### Truncated Gaussian simulation (`cosma.tgs`): the alternative to SIS

SIS stays the default. Truncated Gaussian simulation (TGS) suits **ordered** facies (for example shoreface sand < silt < mud):
one Gaussian field is simulated (sequential Gaussian simulation) and cut at thresholds from the local cumulative proportions along
the order you give. Compared with SIS:

* proportions are reproduced without SIS's clipping bias (tested at p = 0.5 and 0.3, within 0.03);
* the indicator correlation matches the bivariate-Gaussian theory (tested against `scipy` at four lags, within 0.07), so the
  short-lag shortfall of SIS does not occur;
* but **one** Gaussian variogram (attach it to each interval with transform `normal_score` under the name you pass as
  `variogram`) controls every facies; facies can only touch their neighbours in the order; and the order must be given
  (`TgsSettings(order, variogram, max_neighbours, search_factor, levels, gibbs_sweeps)`, no defaults);
* hard facies: the Gaussian values at hard cells are drawn inside their threshold interval by Gibbs sampling (`gibbs_sweeps`),
  the field is conditioned on them and the hard facies are written back exactly, also when their local proportion is 0.

### Simulate facies dialog (Facies menu > Simulate facies...)

Needs wells, the facies, saved vertical proportion curves and a grid. Choose SIS (preselected) or truncated Gaussian, enter neighbours,
search radius, multigrid levels and seed (all typed). Truncated Gaussian also needs the facies order (confirm it with the checkbox),
the name of the Gaussian variogram on the intervals and the Gibbs sweeps. If proportion maps exist you must choose what to do where
a map has no data and the largest share of cells curve and map may leave empty. **Simulate** runs one realization and shows a layer
and a section plus the report; **Save settings** is possible only after a simulation with exactly those entries. The realization
itself is not stored (realization manager, M4.7). Project format version 12.

## Property simulation (`cosma.property_sim`, M4.5 engine)

Sequential Gaussian simulation of one cell property (porosity, VSHALE, ...) on a facies realization, **per interval and facies**: only
cells of that facies in that interval are simulated and conditioned, on that domain's own cell-scale hard values
(`hard_property_data`: well samples averaged to cells with the averaging and curve of the data-check property, same coverage rule).

* For every domain you choose the distribution: `'data'` (empirical, needs `min_data` cell values, back transform never leaves the data
  range) or `'user'` (normal or log-normal, mean and standard deviation of the property at cell scale). Hard values that are very
  unlikely under a typed distribution (tail probability below `conflict_probability`) stop the run; so do hard values outside the
  optional bounds. Simulated values are clipped to the bounds.
* Each domain needs a variogram with transform `normal_score` named by your pattern (`"PHI {facies}"`); a missing one stops the run.
* Hard values are exact; same seed = same realization. Several wells in one cell are averaged.
* Tested against known truth: simulated mean/standard deviation of normal and log-normal domains, normal-score correlation vs the
  spherical model at four lags, data distribution (KS), conditioning pull around a hard value.
* Not in this step: trends, correlation between properties (phi-perm with scatter, kV/kH are M4.6), boundary crossing between
  intervals, any dialog or project storage of the setup.

## Permeability from porosity (`cosma.perm_sim`, M4.6 engine)

`log10 k = a + b * phi + scatter` per interval and facies, on the simulated porosity of a realization; then PERMY = PERMX x kY/kX and
PERMZ = PERMX x kV/kH (kV/kH typed per interval and facies).

* The relation of each domain is typed (`a`, `b`, scatter = standard deviation of log10 k) or taken from the core regression at the level
  you name (`facies x interval`, `facies, all intervals`, `interval, all facies`). A core level that did not pass the data-check
  thresholds (pairs, r2) stops the run with the reason; the data checks suggest a level, you choose.
* A core-to-reservoir shift (log10 units) is added to core intercepts; you type it (0 = none).
* Porosity outside the porosity range of the core pairs: `allow` (extrapolate), `clamp` or `stop`; the cell count is reported.
* Optional bounds clip PERMX. The scatter is independent per cell or, by choice, spatially correlated (see below).
* Not honoured: hard permeability data. Not yet: dialog, project storage of the setup.

## Realizations (up to 100)

`cosma.realizations` makes realizations of facies, properties and permeability from seeds and checks each against its target.

* **Reproducible.** A realization is fixed by the plan, the base seed, its number and its attempt; every stage's seed is derived with
  numpy's `SeedSequence`, so the order of generation does not matter. Arrays are stored as float32, and permeability is computed from
  the float32 porosity, so a regenerated realization equals a cached one. Regeneration is guaranteed only within the same
  `GENERATOR_VERSION`. The optional cache (compressed `.npz`, one file per realization) wins over regeneration when the versions
  differ, and the difference is reported; a cache made from a different plan is refused.
* **Target check.** Per property, interval and facies: mean, variance and shape (Kolmogorov-Smirnov) of the normal scores against
  N(0,1); per interval and facies: the fraction against the mean local target proportion. Because neighbouring cells are correlated,
  the tests use an effective sample size `min(n, V/A)` from the integral volume of the variogram, and the significance
  `(1 - confidence)` is split over all tests of the realization (Bonferroni). Measured on a 30x30 grid: 0 of 40 correct realizations
  failed, and a target shifted by 0.3 sigma was caught in 10 of 10. Limits: the effective size ignores the domain's shape and the
  conditioning to wells; with TGS the facies fractions use the Gaussian variogram (lenient); a clustered well data set used as
  the 'data' target is not the true field distribution, so a high failure share points at the target, not at the simulation.
  Domains with fewer than 5 effective samples are reported as not testable.
* **Failures.** `keep` (flagged), `discard` (no arrays), or `redraw` up to a user-set number of times (facies and properties, or
  properties only); when redraws run out the last attempt is kept and flagged. A flagged realization is exported only with an explicit
  `allow_flagged`.
* **Export (one realization at a time).** `RealizationManager.export(n, path)` writes the GRDECL with the exported properties
  (PORO, ...) and PERMX/PERMY/PERMZ. Properties marked `export=False` (for example `VSHALE`) are not written but are given to the
  fault transmissibility, so the SGR fault multipliers are computed from that realization's VSHALE. Cells without a facies are written as 0.
  The plan (`RealizationPlan`) is saved in the project (format 13). There is no dialog yet.

### Properties, permeability and realizations dialog (Facies menu)

Three tabs: properties (name, keyword, variogram name pattern, bounds, search; per interval and facies the source of the distribution:
data or typed normal/log-normal), permeability (typed or core relation per interval and facies, kv/kh, ky/kx, core shift, behaviour
outside the core porosity range), realizations (count, base seed, confidence, failure handling, optional cache folder). Everything
starts unset. "Check setup" makes realization 0 from exactly the entries and only then can the plan be saved; any edit withdraws the
check. "Run all" makes the realizations (Stop is honoured between realizations), the report lists failed tests, and "Export selected"
writes one realization (flagged ones only with the tick). Tested offscreen only, not on a real display or Windows; a long run blocks
the window between realizations (no worker thread yet).

### Progress, estimated time and Stop (long runs)

Facies simulation, the setup check and "Run all" run in a worker thread, so the window stays usable. A bar shows progress, a line
names what is being simulated (realization, interval, property / facies), and the elapsed and estimated remaining time are shown.
* **What the bar counts:** active cells. Inside a realization, facies and every property weigh one active-cell count each,
  permeability a tenth and the target check a twentieth (rough guesses; they only shape the bar within a realization).
* **Estimated time:** a straight-line extrapolation of the time spent over the fraction done, so it is rough in the first realization
  and shows "estimating" until there is something to extrapolate from. It cannot foresee redrawn realizations or an interval that is
  slower than the earlier ones. Realizations taken from the cache are excluded from the estimate and are listed as `[cache]`.
* **Stop:** takes effect at the next safe point, i.e. after the interval or domain being simulated (one numerical kernel cannot be
  interrupted). A stopped realization is not stored; finished ones stay in the cache. Closing the dialog stops the run first.
* While a run is going the tabs are locked so the entries cannot change under it.
* Not covered yet: building the grid (shown only as "building the grid" without a bar), the other dialogs, and speed itself
  (still single-threaded, about 100k cells/s).

### Running realizations in parallel (`cosma.realization_runner`)

"Worker processes" in the realizations tab (a whole number, no preset; 1 = this process only). Realizations are independent and
every seed derives from (base seed, number, attempt, stage), so **the result does not depend on the number of workers or on the
order they finish in** (tested: 1 worker and 2 workers give bit-identical cache files and the same exported GRDECL).
* Workers are separate processes started with `spawn` on every platform. Each receives the project, grid and plan once and builds its
  own manager; with a cache folder each worker writes its own files. The GUI entry guards `__main__` and calls `freeze_support()`.
* **Memory** grows with the workers (each holds the model and its own arrays). The run states an estimate (this process: 250 MB +
  200 bytes per cell; each worker: 250 MB + 220 bytes per cell, calibrated by the benchmark below) and is refused only when the
  estimate exceeds the installed RAM; above 70% it is flagged in the report. It is an estimate from one synthetic case.
* **Memory of results:** run records no longer keep arrays (a 10M-cell realization is about 170 MB; 100 of them would not fit). They
  stay in the cache; `export` reloads them or, without a cache, regenerates the realization from its seeds (same result).
* The bar and the estimated time count the summed progress of the running realizations; the stage line shows how many have
  finished. Stop cancels realizations that have not started and stops running ones at their next safe point; finished ones stay.
* Limits: only 2 cores were available for the benchmark, so scaling beyond 2 workers is NOT measured (expect less than
  proportional gains once memory bandwidth is shared). Each worker starts slowly (imports, loading compiled kernels; the first ever
  run compiles them). Parallelism is across realizations only: a single realization still runs on one core.

### Benchmark (`benchmarks/bench_realizations.py`, 2026-11-12)

Synthetic flat model, two intervals, SIS facies + PORO + VSHALE + permeability, 12 neighbours, 3 multigrid levels, variogram ranges
1500 m / 10 m, 100 m x 100 m x 1 m cells, three wells; Linux, 2 cores, 8 GB.

| | 1M cells (100x100x100) | 4M cells (200x200x100) |
|---|---|---|
| one realization, one core | 33 s | 149 s |
| facies (SIS) | 10.9 s (92k cells/s) | 47.8 s (84k cells/s) |
| each property (SGS) | 10.9 s (92k cells/s) | 49 s (81k cells/s) |
| permeability / target check | 0.1 s / 0.3 s | 0.4 s / 1.3 s |
| peak memory, one process | 0.43 GB | 1.05 GB |
| 2 workers, wall time | 2 realizations: 37 s (1.78x) | 4 realizations: 295 s (2.02x) |
| peak memory, window + 2 workers | 1.33 GB | 2.85 GB |

Reading it: time per realization is about cells x (1 + number of properties) / 82,000 seconds per core at 4M cells; throughput per
cell falls about 10% from 1M to 4M. 100 realizations of this 4M case take about 4.1 hours on one core and about 2.1 hours on two.
Not covered: faulted or pinched geometry, many wells (more hard data), longer search templates, more neighbours, TGS, Windows, and
more than 2 cores. All of these can change the figures; rerun the script on the target machine.

### Correlated properties (for example VSHALE against porosity)

A property can be tied to one simulated before it. Per interval and facies the user types the correlation rho of the two
properties' normal scores (between -1 and 1, never preset). The secondary scores are `rho * z1 + sqrt(1 - rho^2) * e`, where z1 are
the primary's scores and e is an independent field simulated with the secondary's own variogram.
* Exact by construction: the secondary keeps its own distribution; the score correlation is rho; rho = 0 gives exactly the independent
  simulation; the primary is never changed. Hard values of the secondary stay exact (the residual e is conditioned on
  `(z2 - rho * z1) / sqrt(1 - rho^2)` at the hard cells).
* Hard values that contradict the correlation in the same cell (tail probability below the property's conflict probability) stop the
  run. The report compares rho with the score correlation at the hard cells when there are at least three.
* Limits: the correlation is typed, not derived from data (the report only shows the data figure). The spatial structure of the
  secondary is a mix, `rho^2 * (primary variogram) + (1 - rho^2) * (its own variogram)`, so its own variogram is reproduced only
  when the two are alike or rho is small. Only one primary per property (chains are allowed: C on B on A). The correlation is of
  normal scores, which differs from the correlation of the raw values when a distribution is strongly skewed.

### Spatially correlated permeability scatter

The scatter about the porosity-permeability line is a choice: `independent` (as before) or `correlated`, a standard-normal field per
interval and facies with a normal-score variogram the user names. The line, the spread and kv/kh, ky/kx are unchanged; at equal
porosity high- and low-permeability cells now form continuous bodies instead of salt-and-pepper noise, which matters for how a
plume fingers. Limits: the residual field is not conditioned to core permeability; it costs one more simulation pass (about a third
more time per realization with two properties); the variogram of the residuals is the user's to justify (there is no tool yet to
compute it from core residuals).
