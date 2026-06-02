# Plan: Support Large Non-Georeferenced Images

## Short answer

Yes, the current repository is designed primarily for **maps and geospatial rasters**. It expects georeferenced inputs such as shapefiles and GeoTIFFs, and much of the implementation depends on CRS, raster transforms, map bounds, and map-unit tile sizes.

To support large ordinary images without georeference, the repository should add a separate **`large_images/` implementation** with a pixel-coordinate workflow. The goal is to keep `rsisa/` dedicated to maps while letting users process large images whose annotations and predictions are expressed directly in image pixels.

## Current geospatial assumptions

The current code assumes:

- Input annotations are polygon shapefiles.
- Input imagery is usually a GeoTIFF.
- Tile extents are computed from map bounds or GeoTIFF bounds.
- Coordinates are map coordinates, usually meters.
- Pixel masks are converted back to polygons using raster transforms and CRS.
- Outputs are geospatial shapefiles.
- Tile files are named by tile grid index, such as `0_0.tif`, `1_0.tif`.

These assumptions are appropriate for orthomosaics, remote-sensing maps, and GIS workflows. They are not necessary for ordinary large images.

## Target large-image workflow

For non-georeferenced images, the desired workflow should be:

1. Load a large image.
2. Load annotations in pixel coordinates, if available.
3. Split the image into overlapping image tiles.
4. Split or crop annotation polygons/masks into matching tile coordinates.
5. Run an instance segmentation model on each tile.
6. Convert tile predictions from local tile pixels to global image pixels.
7. Merge duplicate predictions across tile overlaps.
8. Save final results in image-coordinate formats.

## Proposed input formats

The large-image workflow should support at least one polygon format and one mask format.

Recommended initial formats:

| Input type | Format | Notes |
| --- | --- | --- |
| Image | `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` | Treated as plain pixel images, not geospatial rasters. |
| Polygon annotations | COCO JSON | Common for instance segmentation and already compatible with many tools. |
| Polygon annotations | GeoJSON-like JSON without CRS | Useful if annotations are already polygons in image pixels. |
| Mask annotations | Label image / instance mask PNG | Each object instance has a unique integer ID. |
| Tile predictions | Pickle/zip schema compatible with `large_images` registration | Reuse the useful fields from the current prediction dictionaries where possible. |

Recommended first implementation target: **COCO JSON + image files**.

## Proposed output formats

The non-georeferenced path should not require shapefiles. Recommended outputs:

| Output | Purpose |
| --- | --- |
| Merged COCO JSON | Standard final instance output for large images. |
| GeoJSON-like pixel polygon JSON | Easy to inspect and lightweight. |
| Instance mask PNG/TIFF | Pixel-level output where each merged instance has a unique ID. |
| Overlay PNG | Visual QA of final merged instances on the original image. |
| Tile prediction pickle/zip | Compatibility with the existing registration workflow. |
| Timing/memory `.npy` or `.json` | Preserve existing analysis capability. |

Shapefile output can remain available only for geospatial workflows.

## Repository structure decision

Keep the map and plain-image implementations separate, but mirror the core file names:

| Folder | Responsibility |
| --- | --- |
| `rsisa/` | Existing geospatial/map implementation using shapefiles, GeoTIFFs, CRS, and map coordinates. |
| `large_images/` | New plain large-image implementation using image pixels, COCO-style data, masks, and image-coordinate outputs. It should use the same core file/module names as `rsisa/rsisa`. |

Do not put the non-georeferenced implementation inside `rsisa/rsisa`. `rsisa` should remain the package for maps. The new `large_images/` folder should mirror the core `rsisa/rsisa` module names so the two implementations are easy to compare:

| Map module | Large-image module | Large-image role |
| --- | --- | --- |
| `rsisa/rsisa/annotation_map_split.py` | `large_images/annotation_map_split.py` | Split plain images and pixel-coordinate annotations into overlapping tiles. |
| `rsisa/rsisa/instance_registration.py` | `large_images/instance_registration.py` | Register and merge tile predictions in global pixel coordinates. |
| `rsisa/rsisa/evaluation.py` | `large_images/evaluation.py` | Evaluate merged large-image outputs against pixel-coordinate ground truth. |
| `rsisa/rsisa/workflow.py` | `large_images/workflow.py` | Orchestrate the plain large-image workflow. |
| `rsisa/rsisa/ellipse_instance_generation.py` | `large_images/ellipse_instance_generation.py` | Optional synthetic plain-image instance generation for tests/tutorials. |

The module names should match, but the semantics differ: `rsisa` modules operate in geospatial map coordinates; `large_images` modules operate in image pixel coordinates.

Add a top-level `tests_large_images/` package that mirrors the existing `rsisa/tests/` layout. Use it to cover the `large_images/` implementation with the same style of unit tests, but operating on plain image pixels instead of map coordinates.

## `tests_large_images/` structure

The `tests_large_images/` package should mirror the `large_images/` implementation in the same way that `rsisa/tests/` mirrors `rsisa/`. The goal is to keep the test suite focused on pixel-coordinate behavior, with small deterministic fixtures and direct assertions about tile offsets, annotation clipping, registration, and output formats.

Recommended test layout:

| Test file | Responsibility | Key assertions |
| --- | --- | --- |
| `tests_large_images/test_annotation_map_split.py` | Validate plain-image tiling and annotation splitting. | Correct tile count, tile offsets, local annotation coordinates, empty-tile handling, and metadata writing. |
| `tests_large_images/test_instance_registration.py` | Validate global pixel registration and overlap merging. | Local-to-global conversion, duplicate suppression across tiles, score filtering, and merged output structure. |
| `tests_large_images/test_evaluation.py` | Validate pixel-coordinate evaluation helpers. | IoU correctness, uniqueness calculations, and threshold-based matching. |
| `tests_large_images/test_workflow.py` | Validate end-to-end plain-image orchestration. | Split, save pickles, register predictions, and export final outputs with the expected files. |
| `tests_large_images/test_ellipse_instance_generation.py` | Validate synthetic fixture generation for large images. | Deterministic image output, pixel annotations, and boundary-crossing objects for regression coverage. |

Suggested fixture set:

- A small `256 x 256` or `512 x 512` synthetic image stored in a temporary directory.
- One object fully inside a tile.
- One object crossing a horizontal tile boundary.
- One object crossing a vertical tile boundary.
- One object crossing both boundaries at a corner.
- One empty tile to verify optional empty-tile handling.

Suggested shared helpers inside `tests_large_images/`:

- Generate deterministic plain-image inputs with fixed shapes and colors.
- Build minimal COCO JSON or pixel-polygon JSON annotations.
- Create fake tile prediction dictionaries compatible with `large_images.instance_registration.Instance_Registration`.
- Compare polygon coordinates, bbox fields, and tile offsets using exact or near-exact pixel checks.

Testing phases for `tests_large_images/`:

### Phase 1: Splitter coverage

- Verify tiling on plain images with overlap.
- Verify annotation clipping into local tile coordinates.
- Verify `tiles.json` contents and tile naming.

### Phase 2: Registration coverage

- Verify local tile predictions convert to global pixels correctly.
- Verify overlap-based merge behavior across adjacent tiles.
- Verify non-overlapping instances remain separate.

### Phase 3: Workflow coverage

- Verify the full plain-image pipeline from split to merged export.
- Verify output files are written in the expected pixel-coordinate formats.
- Verify the workflow remains deterministic for the same fixture image.

### Phase 4: Regression coverage

- Add boundary-crossing cases that exercise corner overlaps.
- Add tests for empty tiles, tiny fragments, and score-threshold filtering.
- Add round-trip checks for the chosen output format, such as COCO JSON.

This test package should stay intentionally small and explicit. It should not depend on geospatial metadata, CRS transforms, or shapefile-specific behavior.

## Core design change

Inside `large_images/`, use a pixel-coordinate backend. If shared abstractions are introduced later, they should not force the map package and the image package into the same API prematurely.

Potential backend naming:

| Backend | Coordinate system | Location |
| --- | --- | --- |
| `GeoCoordinateBackend` | CRS/map coordinates and raster transforms | Existing `rsisa/` behavior, only if refactored later. |
| `PixelCoordinateBackend` | Global image pixels `(x, y)` | New `large_images/` behavior. |

The registration algorithm should operate on a common internal representation:

```python
{
    "tile_index": (i, j),
    "global_mask": [(x, y), ...],
    "global_bbox": [xmin, ymin, xmax, ymax],
    "score": float,
    "label": int,
    "id": str,
}
```

For maps, global coordinates are eventually converted to CRS polygons by `rsisa/`. For plain images, global coordinates remain pixels inside `large_images/`.

## New modules to add

### `large_images/annotation_map_split.py`

Responsible for splitting ordinary images and pixel annotations.

Main classes should mirror the map implementation:

```python
class Tile_Splitter:
    def __init__(
        self,
        image_path,
        save_dir,
        tile_size,
        overlap,
        annotation_path=None,
        annotation_format="coco",
        keep_empty_tiles=False,
    ):
        ...
```

```python
class Dataset:
    ...
```

Inputs:

- Large image path.
- Tile size in pixels.
- Overlap in pixels.
- Optional annotation file.
- Annotation format.

Outputs:

- Tile images: `{x}_{y}.png` or `{x}_{y}.tif`.
- Tile annotation files or one combined tile annotation JSON.
- Tile metadata file, such as `tiles.json`, containing global offsets and dimensions.

`Dataset` responsibilities:

- Read tile images.
- Read tile annotations in local tile pixels.
- Return Mask R-CNN-style `(image, target)` pairs.
- Save tile pickle/zip files compatible with `large_images.instance_registration.Instance_Registration`.

### `large_images/instance_registration.py`

Pixel-coordinate version of instance registration.

Main class should mirror the map implementation:

```python
class Instance_Registration:
    ...
```

Responsibilities:

- Read tile predictions.
- Convert local tile masks to global image pixels using tile offsets.
- Merge overlapping predictions.
- Save merged output as COCO JSON, pixel polygons, masks, and overlays.

This module can reuse ideas from the existing map registration logic, but the implementation should stay in `large_images/` unless shared utility extraction becomes clearly worthwhile.

### `large_images/evaluation.py`

Pixel-coordinate evaluation helpers.

Recommended functions should mirror the map implementation where possible:

- `IoU(poly1, poly2)`
- `uniqueness(TP, TN)`
- `evaluate(groundtruth_file, merged_file, IoU_threshold=...)`

The input/output files can be COCO JSON, pixel-polygon JSON, or instance masks rather than shapefiles.

### `large_images/workflow.py`

Plain-image workflow orchestration.

Main class should mirror the map implementation:

```python
class Workflow:
    ...
```

The config dictionary should use pixel units:

- `tile_size`
- `overlap`
- `image_path`
- `annotation_path`
- `annotation_format`
- `save_dir`
- `iou_threshold`
- `zip`

### `large_images/ellipse_instance_generation.py`

Optional synthetic instance generation for tests and tutorials.

It should generate ordinary plain images and pixel-coordinate annotations rather than shapefiles/GeoTIFFs. The file name mirrors the map module because the purpose is similar: generate synthetic instances for controlled experiments.

## Avoid refactoring `rsisa/` at first

The current `rsisa` `Instance_Registration` has useful logic, but it mixes:

- file I/O,
- tile ordering,
- mask thresholding,
- overlap-region matching,
- global coordinate conversion,
- polygon conversion,
- shapefile writing.

For the first implementation, do not refactor this code. Instead, implement the pixel-coordinate version in the mirrored `large_images/` files and copy/adapt only the concepts that are needed. After the plain-image path works and tests pass, shared utilities can be extracted if the duplication is real and stable.

Potential shared logic, if extracted later:

| Shared logic | Keep shared? |
| --- | --- |
| Detection threshold filtering | Yes |
| Segmentation threshold filtering | Yes |
| Largest-contour cleanup | Yes |
| Tile-edge location detection | Yes |
| Overlap-region mask IoU | Yes |
| Neighbor-tile merge logic | Yes |
| Converting masks to shapefiles | Keep in `rsisa/` only |
| Reading CRS/raster bounds | Keep in `rsisa/` only |
| Converting local pixels to global pixels | Shared idea, backend-specific implementation |

## Tile metadata

Plain images need explicit metadata because there is no GeoTIFF transform.

Create a `tiles.json` like:

```json
{
  "source_image": "large_image.png",
  "image_width": 120000,
  "image_height": 80000,
  "tile_size": 1024,
  "overlap": 128,
  "tiles": [
    {
      "name": "0_0.png",
      "index": [0, 0],
      "x_offset": 0,
      "y_offset": 0,
      "width": 1024,
      "height": 1024
    }
  ]
}
```

This file replaces the role of geospatial bounds/transforms in the non-georeferenced workflow.

## Annotation splitting for large images

For polygon annotations:

1. Read global pixel polygons.
2. For each tile, create a tile rectangle in global pixel coordinates.
3. Intersect each polygon with candidate tile rectangles.
4. Convert intersected polygon coordinates to local tile coordinates by subtracting tile offsets.
5. Save local tile annotations.

For instance masks:

1. Crop the instance-label mask using tile windows.
2. Preserve instance IDs.
3. Optionally discard tiny cropped fragments.
4. Convert fragments into masks or polygons for model training.

## Prediction merging for large images

The current merge logic can mostly stay conceptually the same:

1. Read predicted masks per tile.
2. Filter by detection score.
3. Threshold masks.
4. Identify masks touching overlap regions.
5. Convert local tile mask pixels to global image pixels:

```python
global_x = tile_x_offset + local_x
global_y = tile_y_offset + local_y
```

6. Compare global overlap masks between neighboring tiles.
7. Merge instances when overlap IoU exceeds the threshold.
8. Export merged instances.

## API proposal

High-level user-facing API:

```python
from large_images.workflow import Workflow

workflow = Workflow(
    image_path="image.png",
    annotation_path="annotations.json",
    save_dir="output",
    tile_size=1024,
    overlap=128,
    annotation_format="coco",
)

workflow.split()
workflow.save_training_pickles()
workflow.register_predictions(
    prediction_dir="output/predictions",
    output_format="coco",
)
```

This keeps the plain-image path easy to use without requiring GIS knowledge.

## Implementation phases

### Phase 1: Synthetic fixture generation

- Add `large_images/ellipse_instance_generation.py` first.
- Implement deterministic plain-image fixture generation and pixel-coordinate annotations.
- Add `tests_large_images/test_ellipse_instance_generation.py` to verify the generated image, polygons, and boundary-crossing fixtures.
- Use these fixtures as the input basis for later splitter and registration tests.

### Phase 2: Plain-image splitting

- Add `large_images/annotation_map_split.py` next, because the rest of the pipeline depends on tile images and `tiles.json`.
- Implement `Tile_Splitter` for plain image tiling.
- Support COCO polygon annotation splitting and local tile-coordinate conversion.
- Implement `Dataset` in the same file so training pickles can be produced from the split tiles.
- Add `tests_large_images/test_annotation_map_split.py` to verify tile count, tile offsets, clipping, empty-tile handling, metadata writing, and dataset return values.

### Phase 3: Pixel-coordinate registration

- Add `large_images/instance_registration.py` after the splitter, because it needs the tile metadata and training/prediction artifacts produced earlier.
- Implement `Instance_Registration` for global pixel coordinates.
- Convert local tile masks and boxes into global image pixels.
- Merge overlapping predictions and export merged results as COCO JSON, pixel polygons, masks, and overlays.
- Add `tests_large_images/test_instance_registration.py` to verify local-to-global conversion, score filtering, duplicate suppression across tiles, and merged output structure.

### Phase 4: Evaluation helpers

- Add `large_images/evaluation.py` after registration, because evaluation should consume merged pixel-coordinate outputs.
- Implement `IoU`, `uniqueness`, and `evaluate` for pixel-coordinate annotations and outputs.
- Add `tests_large_images/test_evaluation.py` to verify IoU correctness, uniqueness calculations, and threshold-based matching against merged outputs from the previous phase.

### Phase 5: End-to-end workflow

- Add `large_images/workflow.py` last, because it orchestrates the complete split, predict, register, and evaluate flow.
- Wire together the splitter, dataset, registration, and evaluation steps using pixel units.
- Add `tests_large_images/test_workflow.py` to verify the full plain-image pipeline, output file creation, and deterministic behavior.

### Phase 6: Output extensions and documentation

- Add any remaining output helpers, such as instance-mask export and polygon simplification, after the core pipeline is stable.
- Keep optional shapefile conversion out of the default path unless georeferencing metadata is explicitly supplied.
- Add a large-image tutorial notebook and a README section comparing map vs image workflows.
- Document accepted annotation formats and output formats.

## Testing strategy

Create small deterministic fixtures:

- A `256 x 256` plain image.
- A rectangle instance fully inside one tile.
- A circle or polygon crossing a horizontal tile boundary.
- A polygon crossing a vertical tile boundary.
- A polygon crossing both boundaries at a corner.

Expected tests, in dependency order:

- Fixture generation produces deterministic image and annotation outputs.
- Tile count and tile offsets are correct.
- Annotation fragments are clipped correctly into local tile coordinates.
- Dataset samples return the expected image/target structure.
- Local-to-global pixel conversion is correct.
- Duplicate predictions across overlap merge into one instance.
- Non-overlapping nearby instances do not merge.
- IoU and uniqueness computations are correct for pixel-coordinate shapes.
- The end-to-end workflow writes valid merged outputs.
- Final COCO JSON has valid image, annotation, bbox, segmentation, area, and score fields.

## Backward compatibility

Do not replace or rename the current geospatial workflow. Add the plain-image workflow alongside it in `large_images/`.

Recommended naming:

- Keep existing `Tile_Splitter`, `Dataset`, and `Instance_Registration` for geospatial use.
- Add matching classes under `large_images/` with the same names, such as `Tile_Splitter`, `Dataset`, `Instance_Registration`, and `Workflow`.
- Keep module names mirrored: `annotation_map_split.py`, `instance_registration.py`, `evaluation.py`, `workflow.py`, and optionally `ellipse_instance_generation.py`.
- Later, optionally extract shared lower-level utilities if both implementations converge naturally.

This avoids breaking users who already rely on shapefiles and GeoTIFFs.

## Main risks

- COCO polygons and masks can represent holes and multipolygons differently from Shapely/geospatial formats.
- Very large plain images may exceed PIL/OpenCV default safety limits or memory expectations.
- Exporting full global masks as coordinate lists can become memory-heavy; compressed masks or RLE should be supported for large outputs.
- Current `rsisa/` path parsing assumes `/` in several places; `large_images/` should use `pathlib` from the start.
- Tile-edge behavior must be carefully tested for partial edge tiles.

## Recommended direction

Add a **`large_images/` pixel-coordinate implementation** rather than forcing non-georeferenced images through fake CRS/GeoTIFF metadata. Fake georeferencing might work as a quick hack, but it would make the code harder to understand and would produce misleading outputs.

The clean long-term design is:

- `rsisa/` maps use CRS/map coordinates and shapefile/GeoTIFF outputs.
- `large_images/` plain images use pixel coordinates and COCO/mask/overlay outputs.
- Both share the same core idea: tile, predict, convert to global coordinates, merge duplicates, export final instances.
