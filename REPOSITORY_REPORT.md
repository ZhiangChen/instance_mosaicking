# Repository Report: Instance Mosaicking

## Executive Summary

This repository contains one package, `instance_mosaicking`, with two workflow families:

- `instance_mosaicking.maps`: the geospatial map workflow.
- `instance_mosaicking.large_images`: the plain large-image workflow using pixel coordinates.

Both workflows solve the same high-level problem: split a large input into overlapping tiles, produce or consume tile-level instance masks, convert tile predictions into a global coordinate system, merge duplicate instances across tile overlaps, and export merged outputs.

## Current Layout

| Path | Purpose |
| --- | --- |
| `instance_mosaicking/maps/` | Geospatial map implementation for shapefiles, GeoTIFFs, CRS-aware splitting, registration, evaluation, and workflow orchestration. |
| `instance_mosaicking/large_images/` | Plain-image implementation for pixel-coordinate tiling, COCO/mask annotations, registration, evaluation, and workflow orchestration. |
| `instance_mosaicking/mask_rcnn/` | Mask R-CNN helper code retained for model training/inference workflows. |
| `instance_mosaicking/analysis/` | Performance and timing analysis scripts/notebooks for map experiments. |
| `instance_mosaicking/tests_large_images/` | Self-contained pytest suite for the large-image workflow. |
| `instance_mosaicking/tests_maps/` | Legacy map workflow examples with hard-coded external data paths. |
| `jupyter_notebooks/` | Map tutorials plus the synthetic large-image tutorial. |
| `docs/` | Figures and analysis plots used by documentation/manuscript material. |
| `docker/` | Dockerfile and helper commands for a GDAL-ready map workflow environment. |

## Maps Workflow

The `maps` package is the geospatial path. It operates in map coordinates and is intended for orthomosaics, remote-sensing rasters, and other CRS-aware datasets.

Main inputs:

- polygon shapefiles
- GeoTIFF rasters
- CRS metadata
- tile prediction pickle/zip files

Main outputs:

- tile shapefiles and optional tile rasters
- Mask R-CNN-style tile pickles/zips
- per-tile merged shapefiles
- final merged shapefile
- timing, memory, and evaluation arrays

Key modules:

- `instance_mosaicking.maps.annotation_map_split`
- `instance_mosaicking.maps.instance_registration`
- `instance_mosaicking.maps.evaluation`
- `instance_mosaicking.maps.ellipse_instance_generation`
- `instance_mosaicking.maps.workflow`

The map registration algorithm uses the original overlap mechanism: instances are bucketed by tile edge/corner location, compared only with corresponding adjacent-tile candidates, and merged when overlap-region mask IoU exceeds the configured threshold.

## Large-Images Workflow

The `large_images` package is the non-georeferenced path. It operates directly in image pixels and does not require GDAL, raster transforms, CRS metadata, or shapefiles.

Supported inputs:

- plain images: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`
- COCO polygon JSON annotations
- PNG/TIF instance-label masks where `0` is background and nonzero values are instance IDs
- tile prediction pickle/zip files

Main outputs:

- `tiles.json`
- `tile_annotations.json`
- tile images under `tiles/`
- optional mask crops under `tile_masks/`
- tile prediction pickles/zips
- `merged_instances.json`
- `merged_pixel_polygons.json`
- `merged_instance_mask.png`
- `merged_overlay.png`

Large-image registration follows the same overlap-region merge mechanism as `maps`; source IDs are retained only as metadata and do not drive merging.

## Packaging And Imports

Canonical imports now use the `instance_mosaicking` namespace:

```python
from instance_mosaicking.maps.annotation_map_split import Tile_Splitter
from instance_mosaicking.large_images.workflow import Workflow
```

The distribution metadata lives at `instance_mosaicking/setup.py` and is named `instance-mosaicking`.

## Tests And Validation

Self-contained large-image tests:

```powershell
.\.venv\Scripts\python.exe -m pytest instance_mosaicking\tests_large_images -q
```

These tests cover synthetic fixture generation, COCO polygon splitting, PNG/TIF mask splitting, dataset mask creation, pickle output, overlap-only registration, evaluation helpers, and end-to-end workflow execution.

Map tests in `instance_mosaicking/tests_maps` are legacy examples. They require external data under Docker-style paths and should not be treated as portable unit tests.

## Known Risks

- Full map dependencies include GDAL/rasterio/geopandas and can be difficult to install on Windows without a prepared geospatial environment.
- Some legacy map notebooks and scripts still contain historical `/root/instance_mosaicking/...` data paths in saved outputs or examples.
- The large-image workflow is intentionally lightweight and does not produce shapefiles.
