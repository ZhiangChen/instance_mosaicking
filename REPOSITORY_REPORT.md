# Repository Report: Instance Mosaicking / RSISA

## Executive summary

This repository implements **RSISA**, a remote-sensing instance segmentation workflow for large orthomosaic images and maps. Its main contribution is not a new neural network model; it is the geospatial pre-processing and post-processing around instance segmentation:

1. **Annotation map splitting**: split a large polygon annotation shapefile, and optionally its raster image, into overlapping geospatial tiles.
2. **Tile-level instance data creation**: convert tiled `.tif`/`.shp` pairs into Mask R-CNN-style dictionaries or pickle files.
3. **Instance registration / mosaicking**: merge duplicate object instances predicted on overlapping tiles back into a single georeferenced shapefile.
4. **Evaluation and analysis**: compare merged polygons to ground truth and generate timing, memory, and performance plots.

The code is organized as a Python package under `rsisa/rsisa`, with tutorial notebooks, example scripts, a Mask R-CNN helper folder, and Docker setup.

## Repository layout

| Path | Purpose |
| --- | --- |
| `README.md` | Main project overview, installation instructions, and tutorial links. |
| `docker/` | Ubuntu-based Dockerfile and helper commands for running notebooks/package installation. |
| `docs/` | Figures and SVG plots used by the README/manuscript. |
| `jupyter_notebooks/` | Synthetic and real-data tutorials for the RSISA workflow. |
| `rsisa/setup.py` | Package metadata and Python dependencies. |
| `rsisa/rsisa/` | Core RSISA package: generation, splitting, registration, evaluation, workflow orchestration. |
| `rsisa/mask_rcnn/` | Mask R-CNN dataset/model/training/evaluation/visualization helpers. |
| `rsisa/analysis/` | Scripts and notebooks for performance/time-complexity experiments. |
| `rsisa/tests/` | Example-style scripts with hard-coded `/root/rsisa/...` paths. They are not isolated pytest unit tests. |

## Main end-to-end workflow

The intended synthetic workflow is:

1. Generate random ellipse polygons and an optional fake raster:
   - Input: synthetic parameters such as tile length, density, ellipse size range, overlap, pixel resolution.
   - Output: `ellipses_*.shp` and optionally `ellipses_*.tif`.

2. Split annotation and raster maps into tiles:
   - Input: source shapefile, optional source `.tif`, CRS, tile size, overlap, study area bounds.
   - Output: tile shapefiles named `{x}_{y}.shp`, optional tile rasters named `{x}_{y}.tif`, timing/memory `.npy`.

3. Convert tile annotations to model-style instance data:
   - Input: directory containing matching `{x}_{y}.tif` and `{x}_{y}.shp`.
   - Output: Python `(image, target)` items in memory, optional `{x}_{y}.pickle` or `{x}_{y}.zip` files.

4. Register and merge instances across tile overlaps:
   - Input: directory of tile pickle/zip predictions plus matching tile image paths embedded in each pickle.
   - Output: per-tile `{x}_{y}_merge.shp`, `_remaining.shp`, final merged shapefile, `_time_space.npy`, and optional `_twins.pickle`.

5. Evaluate merged output:
   - Input: cropped ground-truth shapefile and merged shapefile.
   - Output: printed accuracy, precision, recall, FP/FN diagnostics, and `{merged_name}_results.npy`.

For real data, the generation step is skipped. Real orthomosaic `.tif` and annotation `.shp` files are split, optionally passed through Mask R-CNN, then registered and evaluated.

## Core package scripts

### `rsisa/rsisa/ellipse_instance_generation.py`

Generates synthetic ellipse instances and writes them as geospatial polygons.

Key functions:

| Function | Inputs | Outputs |
| --- | --- | --- |
| `generate_ellipse(center_x, center_y, width, height, angle)` | Ellipse center, major/minor dimensions, rotation angle. | A Shapely `Polygon` approximating the ellipse. |
| `spawn_ellipses(tile_length, ellipse_length_range, density, tile_number, overlap, shp_path, generate_tiff, xres=0.01, yres=0.01)` | Synthetic scene parameters, output directory, raster-generation flag, raster resolution. | `ellipses_{...}.shp`; if `generate_tiff=True`, also `ellipses_{...}.tif`. |

Important details:

- Uses EPSG `32611` / WGS 84 UTM zone 11N.
- Number of generated ellipses is `tile_number ** 2 * density`.
- The raster output is a single-band fake GeoTIFF with zero-valued pixels, used mainly to define geospatial bounds and pixel resolution.

### `rsisa/rsisa/annotation_map_split.py`

Contains two major classes: `Tile_Splitter` and `Dataset`.

#### `Tile_Splitter`

Splits a large annotation shapefile into overlapping tile shapefiles. If a source raster is supplied, it also clips matching raster tiles.

Constructor inputs:

| Input | Meaning |
| --- | --- |
| `shapefile_path` | Source polygon shapefile to split. |
| `save_dir` | Output tile directory. Created if missing. |
| `crs` | CRS used when no raster is supplied. |
| `area_x1`, `area_y1`, `area_x2`, `area_y2` | Study-area bounds when no raster is supplied. |
| `tile_size` | Tile width/height in map units, typically meters. |
| `overlap` | Tile overlap in map units. |
| `tif_path` | Optional source GeoTIFF; if present, its bounds and CRS override area/CRS arguments. |
| `keep_instance_tif` | If true, only saves raster tiles where instances exist. |

Main output files:

- `{i}_{j}.shp` tile shapefiles with polygon geometry and string `id`.
- `{i}_{j}.tif` tile rasters when `tif_path` is supplied.
- `{source_name}_time_space.npy`, containing elapsed time and memory samples.

#### `Dataset`

Creates Mask R-CNN-style instance data from tiled `.tif` and `.shp` files.

Constructor inputs:

| Input | Meaning |
| --- | --- |
| `pixel_size` | Target square image size in pixels. |
| `split_path` | Directory containing tile `.tif` files and matching `.shp` files. |
| `input_channel` | Channels selected from the raster image. Supports one or multiple channels. |
| `max_tile_number` | Search bound used to detect edge tiles. |

`__getitem__(idx)` output:

- `image`: normalized NumPy array shaped channel-first as `(C, H, W)`.
- `target`: dictionary with `boxes`, `labels`, `masks`, `image_id`, `area`, `iscrowd`, `image_name`, `scores`, and `ids`.
- Returns `(image, None)` if no valid objects exist for that tile.

Other outputs:

- `show(idx)` writes `rgb.png` and `masks.png` into the split directory.
- `show_overlay(...)` writes `overlap.png` and `overlap_background.png`.
- `show_bbox(idx)` writes `bbox.png` and `bbox_background.png`.
- `save_pickles(zip=False)` writes `{tile}.pickle`; with `zip=True`, writes `{tile}.zip` and removes the pickle.

### `rsisa/rsisa/instance_registration.py`

Registers and merges duplicate instances across overlapping prediction tiles.

Main class: `Instance_Registration`.

Constructor inputs:

| Input | Meaning |
| --- | --- |
| `instance_dir` | Directory containing tile `.pickle` or `.zip` prediction files. |
| `save_shapefile` | Final merged shapefile path. |
| `tif_height_pixel`, `tif_width_pixel` | Original tile raster dimensions. |
| `tif_height_res`, `tif_width_res` | Raster resolution in map units per pixel. Vertical resolution is typically negative. |
| `tif_map_file` | Optional full-map or reference GeoTIFF used to set global origin. |
| `tile_overlap_ratio` | Overlap divided by tile size. |
| `detection_threshold` | Filters predictions by confidence score. |
| `segmentation_threshold` | Converts predicted mask scores to binary masks. |
| `iou_threshold` | Threshold for merging overlap-region masks. |
| `disable_merge` | If true, writes instances without cross-tile merging. |
| `test` | `True` for 3D masks from synthetic data; `False` for 4D model predictions. |
| `unzip` | Reads `.zip` files instead of `.pickle` files. |

Expected pickle/zip contents:

```python
{
    "image": ...,
    "bb": ...,
    "labels": ...,
    "scores": ...,
    "masks": ...,
    "image_name": ".../{x}_{y}.tif",
    "ids": ...
}
```

Main methods and outputs:

| Method | Inputs | Outputs |
| --- | --- | --- |
| `start_registration(continue_regitration=False)` | Optional continuation flag. | Processes tile files, saves per-tile merge shapefiles, writes `{save_name}_time_space.npy`, returns updated tile files and timestamps. |
| `combine_shapefiles()` | Uses generated merge shapefiles and remaining shapefile. | Writes final `save_shapefile`. |
| `_save_tile(tile_indices)` | Internal tile key. | Writes `{x}_{y}_merge.shp` in `instance_dir`. |
| `_save_remaining_tiles()` | Internal pending instances. | Writes `{save_name}_remaining.shp`. |
| `clean_twin_instances()` / `merge_twins()` | Internal duplicate groups. | Removes/combines duplicate merged instances before final output. |

The algorithm classifies instances by whether their mask touches tile borders or corners, compares overlap-region masks with neighboring tiles, and merges global masks when IoU exceeds the threshold.

### `rsisa/rsisa/evaluation.py`

Evaluates merged polygons against ground truth.

Key functions:

| Function | Inputs | Outputs |
| --- | --- | --- |
| `IoU(poly1, poly2)` | Two Shapely polygons. | Scalar intersection-over-union. |
| `uniqueness(TP, TN)` | Lists of true-positive and true-negative IDs. | Duplicate IDs found in successful outputs. |
| `evaluate(groundtruth_shapefile, merged_shapefile, IoU_threshold=0.88)` | Ground-truth and merged shapefile paths. | Prints total, TP/TN/FP/FN, duplicate IDs, accuracy, precision, recall; writes `{merged_name}_results.npy`. |

Assumption:

- The merged shapefile has an `id` field. IDs containing commas are treated as merged positives; single IDs are treated as unmerged negatives.

### `rsisa/rsisa/workflow.py`

Orchestrates the full synthetic experiment.

Input:

- A `config` dictionary with `density`, `tile_length`, `tile_number`, `overlap`, `pixel_size`, `ellipse_min`, `ellipse_max`, `iou_threshold`, `zip`, and `save_dir`.

Outputs:

- Synthetic shapefile and raster.
- Split tile shapefiles/rasters.
- Pickle or zip tile instance files.
- Cropped ground-truth shapefile.
- Merged shapefile.
- Evaluation `.npy` and timing/memory `.npy` files.

The `__main__` block runs a hard-coded large synthetic configuration under `/root/rsisa/data/random_generation_8`.

## Analysis scripts

### `rsisa/analysis/analysis.py`

Automates synthetic experiments for performance and time-complexity analysis.

Functions:

| Function | Inputs | Outputs |
| --- | --- | --- |
| `create_config(param, save_dir)` | Numeric parameter vector and output directory. | Workflow config dictionary. |
| `format_params(params, save_dir)` | Parameter matrix. | List of configs for one-at-a-time parameter sweeps. |
| `automate_performance_evaluation(save_dir)` | Output directory. | Runs many `Workflow` experiments and produces their workflow outputs. |
| `automate_time_analysis_M(save_dir)` | Output directory. | Runs a fixed high tile-number experiment. |
| `automate_time_analysis_K(k, save_dir)` | Density `k`, output directory. | Runs density sweep experiment. |
| `automate_time_analysis_P(p, save_dir)` | Ellipse max-size parameter `p`, output directory. | Runs max-pixel-size sweep experiment. |

The `__main__` block creates many `/root/rsisa/data/time_complexity_*` experiment directories and runs repeated workflows.

### `rsisa/analysis/plot_results.ipynb`

Loads experiment `.npy` files from a sibling `data` directory, computes means/stds, and saves SVG plots into `docs/`.

Inputs:

- Recursive `.npy` files under `../../data`, excluding `twins`.
- Files containing `results.npy` and `time_space.npy`.

Outputs:

- `docs/annotation_split_time.svg`
- `docs/annotation_split_space.svg`
- `docs/instance_registration_tile_time.svg`
- `docs/instance_registration_tile_space.svg`
- `docs/instance_registration_max_pixel_time.svg`
- `docs/instance_registration_max_pixel_space.svg`
- `docs/instance_registration_density_time.svg`
- `docs/instance_registration_density_space.svg`

## Mask R-CNN scripts

### `rsisa/mask_rcnn/dataset.py`

Builds train/validation/test splits and serves geospatial tile data to PyTorch.

Inputs:

- `create_datasets(data_path, split=(0.6, 0.8))`: directory of `.tif` tiles.
- `Dataset(json_file_list, pixel_size, input_channel=(0,1,2), transforms=None)`: split JSON files and image settings.

Outputs:

- `create_datasets` writes `train_split.json`, `valid_split.json`, and `test_split.json`.
- `Dataset.__getitem__` returns `(image, target)` with image tensor/array and Mask R-CNN target fields.
- `imageStat(Nm)` returns mean, std, max, min per selected channel.
- `show(idx)` displays image/mask visualizations.

### `rsisa/mask_rcnn/mask_rcnn.py`

Builds a torchvision Mask R-CNN model.

Input:

- Class count, image normalization statistics, channel count, max detections per image, and anchor sizes.

Output:

- A configured `torchvision.models.detection.MaskRCNN` model with replaced classification/mask heads and optional non-RGB first convolution.

### `rsisa/mask_rcnn/mask_rcnn_utils.py`

Torchvision reference utilities for training loops and distributed execution.

Inputs/outputs:

- Tracks smoothed metrics, gathers/reduces distributed data, creates warmup LR schedulers, creates directories, manages distributed initialization, and provides `collate_fn`.

### `rsisa/mask_rcnn/mask_rcnn_transforms.py`

Simple data transforms for detection datasets.

Inputs:

- Image and target dictionary.

Outputs:

- `Compose` applies transforms in sequence.
- `RandomHorizontalFlip` flips image, boxes, masks, and keypoints.
- `ToTensor` converts images to float tensors.

### `rsisa/mask_rcnn/coco_utils.py`

Torchvision COCO conversion/loading helpers.

Inputs:

- COCO image/annotation folders or a dataset that returns Mask R-CNN targets.

Outputs:

- COCO-compatible API objects, filtered COCO datasets, masks from polygon annotations, and converted dataset dictionaries.

### `rsisa/mask_rcnn/coco_eval.py`

COCO evaluator wrapper adapted from torchvision references.

Inputs:

- COCO ground-truth API object, IoU types, and prediction dictionaries.

Outputs:

- COCO evaluation accumulation and summaries for bbox, segmentation, and keypoints.

### `rsisa/mask_rcnn/visualize.py`

Visualization utilities adapted from Matterport Mask R-CNN.

Inputs:

- Images, boxes, masks, class IDs/names, scores, captions, overlaps, model weights.

Outputs:

- Matplotlib figures, masked image arrays, tables, precision-recall plots, overlap plots, and box/mask overlays.

### `rsisa/mask_rcnn/mask_rcnn.ipynb`

Training/inference tutorial notebook.

Inputs:

- Tile data under `data/rock`, split JSON files, Mask R-CNN settings, pretrained torchvision weights, model checkpoint paths.

Outputs:

- Trained model parameters under `model/epoch_*.param`.
- COCO-style evaluation summaries.
- Inference pickle/zip outputs under `data/rock`.
- Optional PNG overlays for predictions.

## Tutorial notebooks

### `jupyter_notebooks/synthetic_data_tutorial.ipynb`

Walks through synthetic ellipse generation, annotation splitting, pickle/zip generation, instance registration, and evaluation.

Main inputs:

- Synthetic parameters: `tile_length=10`, `density=100`, `tile_number=3`, `overlap=1`, `pixel_size=0.01`, ellipse range `(0.3, 1.0)`.
- Output path `/root/rsisa/data/random_generation_tutorial`.

Main outputs:

- Synthetic `ellipses_*.shp` and `.tif`.
- `split_shp_tif` tile files.
- Tile pickle/zip files.
- Merged shapefile.
- Evaluation metrics and `.npy` outputs.

### `jupyter_notebooks/real_data_tutorial.ipynb`

Runs the workflow on real rock data.

Main inputs:

- `/root/rsisa/data/rocks/rocks.shp`
- `/root/rsisa/data/rocks/rocks.tif`
- Tile and pixel settings, plus optional Mask R-CNN prediction outputs.

Main outputs:

- Split real-data tiles.
- Pickle/zip tile data.
- Visual QA images from `Dataset.show*`.
- `rocks_merge.shp`
- Evaluation metrics against `0_0.shp` or filtered shapefiles.

## Example scripts in `rsisa/tests`

These files execute examples directly at import/runtime and use fixed absolute paths. They are useful as workflow references but are not self-contained automated tests.

| File | Purpose | Inputs | Outputs |
| --- | --- | --- | --- |
| `test_ellipse_instance_generation.py` | Calls `spawn_ellipses`. | Hard-coded `/root/rsisa/data/random_generation`. | Synthetic shapefile and fake GeoTIFF. |
| `test_annotation_map_split.py` | Demonstrates split with/without TIFF, cropping, and pickle creation. | Hard-coded generated ellipse files. | Tile `.shp`, tile `.tif`, cropped shapefile, visual PNGs, pickles. |
| `test_instance_registration.py` | Runs registration on generated split data. | Hard-coded `split_shp_tif` directory and merge path. | Per-tile merge shapefiles and final merged shapefile. |
| `test_evaluation.py` | Evaluates one generated experiment. | Hard-coded ground-truth and merged shapefile paths. | Printed metrics and `_results.npy`. |
| `__init__.py` | Empty package marker. | None. | None. |

## Packaging and environment

### `rsisa/setup.py`

Defines package name `rsisa`, version `0.1.1`, Python `>=3.6`, and dependencies:

- `numpy`, `pandas`, `geopandas`, `rasterio`, `rioxarray`, `opencv-python`, `tqdm`, `shapely`, `matplotlib`, `fiona`, `pyproj`, `GDAL`.

### `docker/Dockerfile`

Builds from `ubuntu:focal`, installs OS packages for Python, GDAL, Git, and image libraries, then installs geospatial Python dependencies and Jupyter via `pip3`.

### `docker/bash_help.md`

Documents cloning, Docker build/run, Jupyter startup, opening a new shell inside the container, and restarting the stopped container.

## Data formats

### Input formats

- Polygon shapefiles (`.shp`) with instance geometries.
- GeoTIFF raster maps (`.tif`) for real imagery or synthetic map bounds.
- Tile prediction pickle/zip files with image, masks, scores, labels, image path, and IDs.
- JSON file lists for Mask R-CNN train/validation/test splits.
- NumPy `.npy` experiment results for analysis notebooks.

### Output formats

- Tile shapefiles: `{x}_{y}.shp`
- Tile rasters: `{x}_{y}.tif`
- Dataset pickles/zips: `{x}_{y}.pickle` or `{x}_{y}.zip`
- Per-tile merged shapefiles: `{x}_{y}_merge.shp`
- Remaining instances shapefile: `{merged_name}_remaining.shp`
- Final merged shapefile: user-provided `save_shapefile`
- Timing/memory arrays: `*_time_space.npy`
- Evaluation arrays: `*_results.npy`
- Visualization PNGs: `rgb.png`, `masks.png`, `overlap.png`, `overlap_background.png`, `bbox.png`, `bbox_background.png`
- Analysis SVGs in `docs/`
- Mask R-CNN checkpoints: `model/epoch_*.param`

## Notable implementation assumptions and risks

- Many paths in scripts/notebooks assume a Docker/Linux layout under `/root/rsisa/...`.
- File-name parsing assumes tile files are named like `{x}_{y}.tif`, `{x}_{y}.pickle`, or `{x}_{y}.zip`.
- Some path parsing uses `/` separators, which can be fragile on Windows paths.
- The example test files run code at top level and require pre-existing data; they are not portable unit tests.
- `Dataset.save_pickles()` assumes `target` is not `None`; tiles without valid instances could fail if present in `data_files`.
- `Instance_Registration` reads `self.tile_files[1]` during initialization, so it expects at least two prediction files.
- Several scripts import reference modules with local absolute/relative assumptions, especially in `rsisa/mask_rcnn`.
- The Mask R-CNN code uses APIs such as `pretrained=True` that may be deprecated in newer torchvision versions.

## Quick mental model

RSISA turns a large geospatial instance segmentation problem into a tiled problem and then stitches the instance predictions back together. The important invariant is that every tile carries enough spatial metadata, naming convention, and overlap context for masks to be converted from local tile pixels into global coordinates. Once masks are in that shared coordinate space, overlapping border fragments can be compared, merged, converted back into polygons, and written as a final shapefile.
