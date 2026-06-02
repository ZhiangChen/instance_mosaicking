# Instance Mosaicking

Instance mosaicking provides preprocessing and postprocessing workflows for instance segmentation on two kinds of large inputs:

- `instance_mosaicking.maps`: geospatial map workflows using shapefiles, GeoTIFFs, CRS transforms, and shapefile outputs.
- `instance_mosaicking.large_images`: non-georeferenced large-image workflows using pixel coordinates, COCO JSON, instance-label masks, and image-coordinate outputs.

The map workflow keeps the original remote-sensing algorithms for annotation map splitting and prediction instance registration. The large-image workflow mirrors the same core idea in plain image pixels: split, predict, convert tile masks to global coordinates, merge duplicates across tile overlaps, and export final instances.

## Installation

For the lightweight large-image workflow:

```powershell
.\.venv\Scripts\python.exe -m pip install numpy pillow pytest
```

For the full geospatial map workflow, install the package dependencies from [instance_mosaicking/setup.py](./instance_mosaicking/setup.py). The map workflow depends on GDAL/rasterio/geopandas, so a GDAL-ready environment or Docker image is recommended.

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\instance_mosaicking
```

## Large Images

Use `instance_mosaicking.large_images` for plain `.png`, `.jpg`, `.tif`, or `.tiff` images without CRS metadata.

```python
from instance_mosaicking.large_images.workflow import Workflow

workflow = Workflow(
    image_path="large_image.png",
    annotation_path="annotations.json",
    save_dir="large_image_output",
    tile_size=1024,
    overlap=128,
    annotation_format="coco",
    keep_empty_tiles=True,
)

workflow.split()
workflow.save_training_pickles()
workflow.register_predictions("large_image_output")
```

Supported large-image annotation formats:

- `annotation_format="coco"` for COCO polygon JSON.
- `annotation_format="mask"` for PNG/TIF instance-label masks, where `0` is background and each nonzero value is one object instance.

The large-image tutorial is [jupyter_notebooks/synthetic_large_image_tutorial.ipynb](./jupyter_notebooks/synthetic_large_image_tutorial.ipynb).

## Maps

Use `instance_mosaicking.maps` for geospatial inputs:

```python
from instance_mosaicking.maps.annotation_map_split import Tile_Splitter, Dataset
from instance_mosaicking.maps.instance_registration import Instance_Registration
```

The map workflow expects shapefiles, GeoTIFFs, and CRS metadata. It writes tiled shapefiles/rasters and final merged shapefiles. The map tutorials are:

- [jupyter_notebooks/synthetic_map_tutorial.ipynb](./jupyter_notebooks/synthetic_map_tutorial.ipynb)
- [jupyter_notebooks/real_map_tutorial.ipynb](./jupyter_notebooks/real_map_tutorial.ipynb)

## Testing

The large-image tests are self-contained and run without GDAL:

```powershell
.\.venv\Scripts\python.exe -m pytest instance_mosaicking\tests_large_images -q
```

The map example tests under `instance_mosaicking/tests_maps` are workflow references with hard-coded data paths and are not portable unit tests.
