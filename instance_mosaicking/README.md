# Instance Mosaicking Package

This package contains two implementations:

- `instance_mosaicking.maps` for geospatial map workflows with shapefiles, GeoTIFFs, CRS transforms, and shapefile outputs.
- `instance_mosaicking.large_images` for plain large-image workflows with pixel-coordinate COCO/mask annotations and JSON/PNG outputs.

Preferred imports:

```python
from instance_mosaicking.maps.annotation_map_split import Tile_Splitter
from instance_mosaicking.large_images.workflow import Workflow
```

The large-image workflow can be tested without GDAL:

```powershell
python -m pytest instance_mosaicking\tests_large_images -q
```
