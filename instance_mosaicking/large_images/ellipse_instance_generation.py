import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


def _ellipse_polygon(xyxy, steps=48):
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx = (x2 - x1) / 2.0
    ry = (y2 - y1) / 2.0
    points = []
    for idx in range(steps):
        angle = 2.0 * math.pi * idx / steps
        points.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
    return points


def _polygon_area(points):
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _ellipse_annotation(annotation_id, image_id, category_id, xyxy):
    x1, y1, x2, y2 = xyxy
    polygon = _ellipse_polygon(xyxy)
    width = x2 - x1
    height = y2 - y1
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [[round(value, 3) for point in polygon for value in point]],
        "bbox": [x1, y1, width, height],
        "area": _polygon_area(polygon),
        "iscrowd": 0,
    }


def _ellipse_specs(ellipse_count, image_size):
    base_specs = [
        ((24, 24, 76, 76), (220, 74, 74)),
        ((96, 36, 156, 92), (62, 142, 205)),
        ((38, 104, 92, 164), (80, 175, 120)),
        ((104, 104, 168, 168), (235, 181, 70)),
        ((206, 28, 268, 88), (186, 102, 190)),
        ((292, 42, 352, 102), (87, 166, 155)),
        ((198, 126, 260, 188), (210, 118, 76)),
        ((292, 132, 350, 190), (105, 111, 205)),
        ((28, 224, 86, 282), (76, 168, 214)),
        ((118, 222, 178, 282), (218, 138, 172)),
        ((212, 226, 270, 286), (118, 188, 96)),
        ((300, 230, 360, 292), (226, 190, 82)),
    ]
    width, height = image_size
    return [
        (xyxy, color)
        for xyxy, color in base_specs[:ellipse_count]
        if xyxy[2] <= width and xyxy[3] <= height
    ]


def generate_fixture(save_dir, image_size=(256, 256), file_prefix="large_image_fixture", ellipse_count=4):
    """Create a deterministic plain-image fixture with ellipse COCO polygons."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    width, height = image_size
    image_path = save_dir / f"{file_prefix}.png"
    annotation_path = save_dir / f"{file_prefix}.json"
    mask_path = save_dir / f"{file_prefix}_mask.png"
    mask_tif_path = save_dir / f"{file_prefix}_mask.tif"

    image = Image.new("RGB", image_size, (242, 244, 238))
    instance_mask = Image.new("I", image_size, 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(instance_mask)
    objects = _ellipse_specs(ellipse_count, image_size)
    if len(objects) != ellipse_count:
        raise ValueError("ellipse_count is too large for the requested image_size")
    annotations = []
    for idx, (xyxy, color) in enumerate(objects, start=1):
        draw.ellipse(xyxy, fill=color)
        mask_draw.ellipse(xyxy, fill=idx)
        annotations.append(_ellipse_annotation(idx, 1, 1, xyxy))

    image.save(image_path)
    instance_mask.convert("L").save(mask_path)
    instance_mask.save(mask_tif_path)
    coco = {
        "images": [{"id": 1, "file_name": image_path.name, "width": width, "height": height}],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "instance"}],
    }
    annotation_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return {
        "image_path": str(image_path),
        "annotation_path": str(annotation_path),
        "mask_path": str(mask_path),
        "mask_tif_path": str(mask_tif_path),
        "annotations": coco,
    }
