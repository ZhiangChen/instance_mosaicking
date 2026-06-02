import json
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


def _tile_starts(length, tile_size, overlap):
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be non-negative and smaller than tile_size")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _annotation_polygons(annotation):
    polygons = []
    segmentation = annotation.get("segmentation", [])
    if isinstance(segmentation, list):
        for coords in segmentation:
            if len(coords) >= 6:
                polygons.append([(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)])
    return polygons


def _clip_polygon_to_rect(points, left, top, right, bottom):
    def clip_edge(vertices, inside, intersect):
        if not vertices:
            return []
        output = []
        previous = vertices[-1]
        previous_inside = inside(previous)
        for current in vertices:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersect(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersect(previous, current))
            previous = current
            previous_inside = current_inside
        return output

    def vertical(x_value):
        return lambda a, b: (x_value, a[1] + (b[1] - a[1]) * ((x_value - a[0]) / (b[0] - a[0])) if b[0] != a[0] else a[1])

    def horizontal(y_value):
        return lambda a, b: (a[0] + (b[0] - a[0]) * ((y_value - a[1]) / (b[1] - a[1])) if b[1] != a[1] else a[0], y_value)

    clipped = list(points)
    clipped = clip_edge(clipped, lambda p: p[0] >= left, vertical(left))
    clipped = clip_edge(clipped, lambda p: p[0] <= right, vertical(right))
    clipped = clip_edge(clipped, lambda p: p[1] >= top, horizontal(top))
    clipped = clip_edge(clipped, lambda p: p[1] <= bottom, horizontal(bottom))
    return clipped


def _polygon_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _polygon_to_coco(points):
    return [float(value) for xy in points for value in xy]


def _bbox_from_polygon(points):
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return [float(minx), float(miny), float(maxx - minx), float(maxy - miny)]


def _bbox_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    minx, maxx = int(xs.min()), int(xs.max())
    miny, maxy = int(ys.min()), int(ys.max())
    return [float(minx), float(miny), float(maxx - minx + 1), float(maxy - miny + 1)]


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
        min_fragment_area=1.0,
        tile_extension=None,
    ):
        self.image_path = Path(image_path)
        self.save_dir = Path(save_dir)
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.annotation_path = Path(annotation_path) if annotation_path else None
        self.annotation_format = annotation_format
        self.keep_empty_tiles = keep_empty_tiles
        self.min_fragment_area = float(min_fragment_area)
        self.tile_extension = tile_extension or self.image_path.suffix or ".png"

    def split(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        tile_dir = self.save_dir / "tiles"
        tile_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(self.image_path)
        width, height = image.size
        source_annotations, categories = self._load_annotations()
        mask_image = self._load_mask_image((width, height)) if self.annotation_path and self.annotation_format == "mask" else None
        mask_dir = self.save_dir / "tile_masks"
        if mask_image is not None:
            mask_dir.mkdir(parents=True, exist_ok=True)
        annotations_by_tile = defaultdict(list)
        tile_records = []
        coco_images = []
        coco_annotations = []
        annotation_id = 1
        image_id = 1

        for j, y_offset in enumerate(_tile_starts(height, self.tile_size, self.overlap)):
            for i, x_offset in enumerate(_tile_starts(width, self.tile_size, self.overlap)):
                tile_width = min(self.tile_size, width - x_offset)
                tile_height = min(self.tile_size, height - y_offset)
                name = f"{i}_{j}{self.tile_extension}"
                if mask_image is not None:
                    fragments = self._clip_mask_annotations(mask_image, mask_dir, name, x_offset, y_offset, tile_width, tile_height)
                else:
                    fragments = self._clip_annotations(source_annotations, x_offset, y_offset, tile_width, tile_height)
                if self.annotation_path and not self.keep_empty_tiles and not fragments:
                    continue

                image.crop((x_offset, y_offset, x_offset + tile_width, y_offset + tile_height)).save(tile_dir / name)
                tile_record = {
                    "name": name,
                    "index": [i, j],
                    "x_offset": x_offset,
                    "y_offset": y_offset,
                    "width": tile_width,
                    "height": tile_height,
                }
                tile_records.append(tile_record)
                coco_images.append({"id": image_id, "file_name": name, "width": tile_width, "height": tile_height})
                for fragment in fragments:
                    fragment["id"] = annotation_id
                    fragment["image_id"] = image_id
                    annotations_by_tile[name].append(fragment)
                    coco_annotations.append(fragment)
                    annotation_id += 1
                image_id += 1

        metadata = {
            "source_image": self.image_path.name,
            "image_width": width,
            "image_height": height,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "tiles": tile_records,
        }
        (self.save_dir / "tiles.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        tile_coco = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories or [{"id": 1, "name": "instance"}],
        }
        (self.save_dir / "tile_annotations.json").write_text(json.dumps(tile_coco, indent=2), encoding="utf-8")
        return metadata

    def _load_annotations(self):
        if not self.annotation_path:
            return [], [{"id": 1, "name": "instance"}]
        if self.annotation_format == "mask":
            return [], [{"id": 1, "name": "instance"}]
        if self.annotation_format != "coco":
            raise ValueError("Only COCO polygon and mask annotations are currently supported")
        data = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        annotations = []
        for annotation in data.get("annotations", []):
            for poly in _annotation_polygons(annotation):
                annotations.append((annotation, poly))
        return annotations, data.get("categories", [{"id": 1, "name": "instance"}])

    def _load_mask_image(self, image_size):
        mask_image = Image.open(self.annotation_path)
        if mask_image.size != image_size:
            raise ValueError("Mask annotation size must match the source image size")
        return mask_image

    def _clip_annotations(self, annotations, x_offset, y_offset, tile_width, tile_height):
        fragments = []
        for source, poly in annotations:
            clipped = _clip_polygon_to_rect(poly, x_offset, y_offset, x_offset + tile_width, y_offset + tile_height)
            area = _polygon_area(clipped)
            if area < self.min_fragment_area:
                continue
            local = [(x - x_offset, y - y_offset) for x, y in clipped]
            fragments.append(
                {
                    "category_id": source.get("category_id", 1),
                    "segmentation": [_polygon_to_coco(local)],
                    "bbox": _bbox_from_polygon(local),
                    "area": float(area),
                    "iscrowd": source.get("iscrowd", 0),
                    "source_annotation_id": source.get("id"),
                }
            )
        return fragments

    def _clip_mask_annotations(self, mask_image, mask_dir, tile_name, x_offset, y_offset, tile_width, tile_height):
        crop = mask_image.crop((x_offset, y_offset, x_offset + tile_width, y_offset + tile_height))
        mask_suffix = self.annotation_path.suffix.lower()
        if mask_suffix not in {".png", ".tif", ".tiff"}:
            mask_suffix = ".png"
        mask_name = f"{Path(tile_name).stem}{mask_suffix}"
        mask_path = mask_dir / mask_name
        crop.save(mask_path)

        crop_array = np.asarray(crop)
        if crop_array.ndim == 3:
            crop_array = crop_array[:, :, 0]
        fragments = []
        for value in sorted(int(v) for v in np.unique(crop_array) if int(v) != 0):
            instance_mask = crop_array == value
            area = int(instance_mask.sum())
            if area < self.min_fragment_area:
                continue
            bbox = _bbox_from_mask(instance_mask)
            if bbox is None:
                continue
            fragments.append(
                {
                    "category_id": 1,
                    "bbox": bbox,
                    "area": float(area),
                    "iscrowd": 0,
                    "source_annotation_id": value,
                    "mask_file": str((Path("tile_masks") / mask_name).as_posix()),
                    "mask_value": value,
                }
            )
        return fragments


class Dataset:
    def __init__(self, split_path, input_channel=None, pixel_size=None):
        self.split_path = Path(split_path)
        self.tile_dir = self.split_path / "tiles"
        self.input_channel = input_channel
        self.pixel_size = pixel_size
        self.metadata = json.loads((self.split_path / "tiles.json").read_text(encoding="utf-8"))
        self.annotation_data = json.loads((self.split_path / "tile_annotations.json").read_text(encoding="utf-8"))
        self.images = self.annotation_data.get("images", [])
        self.annotations_by_image = defaultdict(list)
        for annotation in self.annotation_data.get("annotations", []):
            self.annotations_by_image[annotation["image_id"]].append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image = Image.open(self.tile_dir / image_info["file_name"])
        if self.pixel_size:
            image = image.resize((self.pixel_size, self.pixel_size))
        array = np.asarray(image)
        if self.input_channel is not None and array.ndim == 3:
            array = array[:, :, list(self.input_channel)]
        annotations = self.annotations_by_image.get(image_info["id"], [])
        masks = []
        boxes = []
        labels = []
        areas = []
        for annotation in annotations:
            masks.append(self._mask_from_annotation(annotation, image.size))
            x, y, w, h = annotation["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(annotation.get("category_id", 1))
            areas.append(annotation.get("area", w * h))
        target = {
            "boxes": np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
            "labels": np.asarray(labels, dtype=np.int64),
            "masks": np.asarray(masks, dtype=np.uint8),
            "image_id": image_info["id"],
            "area": np.asarray(areas, dtype=np.float32),
            "iscrowd": np.zeros(len(annotations), dtype=np.int64),
            "ids": [a.get("source_annotation_id", a["id"]) for a in annotations],
        }
        if target["masks"].size == 0:
            target["masks"] = np.zeros((0, image.size[1], image.size[0]), dtype=np.uint8)
        return array, target

    def save_pickles(self, zip_output=False):
        pickle_paths = []
        for idx, image_info in enumerate(self.images):
            image, target = self[idx]
            scores = np.ones(len(target["labels"]), dtype=np.float32)
            data = {
                "image": image,
                "image_name": image_info["file_name"],
                "masks": target["masks"],
                "boxes": target["boxes"],
                "bb": target["boxes"],
                "labels": target["labels"],
                "scores": scores,
                "ids": target["ids"],
            }
            pickle_path = self.split_path / f"{Path(image_info['file_name']).stem}.pickle"
            with pickle_path.open("wb") as handle:
                pickle.dump(data, handle)
            if zip_output:
                zip_path = pickle_path.with_suffix(".zip")
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.write(pickle_path, arcname=pickle_path.name)
                pickle_path.unlink()
                pickle_paths.append(str(zip_path))
            else:
                pickle_paths.append(str(pickle_path))
        return pickle_paths

    def _mask_from_annotation(self, annotation, image_size):
        if "mask_file" in annotation:
            mask_path = self.split_path / annotation["mask_file"]
            mask = np.asarray(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            return (mask == annotation["mask_value"]).astype(np.uint8)

        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        for segmentation in annotation.get("segmentation", []):
            points = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
            if points:
                draw.polygon(points, fill=1)
        return np.asarray(mask, dtype=np.uint8)
