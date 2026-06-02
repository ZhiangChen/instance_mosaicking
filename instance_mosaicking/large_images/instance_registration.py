import json
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _bbox_from_coords(coords):
    ys = [y for _, y in coords]
    xs = [x for x, _ in coords]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return [int(minx), int(miny), int(maxx - minx + 1), int(maxy - miny + 1)]


def _xyxy_from_coords(coords):
    ys = [y for _, y in coords]
    xs = [x for x, _ in coords]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _bbox_segmentation(bbox):
    x, y, w, h = bbox
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def _set_iou(a, b):
    if not a and not b:
        return 0.0
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union if union else 0.0


class Instance_Registration:
    LOCATIONS = ("left", "right", "top", "bottom", "middle", "top-left", "top-right", "bottom-left", "bottom-right")

    def __init__(
        self,
        instance_dir,
        save_dir=None,
        tiles_metadata=None,
        detection_threshold=0.75,
        segmentation_threshold=0.5,
        iou_threshold=0.5,
        disable_merge=False,
        unzip=False,
    ):
        self.instance_dir = Path(instance_dir)
        self.save_dir = Path(save_dir) if save_dir else self.instance_dir
        self.tiles_metadata = Path(tiles_metadata) if tiles_metadata else self.instance_dir / "tiles.json"
        if not self.tiles_metadata.exists() and (self.instance_dir.parent / "tiles.json").exists():
            self.tiles_metadata = self.instance_dir.parent / "tiles.json"
        self.detection_threshold = detection_threshold
        self.segmentation_threshold = segmentation_threshold
        self.iou_threshold = iou_threshold
        self.disable_merge = disable_merge
        self.unzip = unzip
        self.metadata = json.loads(self.tiles_metadata.read_text(encoding="utf-8"))
        self.tile_lookup = {Path(tile["name"]).stem: tile for tile in self.metadata["tiles"]}
        self.tile_by_index = {tuple(tile["index"]): tile for tile in self.metadata["tiles"]}
        self.max_i = max(tile["index"][0] for tile in self.metadata["tiles"])
        self.max_j = max(tile["index"][1] for tile in self.metadata["tiles"])
        self.overlap = int(self.metadata.get("overlap", 0))
        self.instances = []
        self.tiles = {}

    def start_registration(self):
        start = time.time()
        files = self._prediction_files()
        for prediction_file in files:
            self._register_prediction_file(prediction_file)
        self.save_outputs()
        return [str(path) for path in files], np.asarray([time.time() - start])

    def save_outputs(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        coco = self.to_coco()
        coco_path = self.save_dir / "merged_instances.json"
        coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        (self.save_dir / "merged_pixel_polygons.json").write_text(json.dumps(self.to_pixel_polygons(), indent=2), encoding="utf-8")
        self.save_instance_mask(self.save_dir / "merged_instance_mask.png")
        self.save_overlay(self.save_dir / "merged_overlay.png")
        return str(coco_path)

    def to_coco(self):
        annotations = []
        for instance in self._active_instances():
            idx = len(annotations) + 1
            bbox = _bbox_from_coords(instance["coords"])
            annotations.append(
                {
                    "id": idx,
                    "image_id": 1,
                    "category_id": int(instance["label"]),
                    "segmentation": _bbox_segmentation(bbox),
                    "bbox": bbox,
                    "area": len(instance["coords"]),
                    "score": float(instance["score"]),
                    "tile_ids": instance["tile_ids"],
                    "ids": instance["ids"],
                }
            )
        return {
            "images": [
                {
                    "id": 1,
                    "file_name": self.metadata["source_image"],
                    "width": self.metadata["image_width"],
                    "height": self.metadata["image_height"],
                }
            ],
            "annotations": annotations,
            "categories": [{"id": 1, "name": "instance"}],
        }

    def to_pixel_polygons(self):
        features = []
        for instance in self._active_instances():
            idx = len(features) + 1
            bbox = _bbox_from_coords(instance["coords"])
            x, y, w, h = bbox
            features.append(
                {
                    "id": idx,
                    "label": int(instance["label"]),
                    "score": float(instance["score"]),
                    "bbox": bbox,
                    "polygon": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    "area": len(instance["coords"]),
                    "tile_ids": instance["tile_ids"],
                    "ids": instance["ids"],
                }
            )
        return {
            "source_image": self.metadata["source_image"],
            "image_width": self.metadata["image_width"],
            "image_height": self.metadata["image_height"],
            "instances": features,
        }

    def save_instance_mask(self, output_path):
        mask = np.zeros((self.metadata["image_height"], self.metadata["image_width"]), dtype=np.uint16)
        for instance_id, instance in enumerate(self._active_instances(), start=1):
            for x, y in instance["coords"]:
                mask[y, x] = instance_id
        Image.fromarray(mask).save(output_path)

    def save_overlay(self, output_path):
        source = self.tiles_metadata.parent / self.metadata["source_image"]
        if source.exists():
            image = Image.open(source).convert("RGBA")
        else:
            image = Image.new("RGBA", (self.metadata["image_width"], self.metadata["image_height"]), (255, 255, 255, 255))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        colors = [(220, 60, 60, 90), (60, 150, 210, 90), (80, 180, 110, 90), (235, 180, 55, 90)]
        for idx, instance in enumerate(self._active_instances()):
            color = colors[idx % len(colors)]
            for x, y in instance["coords"]:
                draw.point((x, y), fill=color)
        Image.alpha_composite(image, overlay).save(output_path)

    def _active_instances(self):
        return [instance for instance in self.instances if instance is not None]

    def _prediction_files(self):
        return sorted(list(self.instance_dir.glob("*.pickle")) + list(self.instance_dir.glob("*.zip")))

    def _register_prediction_file(self, prediction_file):
        prediction = self._read_prediction(prediction_file)
        tile_name = Path(prediction.get("image_name", prediction_file.stem)).stem
        tile = self.tile_lookup[tile_name]
        masks = _as_numpy(prediction.get("masks", []))
        if masks.ndim == 4:
            masks = masks[:, 0, :, :]
        scores = _as_numpy(prediction.get("scores", np.ones(masks.shape[0] if masks.ndim else 0)))
        labels = _as_numpy(prediction.get("labels", np.ones(masks.shape[0] if masks.ndim else 0)))
        ids = prediction.get("ids", [None] * (masks.shape[0] if masks.ndim else 0))
        for idx, mask in enumerate(masks):
            if scores[idx] < self.detection_threshold:
                continue
            local_mask = mask >= self.segmentation_threshold
            ys, xs = np.nonzero(local_mask)
            if len(xs) == 0:
                continue
            coords = set(zip((xs + tile["x_offset"]).astype(int), (ys + tile["y_offset"]).astype(int)))
            tile_index = tuple(tile["index"])
            instance = {
                "coords": coords,
                "global_bbox": _xyxy_from_coords(coords),
                "locations": {tile_index: self._get_locations(local_mask, tile)},
                "score": float(scores[idx]),
                "label": int(labels[idx]),
                "tile_ids": [tile_name],
                "ids": [ids[idx]],
            }
            self._instance_registration(instance)

    def _instance_registration(self, instance):
        tile_index = next(iter(instance["locations"]))
        locations = instance["locations"][tile_index]
        if self.disable_merge or "middle" in locations:
            self._add_instance(instance)
            return

        merged_ids = []
        for location in locations:
            adjacent_index, adjacent_location = self._adjacent_for(tile_index, location)
            if adjacent_index is None:
                continue
            merged_id = self._merge_instance(instance, location, adjacent_index, adjacent_location)
            if merged_id >= 0:
                merged_ids.append(merged_id)
        unique_merged_ids = sorted(set(merged_ids))
        if not unique_merged_ids:
            self._add_instance(instance)
        elif len(unique_merged_ids) > 1:
            keeper = unique_merged_ids[0]
            for duplicate in unique_merged_ids[1:]:
                self._merge_existing_instances(keeper, duplicate)

    def _add_instance(self, instance):
        self.instances.append(instance)
        instance_id = len(self.instances) - 1
        for tile_index, locations in instance["locations"].items():
            self._ensure_tile(tile_index)
            for location in locations:
                if instance_id not in self.tiles[tile_index][location]:
                    self.tiles[tile_index][location].append(instance_id)

    def _merge_instance(self, instance, location, adjacent_index, adjacent_location):
        if adjacent_index not in self.tiles:
            return -1
        tile_index = next(iter(instance["locations"]))
        for adjacent_id in list(self.tiles[adjacent_index][adjacent_location]):
            adjacent_instance = self.instances[adjacent_id]
            if adjacent_instance is None:
                continue
            if not self._bbox_intersects(instance["global_bbox"], adjacent_instance["global_bbox"]):
                continue
            mask1 = instance["locations"][tile_index][location]
            mask2 = adjacent_instance["locations"].get(adjacent_index, {}).get(adjacent_location)
            if _set_iou(mask1, mask2 or set()) > self.iou_threshold:
                self._absorb_instance(adjacent_id, instance)
                return adjacent_id
        return -1

    def _absorb_instance(self, existing_id, incoming):
        existing = self.instances[existing_id]
        existing["coords"] = existing["coords"].union(incoming["coords"])
        existing["global_bbox"] = _xyxy_from_coords(existing["coords"])
        existing["score"] = max(existing["score"], incoming["score"])
        existing["tile_ids"].extend(incoming["tile_ids"])
        existing["ids"].extend(incoming["ids"])
        for tile_index, locations in incoming["locations"].items():
            existing_locations = existing["locations"].setdefault(tile_index, {})
            self._ensure_tile(tile_index)
            for location, mask in locations.items():
                existing_locations[location] = existing_locations.get(location, set()).union(mask)
                if existing_id not in self.tiles[tile_index][location]:
                    self.tiles[tile_index][location].append(existing_id)

    def _merge_existing_instances(self, keeper_id, duplicate_id):
        if self.instances[keeper_id] is None or self.instances[duplicate_id] is None:
            return
        self._absorb_instance(keeper_id, self.instances[duplicate_id])
        self.instances[duplicate_id] = None
        for buckets in self.tiles.values():
            for location in buckets:
                buckets[location] = [idx for idx in buckets[location] if idx != duplicate_id]

    def _get_locations(self, local_mask, tile):
        ys, xs = np.nonzero(local_mask)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        width = int(tile["width"])
        height = int(tile["height"])
        overlap = min(self.overlap, width, height)
        tile_index = tuple(tile["index"])
        locations = {}

        if overlap > 0 and x1 < overlap:
            partial = local_mask.copy()
            partial[:, overlap:] = False
            locations["left"] = self._convert_global_mask(partial, tile)
        if overlap > 0 and x2 > width - overlap:
            partial = local_mask.copy()
            partial[:, : width - overlap] = False
            locations["right"] = self._convert_global_mask(partial, tile)
        if overlap > 0 and y1 < overlap:
            partial = local_mask.copy()
            partial[overlap:, :] = False
            locations["top"] = self._convert_global_mask(partial, tile)
        if overlap > 0 and y2 > height - overlap:
            partial = local_mask.copy()
            partial[: height - overlap, :] = False
            locations["bottom"] = self._convert_global_mask(partial, tile)
        if not locations:
            return {"middle": self._convert_global_mask(local_mask, tile)}

        if "left" in locations and "top" in locations:
            partial = local_mask.copy()
            partial[:, overlap:] = False
            partial[overlap:, :] = False
            locations["top-left"] = self._convert_global_mask(partial, tile)
        if "left" in locations and "bottom" in locations:
            partial = local_mask.copy()
            partial[:, overlap:] = False
            partial[: height - overlap, :] = False
            locations["bottom-left"] = self._convert_global_mask(partial, tile)
        if "right" in locations and "top" in locations:
            partial = local_mask.copy()
            partial[:, : width - overlap] = False
            partial[overlap:, :] = False
            locations["top-right"] = self._convert_global_mask(partial, tile)
        if "right" in locations and "bottom" in locations:
            partial = local_mask.copy()
            partial[:, : width - overlap] = False
            partial[: height - overlap, :] = False
            locations["bottom-right"] = self._convert_global_mask(partial, tile)
        return locations

    def _convert_global_mask(self, mask, tile):
        ys, xs = np.nonzero(mask)
        return set(zip((xs + tile["x_offset"]).astype(int), (ys + tile["y_offset"]).astype(int)))

    def _adjacent_for(self, tile_index, location):
        x, y = tile_index
        mapping = {
            "left": ((x - 1, y), "right"),
            "right": ((x + 1, y), "left"),
            "top": ((x, y - 1), "bottom"),
            "bottom": ((x, y + 1), "top"),
            "top-left": ((x - 1, y - 1), "bottom-right"),
            "top-right": ((x + 1, y - 1), "bottom-left"),
            "bottom-left": ((x - 1, y + 1), "top-right"),
            "bottom-right": ((x + 1, y + 1), "top-left"),
        }
        adjacent = mapping.get(location)
        if adjacent is None or adjacent[0] not in self.tile_by_index:
            return None, None
        return adjacent

    def _ensure_tile(self, tile_index):
        if tile_index not in self.tiles:
            self.tiles[tile_index] = {location: [] for location in self.LOCATIONS}

    def _bbox_intersects(self, bbox1, bbox2):
        return bbox1[0] <= bbox2[2] and bbox1[2] >= bbox2[0] and bbox1[1] <= bbox2[3] and bbox1[3] >= bbox2[1]

    def _read_prediction(self, prediction_file):
        if prediction_file.suffix == ".zip":
            with zipfile.ZipFile(prediction_file, "r") as archive:
                names = [name for name in archive.namelist() if name.endswith(".pickle")]
                with archive.open(names[0]) as handle:
                    return pickle.load(handle)
        with prediction_file.open("rb") as handle:
            return pickle.load(handle)
