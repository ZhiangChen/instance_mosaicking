import json
import pickle

import numpy as np
from PIL import Image

from instance_mosaicking.large_images.annotation_map_split import Dataset, Tile_Splitter
from instance_mosaicking.large_images.ellipse_instance_generation import generate_fixture
from instance_mosaicking.large_images.instance_registration import Instance_Registration


def _write_registration_fixture(tmp_path, tiles, predictions, image_size=(200, 120), overlap=20):
    tmp_path.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image_size, (255, 255, 255)).save(tmp_path / "source.png")
    metadata = {
        "source_image": "source.png",
        "image_width": image_size[0],
        "image_height": image_size[1],
        "tile_size": 100,
        "overlap": overlap,
        "tiles": tiles,
    }
    (tmp_path / "tiles.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for tile_name, masks, ids in predictions:
        data = {
            "image_name": tile_name,
            "masks": np.asarray(masks, dtype=np.uint8),
            "boxes": np.zeros((len(masks), 4), dtype=np.float32),
            "labels": np.ones(len(masks), dtype=np.int64),
            "scores": np.ones(len(masks), dtype=np.float32),
            "ids": ids,
        }
        with (tmp_path / f"{tile_name}.pickle").open("wb") as handle:
            pickle.dump(data, handle)


def _mask(width, height, x1, y1, x2, y2):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def test_registration_converts_to_global_pixels_and_exports(tmp_path):
    fixture = generate_fixture(tmp_path)
    output = tmp_path / "split"
    Tile_Splitter(fixture["image_path"], output, 128, 32, fixture["annotation_path"], keep_empty_tiles=True).split()
    Dataset(output).save_pickles()
    registration = Instance_Registration(output, save_dir=output, detection_threshold=0.0, iou_threshold=0.5)
    registration.start_registration()
    merged = json.loads((output / "merged_instances.json").read_text(encoding="utf-8"))
    assert len(merged["annotations"]) >= 4
    assert (output / "merged_pixel_polygons.json").exists()
    assert (output / "merged_instance_mask.png").exists()
    assert (output / "merged_overlay.png").exists()


def test_adjacent_overlap_fragments_merge_by_overlap_iou(tmp_path):
    tiles = [
        {"name": "0_0.png", "index": [0, 0], "x_offset": 0, "y_offset": 0, "width": 100, "height": 100},
        {"name": "1_0.png", "index": [1, 0], "x_offset": 80, "y_offset": 0, "width": 100, "height": 100},
    ]
    predictions = [
        ("0_0", [_mask(100, 100, 85, 20, 100, 40)], ["left_piece"]),
        ("1_0", [_mask(100, 100, 5, 20, 20, 40)], ["right_piece"]),
    ]
    _write_registration_fixture(tmp_path, tiles, predictions)
    Instance_Registration(tmp_path, save_dir=tmp_path, detection_threshold=0.0, iou_threshold=0.5).start_registration()
    merged = json.loads((tmp_path / "merged_instances.json").read_text(encoding="utf-8"))
    assert len(merged["annotations"]) == 1
    assert set(merged["annotations"][0]["ids"]) == {"left_piece", "right_piece"}


def test_adjacent_overlap_fragments_do_not_merge_below_iou_threshold(tmp_path):
    tiles = [
        {"name": "0_0.png", "index": [0, 0], "x_offset": 0, "y_offset": 0, "width": 100, "height": 100},
        {"name": "1_0.png", "index": [1, 0], "x_offset": 80, "y_offset": 0, "width": 100, "height": 100},
    ]
    predictions = [
        ("0_0", [_mask(100, 100, 85, 20, 100, 40)], ["same_source"]),
        ("1_0", [_mask(100, 100, 10, 20, 25, 40)], ["same_source"]),
    ]
    _write_registration_fixture(tmp_path, tiles, predictions)
    Instance_Registration(tmp_path, save_dir=tmp_path, detection_threshold=0.0, iou_threshold=0.9).start_registration()
    merged = json.loads((tmp_path / "merged_instances.json").read_text(encoding="utf-8"))
    assert len(merged["annotations"]) == 2


def test_non_adjacent_tiles_are_not_compared_for_merging(tmp_path):
    tiles = [
        {"name": "0_0.png", "index": [0, 0], "x_offset": 0, "y_offset": 0, "width": 100, "height": 100},
        {"name": "2_0.png", "index": [2, 0], "x_offset": 0, "y_offset": 0, "width": 100, "height": 100},
    ]
    predictions = [
        ("0_0", [_mask(100, 100, 85, 20, 100, 40)], ["a"]),
        ("2_0", [_mask(100, 100, 85, 20, 100, 40)], ["b"]),
    ]
    _write_registration_fixture(tmp_path, tiles, predictions)
    Instance_Registration(tmp_path, save_dir=tmp_path, detection_threshold=0.0, iou_threshold=0.5).start_registration()
    merged = json.loads((tmp_path / "merged_instances.json").read_text(encoding="utf-8"))
    assert len(merged["annotations"]) == 2


def test_middle_instances_are_retained_without_merge_attempts(tmp_path):
    tiles = [
        {"name": "0_0.png", "index": [0, 0], "x_offset": 0, "y_offset": 0, "width": 100, "height": 100},
    ]
    predictions = [("0_0", [_mask(100, 100, 35, 35, 55, 55)], ["middle"])]
    _write_registration_fixture(tmp_path, tiles, predictions)
    registration = Instance_Registration(tmp_path, save_dir=tmp_path, detection_threshold=0.0, iou_threshold=0.5)
    registration.start_registration()
    merged = json.loads((tmp_path / "merged_instances.json").read_text(encoding="utf-8"))
    assert len(merged["annotations"]) == 1
    assert registration.tiles[(0, 0)]["middle"] == [0]
