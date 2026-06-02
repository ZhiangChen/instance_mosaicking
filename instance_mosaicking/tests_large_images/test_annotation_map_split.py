import json

import numpy as np

from instance_mosaicking.large_images.annotation_map_split import Dataset, Tile_Splitter
from instance_mosaicking.large_images.ellipse_instance_generation import generate_fixture


def test_tile_splitter_writes_metadata_and_local_annotations(tmp_path):
    fixture = generate_fixture(tmp_path)
    output = tmp_path / "split"
    splitter = Tile_Splitter(
        fixture["image_path"],
        output,
        tile_size=128,
        overlap=32,
        annotation_path=fixture["annotation_path"],
        keep_empty_tiles=True,
    )
    metadata = splitter.split()
    assert len(metadata["tiles"]) == 9
    assert metadata["tiles"][1]["x_offset"] == 96
    tile_annotations = json.loads((output / "tile_annotations.json").read_text(encoding="utf-8"))
    assert len(tile_annotations["images"]) == 9
    assert any(a["source_annotation_id"] == 2 for a in tile_annotations["annotations"])


def test_dataset_returns_masks_and_saves_pickles(tmp_path):
    fixture = generate_fixture(tmp_path)
    output = tmp_path / "split"
    Tile_Splitter(fixture["image_path"], output, 128, 32, fixture["annotation_path"], keep_empty_tiles=True).split()
    dataset = Dataset(output)
    image, target = dataset[0]
    assert image.shape[:2] == (128, 128)
    assert target["boxes"].shape[1] == 4
    paths = dataset.save_pickles()
    assert len(paths) == len(dataset)


def test_png_mask_annotation_split_preserves_instance_masks(tmp_path):
    fixture = generate_fixture(tmp_path)
    output = tmp_path / "mask_png_split"
    metadata = Tile_Splitter(
        fixture["image_path"],
        output,
        128,
        32,
        fixture["mask_path"],
        annotation_format="mask",
        keep_empty_tiles=True,
    ).split()
    tile_annotations = json.loads((output / "tile_annotations.json").read_text(encoding="utf-8"))
    assert len(metadata["tiles"]) == 9
    assert (output / "tile_masks" / "0_0.png").exists()
    assert all("mask_file" in annotation for annotation in tile_annotations["annotations"])
    assert any(annotation["source_annotation_id"] == 1 for annotation in tile_annotations["annotations"])

    dataset = Dataset(output)
    _, target = dataset[0]
    assert target["masks"].shape[1:] == (128, 128)
    assert set(np.unique(target["masks"]).tolist()).issubset({0, 1})
    assert 1 in target["ids"]
    assert target["boxes"].shape[1] == 4


def test_tif_mask_annotation_split_preserves_instance_masks(tmp_path):
    fixture = generate_fixture(tmp_path)
    output = tmp_path / "mask_tif_split"
    Tile_Splitter(
        fixture["image_path"],
        output,
        128,
        32,
        fixture["mask_tif_path"],
        annotation_format="mask",
        keep_empty_tiles=True,
    ).split()
    tile_annotations = json.loads((output / "tile_annotations.json").read_text(encoding="utf-8"))
    assert (output / "tile_masks" / "0_0.tif").exists()
    assert any(annotation["mask_file"].endswith(".tif") for annotation in tile_annotations["annotations"])

    dataset = Dataset(output)
    _, target = dataset[0]
    assert target["masks"].sum() > 0


def test_mask_annotation_can_skip_empty_tiles(tmp_path):
    fixture = generate_fixture(tmp_path, ellipse_count=1)
    output = tmp_path / "mask_sparse_split"
    metadata = Tile_Splitter(
        fixture["image_path"],
        output,
        128,
        32,
        fixture["mask_path"],
        annotation_format="mask",
        keep_empty_tiles=False,
    ).split()
    assert len(metadata["tiles"]) == 1
    assert metadata["tiles"][0]["name"] == "0_0.png"
