import json
from pathlib import Path

from instance_mosaicking.large_images.ellipse_instance_generation import generate_fixture


def test_generate_fixture_writes_image_and_coco(tmp_path):
    fixture = generate_fixture(tmp_path)
    assert Path(fixture["image_path"]).exists()
    annotation_path = Path(fixture["annotation_path"])
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert data["images"][0]["width"] == 256
    assert len(data["annotations"]) == 4
    assert len(data["annotations"][0]["segmentation"][0]) == 96
    assert data["annotations"][0]["bbox"] == [24, 24, 52, 52]
