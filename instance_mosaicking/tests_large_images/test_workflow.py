import json

from instance_mosaicking.large_images.ellipse_instance_generation import generate_fixture
from instance_mosaicking.large_images.workflow import Workflow


def test_workflow_runs_split_pickle_registration(tmp_path):
    fixture = generate_fixture(tmp_path)
    output = tmp_path / "workflow"
    workflow = Workflow(
        image_path=fixture["image_path"],
        annotation_path=fixture["annotation_path"],
        save_dir=output,
        tile_size=128,
        overlap=32,
        keep_empty_tiles=True,
        detection_threshold=0.0,
    )
    workflow.run_with_annotation_pickles()
    merged = json.loads((output / "merged_instances.json").read_text(encoding="utf-8"))
    assert merged["images"][0]["width"] == 256
    assert len(merged["annotations"]) >= 4
