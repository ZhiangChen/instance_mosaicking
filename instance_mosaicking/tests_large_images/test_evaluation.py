from instance_mosaicking.large_images.evaluation import IoU, evaluate, uniqueness
from instance_mosaicking.large_images.ellipse_instance_generation import generate_fixture


def test_iou_and_uniqueness():
    a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    b = [(5, 0), (15, 0), (15, 10), (5, 10)]
    assert round(IoU(a, b), 2) == 0.40
    assert uniqueness([1, 2], [2, 3]) == [2]


def test_evaluate_coco_files(tmp_path):
    fixture = generate_fixture(tmp_path)
    metrics = evaluate(fixture["annotation_path"], fixture["annotation_path"], IoU_threshold=0.9)
    assert metrics["TP"] == 4
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0
