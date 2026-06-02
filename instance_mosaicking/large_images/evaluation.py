import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _annotation_polygon(annotation):
    segmentation = annotation.get("segmentation", [])
    if not segmentation:
        x, y, w, h = annotation["bbox"]
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    coords = segmentation[0]
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]


def _load_annotations(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("annotations", [])


def IoU(poly1, poly2):
    if hasattr(poly1, "intersection") and hasattr(poly1, "union"):
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union if union else 0.0
    points = list(poly1) + list(poly2)
    minx = int(np.floor(min(x for x, _ in points)))
    miny = int(np.floor(min(y for _, y in points)))
    maxx = int(np.ceil(max(x for x, _ in points)))
    maxy = int(np.ceil(max(y for _, y in points)))
    width = max(1, maxx - minx)
    height = max(1, maxy - miny)
    mask1 = Image.new("1", (width, height), 0)
    mask2 = Image.new("1", (width, height), 0)
    draw1 = ImageDraw.Draw(mask1)
    draw2 = ImageDraw.Draw(mask2)
    draw1.polygon([(x - minx, y - miny) for x, y in poly1], fill=1)
    draw2.polygon([(x - minx, y - miny) for x, y in poly2], fill=1)
    arr1 = np.asarray(mask1, dtype=bool)
    arr2 = np.asarray(mask2, dtype=bool)
    union = np.logical_or(arr1, arr2).sum()
    return float(np.logical_and(arr1, arr2).sum() / union) if union else 0.0


def uniqueness(TP, TN):
    values = TP + TN
    unique, counts = np.unique(values, return_counts=True)
    return [int(value) for value, count in zip(unique, counts) if count > 1]


def evaluate(groundtruth_file, merged_file, IoU_threshold=0.5):
    gt_annotations = _load_annotations(groundtruth_file)
    merged_annotations = _load_annotations(merged_file)
    gt_polygons = [(a.get("id"), _annotation_polygon(a)) for a in gt_annotations]
    matched_gt = set()
    tp = 0
    fp = 0
    matches = []
    for merged in merged_annotations:
        merged_poly = _annotation_polygon(merged)
        best = (None, 0.0)
        for gt_id, gt_poly in gt_polygons:
            score = IoU(gt_poly, merged_poly)
            if score > best[1]:
                best = (gt_id, score)
        if best[1] >= IoU_threshold and best[0] not in matched_gt:
            tp += 1
            matched_gt.add(best[0])
            matches.append({"groundtruth_id": best[0], "merged_id": merged.get("id"), "iou": best[1]})
        else:
            fp += 1
    fn = len(gt_annotations) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "accuracy": accuracy, "matches": matches}
