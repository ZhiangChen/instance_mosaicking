from pathlib import Path

from .annotation_map_split import Dataset, Tile_Splitter
from .instance_registration import Instance_Registration


class Workflow:
    def __init__(
        self,
        image_path,
        save_dir,
        tile_size,
        overlap,
        annotation_path=None,
        annotation_format="coco",
        iou_threshold=0.5,
        detection_threshold=0.75,
        segmentation_threshold=0.5,
        keep_empty_tiles=False,
        zip=False,
    ):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.save_dir = Path(save_dir)
        self.tile_size = tile_size
        self.overlap = overlap
        self.annotation_format = annotation_format
        self.iou_threshold = iou_threshold
        self.detection_threshold = detection_threshold
        self.segmentation_threshold = segmentation_threshold
        self.keep_empty_tiles = keep_empty_tiles
        self.zip = zip

    def split(self):
        splitter = Tile_Splitter(
            image_path=self.image_path,
            save_dir=self.save_dir,
            tile_size=self.tile_size,
            overlap=self.overlap,
            annotation_path=self.annotation_path,
            annotation_format=self.annotation_format,
            keep_empty_tiles=self.keep_empty_tiles,
        )
        return splitter.split()

    def save_training_pickles(self):
        dataset = Dataset(self.save_dir)
        return dataset.save_pickles(zip_output=self.zip)

    def register_predictions(self, prediction_dir=None):
        prediction_dir = Path(prediction_dir) if prediction_dir else self.save_dir
        registration = Instance_Registration(
            instance_dir=prediction_dir,
            save_dir=self.save_dir,
            tiles_metadata=self.save_dir / "tiles.json",
            detection_threshold=self.detection_threshold,
            segmentation_threshold=self.segmentation_threshold,
            iou_threshold=self.iou_threshold,
        )
        return registration.start_registration()

    def run_with_annotation_pickles(self):
        self.split()
        self.save_training_pickles()
        return self.register_predictions(self.save_dir)
