import fiftyone as fo

dataset = fo.zoo.load_zoo_dataset("open-images-v7", split="validation")

dataset.export(
    export_dir="F://YOLO_Count_Data/OImgv7",
    dataset_type=fo.types.COCODetectionDataset,
)
