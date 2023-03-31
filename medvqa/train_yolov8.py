import argparse
from ultralytics import YOLO
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_FAST_CACHE_DIR
from medvqa.train_yolov5 import prepare_train_val_test_data__chest_imagenome

from medvqa.utils.common import parsed_args_to_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str, default='yolov8n.pt')
    parser.add_argument('--decent-images-only', action='store_true', default=False)
    parser.add_argument('--validation-only', action='store_true', default=False)
    parser.add_argument('--image-size', type=int, default=416)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--debug', action='store_true', default=False)
    return parser.parse_args()

def train_yolov8(
        model_name_or_path,
        decent_images_only,
        validation_only,
        image_size,
        batch_size,
        workers,
        epochs,
        debug,
    ):
    data_info = prepare_train_val_test_data__chest_imagenome(
        decent_images_only=decent_images_only,
        validation_only=validation_only,
        debug=debug,
    )
    model = YOLO(model=model_name_or_path, task='detect')
    model.train(
        data=data_info['yaml_config_path'],
        epochs=epochs,
        batch=batch_size,
        workers=workers,
        imgsz=image_size,
        project=CHEST_IMAGENOME_FAST_CACHE_DIR, # TODO: maybe consider different folders for different datasets
    )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    train_yolov8(**args)