import os
import argparse
from ultralytics import YOLO
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_CACHE_DIR
from medvqa.datasets.vinbig import VINBIG_CACHE_DIR
from medvqa.train_yolov5 import prepare_train_val_test_data__chest_imagenome, prepare_train_val_test_data__vindrcxr

from medvqa.utils.common import parsed_args_to_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='yolov11n.pt')
    parser.add_argument('--decent_images_only', action='store_true')
    parser.add_argument('--validation_only', action='store_true')
    parser.add_argument('--image_size', type=int, default=416)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--dataset', type=str, default='chest_imagenome', choices=['chest_imagenome', 'vindr_cxr'])    
    return parser.parse_args()

def train_yolov11(
        model_name_or_path,
        decent_images_only,
        validation_only,
        image_size,
        batch_size,
        workers,
        epochs,
        warmup_epochs,
        resume,
        debug,
        device,
        dataset,
    ):
    if dataset == 'chest_imagenome':
        data_info = prepare_train_val_test_data__chest_imagenome(
            decent_images_only=decent_images_only,
            validation_only=validation_only,
            debug=debug,
        )
        project_path = os.path.join(CHEST_IMAGENOME_CACHE_DIR, 'yolov11')
    elif dataset == 'vindr_cxr':
        data_info = prepare_train_val_test_data__vindrcxr(validation_only=validation_only)
        project_path = os.path.join(VINBIG_CACHE_DIR, 'yolov11')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    model = YOLO(model=model_name_or_path, task='detect')
    model.train(
        data=data_info['yaml_config_path'],
        epochs=epochs,
        batch=batch_size,
        workers=workers,
        warmup_epochs=warmup_epochs,
        imgsz=image_size,        
        project=project_path,
        resume=resume,
        device=device,
    )

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    train_yolov11(**args)