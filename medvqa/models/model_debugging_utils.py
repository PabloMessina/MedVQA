def visualize_chest_imagenome_yolov8_predictions(
        model_name_or_path, dicom_id, image_size, conf_thres=0.25, iou_thres=0.45, figsize=(10, 10),
    ):
    # Load model
    from medvqa.models.vision.visual_modules import create_yolov8_model
    from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_NUM_BBOX_CLASSES, CHEST_IMAGENOME_BBOX_NAMES
    class_names = {i:x for i, x in enumerate(CHEST_IMAGENOME_BBOX_NAMES)}
    model = create_yolov8_model(
        model_name_or_path=model_name_or_path,
        nc=CHEST_IMAGENOME_NUM_BBOX_CLASSES,
        class_names=class_names,
    )
    # Load image
    from medvqa.datasets.mimiccxr import get_imageId2PartPatientStudy, get_mimiccxr_medium_image_path
    imageId2PartPatientStudy = get_imageId2PartPatientStudy()    
    part_id, patient_id, study_id = imageId2PartPatientStudy[dicom_id]
    image_path = get_mimiccxr_medium_image_path(part_id, patient_id, study_id, dicom_id)
    print(f'dicom_id = {dicom_id}')
    print(f'image_path = {image_path}')
    from medvqa.datasets.image_processing import get_image_transform
    transform = get_image_transform(
        image_size=image_size,
        use_bbox_aware_transform=True,
    )
    image, image_size_before, image_size_after = transform(image_path, return_image_size=True)
    print(f'image.shape = {image.shape}')
    print(f'image_size_before = {image_size_before}')
    print(f'image_size_after = {image_size_after}')
    # Run model in inference mode
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        results = model(image.unsqueeze(0))
        assert len(results) == 2
        print(f'results[0].shape = {results[0].shape}')
        assert len(results[1]) == 3
    # Obtain predictions
    from ultralytics.yolo.utils.ops import non_max_suppression
    import numpy as np
    predictions = results[0].detach().cpu()
    predictions = non_max_suppression(predictions, conf_thres=conf_thres, iou_thres=iou_thres,
                                      max_det=CHEST_IMAGENOME_NUM_BBOX_CLASSES)
    print(f'len(predictions) (after NMS) = {len(predictions)}')
    predictions = predictions[0].numpy()
    print(f'predictions.shape = {predictions.shape}')
    pred_coords = predictions[:, :4]
    pred_coords /= np.array([image_size_after[1], image_size_after[0], image_size_after[1], image_size_after[0]])
    pred_classes = predictions[:, 5].astype(int)
    print(f'pred_coords.shape = {pred_coords.shape}')
    print(f'pred_classes.shape = {pred_classes.shape}')
    # Visualize predictions
    from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import _visualize_predicted_bounding_boxes__yolo    
    _visualize_predicted_bounding_boxes__yolo(
        dicom_id=dicom_id,
        pred_coords=pred_coords,
        pred_classes=pred_classes,
        figsize=figsize,
        format='xyxy',
    )
    # Release GPU memory
    del model
    del image
    torch.cuda.empty_cache()
    return {
        'dicom_id': dicom_id,
        'pred_coords': pred_coords,
        'pred_classes': pred_classes,
    }