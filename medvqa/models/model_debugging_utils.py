import os
import numpy as np
from medvqa.utils.files import get_cached_pickle_file
from medvqa.utils.logging import print_bold

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

def get_gt_labels_for_dicom_id(dicom_id, use_chexpert=True, use_chest_imagenome=True,
                               chexpert_labels_filename=None, chest_imagenome_labels_filename=None,
                               apply_anatomycal_reordering=False, chest_imagenome_label_names_filename=None,
                               verbose=True, chexpert_labels_to_drop=None, chexpert_labels_to_add=None,
                               chest_imagenome_labels_to_drop=None, chest_imagenome_labels_to_add=None,):
    assert use_chexpert or use_chest_imagenome

    from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import (
        load_chest_imagenome_label_names,
        load_chest_imagenome_label_order,
        load_chest_imagenome_labels,
    )
    from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR, get_detailed_metadata_for_dicom_id
    from medvqa.utils.metrics import chest_imagenome_label_array_to_string, chexpert_label_array_to_string
    from medvqa.utils.constants import CHEXPERT_LABELS
    
    if use_chexpert:
        assert chexpert_labels_filename is not None
        metadata = get_detailed_metadata_for_dicom_id(dicom_id)
        assert len(metadata) == 1
        report_index = metadata[0]['report_index']
        chexpert_labels_path = os.path.join(MIMICCXR_CACHE_DIR, chexpert_labels_filename)
        chexpert_labels = get_cached_pickle_file(chexpert_labels_path)
        chexpert_labels = chexpert_labels[report_index]
        if chexpert_labels_to_drop is not None or chexpert_labels_to_add is not None:
            chexpert_labels = chexpert_labels.copy() # To avoid modifying the original array
        if chexpert_labels_to_drop is not None:
            for x in chexpert_labels_to_drop:
                chexpert_labels[CHEXPERT_LABELS.index(x)] = 0
        if chexpert_labels_to_add is not None:
            for x in chexpert_labels_to_add:
                chexpert_labels[CHEXPERT_LABELS.index(x)] = 1
        if verbose:
            print_bold('chexpert_labels:')
            print(chexpert_label_array_to_string(chexpert_labels))

    if use_chest_imagenome:
        assert chest_imagenome_labels_filename is not None
        chest_imagenome_labels = load_chest_imagenome_labels(chest_imagenome_labels_filename)
        chest_imagenome_labels = chest_imagenome_labels[dicom_id]
        if verbose:
            assert chest_imagenome_label_names_filename is not None
            label_names = load_chest_imagenome_label_names(chest_imagenome_label_names_filename)
        if apply_anatomycal_reordering:
            assert chest_imagenome_label_names_filename is not None
            label_order = load_chest_imagenome_label_order(chest_imagenome_label_names_filename)
            assert len(label_order) == len(chest_imagenome_labels)
            chest_imagenome_labels = chest_imagenome_labels[label_order]
            if verbose:
                assert len(label_names) == len(chest_imagenome_labels)
                label_names = [label_names[i] for i in label_order]
        if chest_imagenome_labels_to_drop is not None or chest_imagenome_labels_to_add is not None:
            chest_imagenome_labels = chest_imagenome_labels.copy() # To avoid modifying the original array
        if chest_imagenome_labels_to_drop is not None:
            for x in chest_imagenome_labels_to_drop:
                chest_imagenome_labels[label_names.index(x)] = 0
        if chest_imagenome_labels_to_add is not None:
            for x in chest_imagenome_labels_to_add:
                chest_imagenome_labels[label_names.index(x)] = 1
        if verbose:
            print_bold('chest_imagenome_labels:')
            print(chest_imagenome_label_array_to_string(chest_imagenome_labels, label_names))

    if use_chexpert and use_chest_imagenome:
        return np.concatenate([chexpert_labels, chest_imagenome_labels])
    elif use_chexpert:
        return chexpert_labels
    elif use_chest_imagenome:
        return chest_imagenome_labels
    else: assert False

def load_and_run_labels2report_gen_model_in_inference_mode(
        input_binary_labels, model_folder_path, is_second_label_source=False, max_report_length=100,
        num_beams=1, use_amp=False, device='GPU'):
    
    from medvqa.models.checkpoint import load_metadata
    from medvqa.datasets.tokenizer import Tokenizer
    from medvqa.models.report_generation.labels2report import Labels2ReportModel
    from medvqa.models.checkpoint import get_checkpoint_filepath
    from medvqa.models.report_generation.labels2report import NLG_Models
    from torch.cuda.amp.autocast_mode import autocast
    import torch
    
    metadata = load_metadata(model_folder_path)
    tokenizer_kwargs = metadata['tokenizer_kwargs']
    model_kwargs = metadata['model_kwargs']
    use_t5 = model_kwargs['nlg_model'] == NLG_Models.T5
    
    # device
    print_bold('device = ', device)
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')

    # Init tokenizer
    print_bold('Create tokenizer')
    tokenizer = Tokenizer(**tokenizer_kwargs)
    
    # Create model
    print_bold('Create model')
    model = Labels2ReportModel(vocab_size=tokenizer.vocab_size,
                            start_idx=tokenizer.token2id[tokenizer.START_TOKEN],
                            device=device, **model_kwargs)
    model = model.to(device)

    # Load model weights
    print_bold('Load model weights')
    checkpoint_path = get_checkpoint_filepath(model_folder_path)
    print('checkpoint_path = ', checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Prepare input labels
    print_bold('Prepare input labels')
    if use_t5:
        from transformers import T5Tokenizer
        t5_tokenizer = T5Tokenizer.from_pretrained(model_kwargs['t5_model_name'])
        assert type(input_binary_labels) == str
        input_binary_labels = [input_binary_labels] # To make it a list
        input_encoding = t5_tokenizer(
            input_binary_labels,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = input_encoding.input_ids.to(device)
        attention_mask = input_encoding.attention_mask.to(device)
    else:
        input_binary_labels = torch.tensor(input_binary_labels, dtype=torch.float32).to(device)
        input_binary_labels = input_binary_labels.unsqueeze(0)
    
    # Run model in inference mode
    print_bold('Run model in inference mode')
    with torch.set_grad_enabled(False):
        model.train(False)
        # Prepare args for model forward
        model_input_kwargs = {                
            'device': device,
            'mode': 'test',
            'max_report_length': max_report_length,
        }
        if use_t5:
            model_input_kwargs['input_ids'] = input_ids
            model_input_kwargs['attention_mask'] = attention_mask
            model_input_kwargs['num_beams'] = num_beams
        else:
            model_input_kwargs['predicted_binary_scores'] = input_binary_labels
            model_input_kwargs['is_second_label_source'] = is_second_label_source

        # Forward pass
        with autocast(enabled=use_amp): # automatic mixed precision
            model_output = model(**model_input_kwargs)
            if use_t5:
                pred_reports = model_output
            else:
                pred_reports = model_output['pred_reports']

    # Convert predicted report from ids to string
    print_bold('Convert predicted report from ids to string')
    pred_report = pred_reports[0]
    if use_t5:
        pred_report = t5_tokenizer.decode(pred_report, skip_special_tokens=True)
    else:
        pred_report = tokenizer.ids2string(tokenizer.clean_sentence(pred_report))
    
    # Return predicted report
    return pred_report