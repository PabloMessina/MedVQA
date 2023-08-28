import os
import numpy as np
from medvqa.models.checkpoint import get_checkpoint_filepath, load_metadata
from medvqa.models.common import load_model_state_dict
from medvqa.utils.constants import DATASET_NAMES
from medvqa.utils.files import get_cached_pickle_file
from medvqa.utils.logging import print_bold
from medvqa.utils.math import rank_vectors_by_dot_product

def visualize_yolov8_predictions(
        model_name_or_path, checkpoint_folder_path, num_classes, class_names, image_path,
        detection_layer_indexes=None,
        max_det=36, conf_thres=0.1, iou_thres=0.1, figsize=(10, 10),
        dataset_name=DATASET_NAMES.MIMICCXR,
        verbose=True,
    ):
    
    # Device
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if detection_layer_indexes is None:
        from medvqa.models.vision.visual_modules import create_yolov8_model
        # from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_NUM_BBOX_CLASSES, CHEST_IMAGENOME_BBOX_NAMES
        # class_names = {i:x for i, x in enumerate(CHEST_IMAGENOME_BBOX_NAMES)}
        model = create_yolov8_model(
            model_name_or_path=model_name_or_path,
            nc=num_classes,
            class_names=class_names,
            verbose=verbose,
        )
    else:
        from medvqa.models.vision.visual_modules import create_yolov8_model_for_multiple_datasets
        assert isinstance(detection_layer_indexes, list)
        assert len(detection_layer_indexes) > 0
        assert all([isinstance(x, int) for x in detection_layer_indexes])
        assert isinstance(num_classes, list)
        assert isinstance(class_names, list)
        model = create_yolov8_model_for_multiple_datasets(
            model_name_or_path=model_name_or_path,
            nc_list=num_classes,
            class_names_list=class_names,
            verbose=verbose,
        )

    # Load pretrained weights from checkpoint
    pretrained_checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    print(f'pretrained_checkpoint_path = {pretrained_checkpoint_path}')
    checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
    model_weights_dict = checkpoint['model']
    clean_model_weights_dict = {}
    for k, v in model_weights_dict.items(): # HACK: Remove 'raw_image_encoder.' prefix from keys
        if k.startswith('raw_image_encoder.'):
            k = k.replace('raw_image_encoder.', '')
            clean_model_weights_dict[k] = v
    load_model_state_dict(model, clean_model_weights_dict, strict=True)
    print('Checkpoint successfully loaded!')
    
    # Load image
    print(f'image_path = {image_path}')
    from medvqa.datasets.image_processing import get_image_transform
    metadata = load_metadata(checkpoint_folder_path)
    image_transform_kwargs = metadata['val_image_transform_kwargs']
    print(f'image_transform_kwargs = {image_transform_kwargs}')
    transform = get_image_transform(**image_transform_kwargs[dataset_name])
    image, image_size_before, image_size_after = transform(image_path, return_image_size=True)
    print(f'image.shape = {image.shape}')
    print(f'image_size_before = {image_size_before}')
    print(f'image_size_after = {image_size_after}')
    
    # Run model in inference mode
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        if detection_layer_indexes is None:
            _, results = model.custom_forward(image.unsqueeze(0))
            assert len(results) == 2
            print(f'results[0].shape = {results[0].shape}')
            assert len(results[1]) == 3
        else:
            _, results = model.custom_forward(image.unsqueeze(0), detection_layer_indexes=detection_layer_indexes)
            assert len(results) == len(detection_layer_indexes)
            assert len(results[0]) == 2
            print(f'results[0][0].shape = {results[0][0].shape}')
            assert len(results[0][1]) == 3
    
    # Obtain predictions
    from ultralytics.yolo.utils.ops import non_max_suppression
    import numpy as np
    if detection_layer_indexes is None:
        predictions = results[0].detach().cpu()
        predictions = non_max_suppression(predictions, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
        print(f'len(predictions) (after NMS) = {len(predictions)}')
        predictions = predictions[0].numpy()
        print(f'predictions.shape = {predictions.shape}')
        pred_coords = predictions[:, :4]
        pred_coords /= np.array([image_size_after[1], image_size_after[0], image_size_after[1], image_size_after[0]])
        pred_classes = predictions[:, 5].astype(int)
        print(f'pred_coords.shape = {pred_coords.shape}')
        print(f'pred_classes.shape = {pred_classes.shape}')
    else:
        pred_coords_list = []
        pred_classes_list = []
        class_names_list = []
        for i, idx in enumerate(detection_layer_indexes):
            print(f'-------- detection_layer_index = {idx}')
            predictions = results[i][0].detach().cpu()
            predictions = non_max_suppression(predictions, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
            print(f'len(predictions) (after NMS) = {len(predictions)}')
            predictions = predictions[0].numpy()
            print(f'predictions.shape = {predictions.shape}')
            pred_coords = predictions[:, :4]
            pred_coords /= np.array([image_size_after[1], image_size_after[0], image_size_after[1], image_size_after[0]])
            pred_classes = predictions[:, 5].astype(int)
            print(f'pred_coords.shape = {pred_coords.shape}')
            print(f'pred_classes.shape = {pred_classes.shape}')
            pred_coords_list.append(pred_coords)
            pred_classes_list.append(pred_classes)
            class_names_list.append(class_names[idx])
        pred_coords = np.concatenate(pred_coords_list, axis=0)
        pred_classes = np.concatenate(pred_classes_list, axis=0)
        class_names = [item for sublist in class_names_list for item in sublist]
        print('--- after concatenation ---')
        print(f'pred_coords.shape = {pred_coords.shape}')
        print(f'pred_classes.shape = {pred_classes.shape}')
        print(f'len(class_names) = {len(class_names)}')
    
    # Visualize predictions
    from medvqa.evaluation.plots import visualize_predicted_bounding_boxes__yolo
    visualize_predicted_bounding_boxes__yolo(
        image_path=image_path,
        pred_coords=pred_coords,
        pred_classes=pred_classes,
        class_names=class_names,
        figsize=figsize,
        format='xyxy',
    )

    # Release GPU memory
    del model
    del image
    torch.cuda.empty_cache()

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


def load_and_run_seq2seq_model_in_inference_mode(
        input_text, model_folder_path=None, model_checkpoint_path=None, max_output_length=100,
        num_beams=1, use_amp=False, device='GPU'):
    
    from medvqa.models.checkpoint import load_metadata, get_checkpoint_filepath
    from medvqa.models.nlp.seq2seq import Seq2SeqModel, Seq2SeqModels
    from torch.cuda.amp.autocast_mode import autocast
    import torch

    if model_folder_path is None:
        assert model_checkpoint_path is not None
        model_folder_path = os.path.dirname(model_checkpoint_path)
    
    metadata = load_metadata(model_folder_path)
    model_kwargs = metadata['model_kwargs']
    use_t5 = model_kwargs['seq2seq_model_name'] == Seq2SeqModels.T5
    use_bart = model_kwargs['seq2seq_model_name'] == Seq2SeqModels.BART
    
    # device
    print_bold('device = ', device)
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    
    # Create model
    print_bold('Create model')
    model = Seq2SeqModel(**model_kwargs)
    model = model.to(device)

    # Load model weights
    print_bold('Load model weights')
    if model_checkpoint_path is None:
        assert model_folder_path is not None
        model_checkpoint_path = get_checkpoint_filepath(model_folder_path)
    print('model_checkpoint_path = ', model_checkpoint_path)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Prepare input text
    print_bold('Prepare input text')
    if use_t5 or use_bart:
        if use_t5:
            from transformers import T5TokenizerFast
            tokenizer = T5TokenizerFast.from_pretrained(model_kwargs['model_name'])
        else:
            from transformers import BartTokenizerFast
            tokenizer = BartTokenizerFast.from_pretrained(model_kwargs['model_name'])
        assert type(input_text) == str or (type(input_text) == list and type(input_text[0]) == str)
        if type(input_text) == str:
            input_text = [input_text] # To make it a list
        input_encoding = tokenizer(
            input_text,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = input_encoding.input_ids.to(device)
        attention_mask = input_encoding.attention_mask.to(device)
    else:
        raise NotImplementedError
    
    # Run model in inference mode
    print_bold('Run model in inference mode')
    with torch.set_grad_enabled(False):
        model.train(False)
        # Prepare args for model forward
        model_input_kwargs = {
            'mode': 'test',
            'max_len': max_output_length,
        }
        if use_t5 or use_bart:
            model_input_kwargs['input_ids'] = input_ids
            model_input_kwargs['attention_mask'] = attention_mask
            model_input_kwargs['num_beams'] = num_beams
        else:
            raise NotImplementedError

        # Forward pass
        with autocast(enabled=use_amp): # automatic mixed precision
            model_output = model(**model_input_kwargs)
            if use_t5 or use_bart:
                output_ids = model_output
            else:
                raise NotImplementedError

    # Convert ids to string
    print_bold('Convert ids to string')
    if use_t5 or use_bart:
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    else:
        raise NotImplementedError
    
    # Release GPU memory
    del model
    torch.cuda.empty_cache()
    
    # Return predicted text
    assert type(output_text) == list
    if len(output_text) == 1:
        return output_text[0]
    return output_text

def load_and_run_fact_encoder_in_inference_mode(
        facts, chest_imagenome_observations, chest_imagenome_anatomical_locations,
        model_folder_path=None, model_checkpoint_path=None, use_amp=False,
        device='GPU'):
    
    from medvqa.models.checkpoint import load_metadata, get_checkpoint_filepath
    from medvqa.models.nlp.fact_encoder import FactEncoder
    from torch.cuda.amp.autocast_mode import autocast
    from medvqa.datasets.fact_embedding.fact_embedding_dataset_management import (
        _LABEL_TO_CATEGORY, _LABEL_TO_HEALTH_STATUS, _LABEL_TO_COMPARISON_STATUS,
    )
    import torch

    if model_folder_path is None:
        assert model_checkpoint_path is not None
        model_folder_path = os.path.dirname(model_checkpoint_path)
    
    metadata = load_metadata(model_folder_path)
    model_kwargs = metadata['model_kwargs']
    
    # device
    print_bold('device = ', device)
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    
    # Create model
    print_bold('Create model')
    model = FactEncoder(**model_kwargs)
    model = model.to(device)

    # Load model weights
    print_bold('Load model weights')
    if model_checkpoint_path is None:
        assert model_folder_path is not None
        model_checkpoint_path = get_checkpoint_filepath(model_folder_path)
    print('model_checkpoint_path = ', model_checkpoint_path)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Prepare input
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_kwargs['huggingface_model_name'], trust_remote_code=True)
    assert type(facts) == list
    assert len(facts) >= 2
    assert all(type(x) == str for x in facts)
    input_encoding = tokenizer(
        facts,
        padding="longest",
        return_tensors="pt",
    )
    input_ids = input_encoding.input_ids.to(device)
    attention_mask = input_encoding.attention_mask.to(device)
    
    # Run model in inference mode
    print_bold('Run model in inference mode')
    with torch.set_grad_enabled(False):
        model.train(False)
        with autocast(enabled=use_amp): # automatic mixed precision
            model_output = model(input_ids=input_ids, attention_mask=attention_mask,
                                 run_metadata_auxiliary_tasks=True,
                                 run_chest_imagenome_obs_task=True,
                                 run_chest_imagenome_anatloc_task=True)
    
    embeddings = model_output['text_embeddings'].detach().cpu().numpy()
    c_logits = model_output['category_logits'].detach().cpu()
    hs_logits = model_output['health_status_logits'].detach().cpu()
    cs_logits = model_output['comparison_status_logits'].detach().cpu()
    cio_logits = model_output['chest_imagenome_obs_logits'].detach().cpu()
    cia_logits = model_output['chest_imagenome_anatloc_logits'].detach().cpu()
    print(f'embeddings.shape = {embeddings.shape}')
    print(f'c_logits.shape = {c_logits.shape}')
    print(f'hs_logits.shape = {hs_logits.shape}')
    print(f'cs_logits.shape = {cs_logits.shape}')
    print(f'cio_logits.shape = {cio_logits.shape}')
    print(f'cia_logits.shape = {cia_logits.shape}')

    ranked_indexes = rank_vectors_by_dot_product(embeddings[0], embeddings)
    pred_c = c_logits.argmax(dim=1).numpy()
    pred_hs = hs_logits.argmax(dim=1).numpy()
    pred_cs = cs_logits.argmax(dim=1).numpy()
    pred_cio = (cio_logits > 0).numpy().astype(int)
    pred_cia = (cia_logits > 0).numpy().astype(int)
    assert pred_cio.shape == (len(facts), len(chest_imagenome_observations))
    assert pred_cia.shape == (len(facts), len(chest_imagenome_anatomical_locations))
    
    print_bold(f'Query: {facts[0]}')
    print(f'\tCategory: {_LABEL_TO_CATEGORY[pred_c[0]]}')
    print(f'\tHealth status: {_LABEL_TO_HEALTH_STATUS[pred_hs[0]]}')
    print(f'\tComparison status: {_LABEL_TO_COMPARISON_STATUS[pred_cs[0]]}')
    print('\tChest Imagenome observations:')
    for j in range(len(chest_imagenome_observations)):
        if pred_cio[0, j] == 1:
            print(f'\t\t{chest_imagenome_observations[j]}')
    print('\tChest Imagenome anatomical locations:')
    for j in range(len(chest_imagenome_anatomical_locations)):
        if pred_cia[0, j] == 1:
            print(f'\t\t{chest_imagenome_anatomical_locations[j]}')
    print('----------------')
    for i, idx in enumerate(ranked_indexes):
        print_bold(f'Fact {i}: {facts[idx]}')
        print(f'\tCategory: {_LABEL_TO_CATEGORY[pred_c[idx]]}')
        print(f'\tHealth status: {_LABEL_TO_HEALTH_STATUS[pred_hs[idx]]}')
        print(f'\tComparison status: {_LABEL_TO_COMPARISON_STATUS[pred_cs[idx]]}')
        print('\tChest Imagenome observations:')
        for j in range(len(chest_imagenome_observations)):
            if pred_cio[idx, j] == 1:
                print(f'\t\t{chest_imagenome_observations[j]}')
        print('\tChest Imagenome anatomical locations:')
        for j in range(len(chest_imagenome_anatomical_locations)):
            if pred_cia[idx, j] == 1:
                print(f'\t\t{chest_imagenome_anatomical_locations[j]}')
    
    # Release GPU memory
    del model
    del input_ids
    del attention_mask
    del embeddings
    del c_logits
    del hs_logits
    del cs_logits
    del model_output
    import gc
    gc.collect()
    torch.cuda.empty_cache()