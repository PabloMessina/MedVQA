import os
import numpy as np
from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_BBOX_NAMES
from medvqa.models.checkpoint import get_checkpoint_filepath, load_metadata, load_model_state_dict
from medvqa.utils.constants import DATASET_NAMES, VINBIG_BBOX_NAMES
from medvqa.utils.files_utils import get_cached_pickle_file, load_pickle
from medvqa.utils.logging_utils import print_bold
from medvqa.utils.math_utils import rank_vectors_by_dot_product

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
    from medvqa.utils.metrics_utils import chest_imagenome_label_array_to_string, chexpert_label_array_to_string
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
        num_beams=1, use_amp=False, device='cuda'):
    
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
    use_flan_t5 = model_kwargs['seq2seq_model_name'] == Seq2SeqModels.FLAN_T5
    use_bart = model_kwargs['seq2seq_model_name'] == Seq2SeqModels.BART
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    print_bold('device = ', device)
    
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
    if use_t5 or use_flan_t5 or use_bart:
        if use_t5:
            from transformers import T5TokenizerFast
            tokenizer = T5TokenizerFast.from_pretrained(model_kwargs['model_name'])
        elif use_flan_t5:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_kwargs['model_name'])
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
        if use_t5 or use_flan_t5 or use_bart:
            model_input_kwargs['input_ids'] = input_ids
            model_input_kwargs['attention_mask'] = attention_mask
            model_input_kwargs['num_beams'] = num_beams
        else:
            raise NotImplementedError

        # Forward pass
        with autocast(enabled=use_amp): # automatic mixed precision
            model_output = model(**model_input_kwargs)
            if use_t5 or use_flan_t5 or use_bart:
                output_ids = model_output
            else:
                raise NotImplementedError

    # Convert ids to string
    print_bold('Convert ids to string')
    if use_t5 or use_flan_t5 or use_bart:
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    else:
        raise NotImplementedError
    
    # Release GPU memory
    del model
    torch.cuda.empty_cache()
    
    # Return predicted text
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

def run_fact_encoder_nli(
        premises, hypotheses, model_folder_path=None, model_checkpoint_path=None, use_amp=False,
        device='GPU'):
    assert len(premises) == len(hypotheses)
    
    from medvqa.models.checkpoint import load_metadata, get_checkpoint_filepath
    from medvqa.models.nlp.fact_encoder import FactEncoder
    from torch.cuda.amp.autocast_mode import autocast
    from medvqa.datasets.nli.nli_dataset_management import _INDEX_TO_LABEL
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
    p_encoding = tokenizer(premises, padding="longest", return_tensors="pt")
    h_encoding = tokenizer(hypotheses, padding="longest", return_tensors="pt")
    p_input_ids = p_encoding.input_ids.to(device)
    p_attention_mask = p_encoding.attention_mask.to(device)
    h_input_ids = h_encoding.input_ids.to(device)
    h_attention_mask = h_encoding.attention_mask.to(device)
    
    # Run model in inference mode
    print_bold('Run model in inference mode')
    with torch.set_grad_enabled(False):
        model.train(False)
        with autocast(enabled=use_amp): # automatic mixed precision
            logits = model.nli_forward(p_input_ids=p_input_ids, p_attention_mask=p_attention_mask,
                                       h_input_ids=h_input_ids, h_attention_mask=h_attention_mask)
    
    # Convert logits to labels
    print_bold('Convert logits to labels')
    labels = logits.argmax(dim=1).detach().cpu().numpy()
    assert labels.shape == (len(premises),)
    labels = [_INDEX_TO_LABEL[x] for x in labels]
    
    # Print results
    print_bold('Results')
    for i in range(len(premises)):
        print(f'Premise: {premises[i]}')
        print(f'Hypothesis: {hypotheses[i]}')
        print(f'Label: {labels[i]}')
        print('----------------')
    
    # Release GPU memory
    del model
    del p_input_ids
    del p_attention_mask
    del h_input_ids
    del h_attention_mask
    del logits
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def run_nli_model(
        premises, hypotheses, model_folder_path=None, model_checkpoint_path=None, use_amp=False,
        device='GPU', use_precomputed_embeddings=False):
    assert len(premises) == len(hypotheses)
    
    from medvqa.models.checkpoint import load_metadata, get_checkpoint_filepath
    from medvqa.models.nlp.nli import BertBasedNLI
    from torch.cuda.amp.autocast_mode import autocast
    from medvqa.datasets.nli.nli_dataset_management import _INDEX_TO_LABEL
    import torch

    if model_folder_path is None:
        assert model_checkpoint_path is not None
        model_folder_path = os.path.dirname(model_checkpoint_path)
    
    metadata = load_metadata(model_folder_path)
    model_kwargs = metadata['model_kwargs']
    if 'hidden_size' not in model_kwargs:
        assert 'nli_hidden_layer_size' in model_kwargs
        model_kwargs['hidden_size'] = model_kwargs['nli_hidden_layer_size'] # for compatibility with other models
    
    # device
    print_bold('device = ', device)
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
    
    # Create model
    print_bold('Create model')
    model = BertBasedNLI(**model_kwargs)
    model = model.to(device)

    # Load model weights
    print_bold('Load model weights')
    if model_checkpoint_path is None:
        assert model_folder_path is not None
        model_checkpoint_path = get_checkpoint_filepath(model_folder_path)
    print('model_checkpoint_path = ', model_checkpoint_path)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    load_model_state_dict(model, checkpoint['model'])

    if use_precomputed_embeddings:
        print_bold('Use precomputed embeddings')
        from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
        ctee = CachedTextEmbeddingExtractor(
            model_name=model_kwargs['huggingface_model_name'],
            model_checkpoint_folder_path=model_folder_path,
        )
        texts = premises + hypotheses
        embeddings = ctee.compute_text_embeddings(texts)
        p_embeddings = embeddings[:len(premises)]
        h_embeddings = embeddings[len(premises):]
        print(f'p_embeddings.shape = {p_embeddings.shape}')
        print(f'h_embeddings.shape = {h_embeddings.shape}')
        with torch.set_grad_enabled(False):
            model.train(False)
            with autocast(enabled=use_amp): # automatic mixed precision
                p_embeddings = torch.tensor(p_embeddings, dtype=torch.float32).to(device)
                h_embeddings = torch.tensor(h_embeddings, dtype=torch.float32).to(device)
                logits = model.forward_with_precomputed_embeddings(p_embeddings, h_embeddings)
    else:
        # Prepare input
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_kwargs['huggingface_model_name'], trust_remote_code=True)
        tokenized_premises = tokenizer(premises, padding="longest", return_tensors="pt")
        tokenized_hypotheses = tokenizer(hypotheses, padding="longest", return_tensors="pt")
        tokenized_premises = {k: v.to(device) for k, v in tokenized_premises.items()}
        tokenized_hypotheses = {k: v.to(device) for k, v in tokenized_hypotheses.items()}
        
        # Run model in inference mode
        print_bold('Run model in inference mode')
        with torch.set_grad_enabled(False):
            model.train(False)
            with autocast(enabled=use_amp): # automatic mixed precision
                logits = model(tokenized_premises, tokenized_hypotheses)
    
    # Convert logits to labels
    print_bold('Convert logits to labels')
    softmax = torch.softmax(logits, dim=1)
    # labels = logits.argmax(dim=1).detach().cpu().numpy()
    labels = softmax.argmax(dim=1).detach().cpu().numpy()
    assert labels.shape == (len(premises),)
    labels = [_INDEX_TO_LABEL[x] for x in labels]
    
    # Print results
    print_bold('Results')
    for i in range(len(premises)):
        print(f'Premise: {premises[i]}')
        print(f'Hypothesis: {hypotheses[i]}')
        print(f'Label: {labels[i]}')
        for j in range(3):
            print(f'\t{softmax[i, j]:.4f} ({_INDEX_TO_LABEL[j]})')
        if use_precomputed_embeddings:
            print(f'\tp_embeddings[{i}][0] = {p_embeddings[i][0]}')
            print(f'\th_embeddings[{i}][0] = {h_embeddings[i][0]}')
        print('----------------')

def run_sentence_autoencoder(sentences, vocab_filepath, model_folder_path=None, model_checkpoint_path=None, use_amp=False, device='GPU'):
    
    from medvqa.models.checkpoint import load_metadata, get_checkpoint_filepath
    from medvqa.models.nlp.fact_encoder import FactEncoder
    from torch.cuda.amp.autocast_mode import autocast
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
    tokenized_sentences = tokenizer(sentences, padding="longest", return_tensors="pt")    
    input_ids = tokenized_sentences.input_ids.to(device)
    attention_mask = tokenized_sentences.attention_mask.to(device)
    
    # Run model in inference mode
    print_bold('Run model in inference mode')
    with torch.set_grad_enabled(False):
        model.train(False)
        with autocast(enabled=use_amp): # automatic mixed precision
            decoded_ids = model.fact_decoder_forward_greedy_decoding(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=input_ids.shape[1] * 2,
            )
    
    # Convert ids to string
    from medvqa.datasets.tokenizer import BasicTokenizer
    decoder_tokenizer = BasicTokenizer(vocab_filepath=vocab_filepath)
    decoded_ids = decoded_ids.detach().cpu().numpy()
    generated_sentences = [decoder_tokenizer.ids2string(x) for x in decoded_ids]
    
    # Print results
    print_bold('Results')
    for i in range(len(sentences)):
        print(f'Input: {sentences[i]}')
        print(f'Generated: {generated_sentences[i]}')
        print('----------------')

    # Release GPU memory
    del model
    del input_ids
    del attention_mask
    del decoded_ids
    import gc
    gc.collect()
    torch.cuda.empty_cache()

class PhraseGroundingVisualizer:
    def __init__(self,
                 text_embedding_model_name,
                 text_embedding_model_checkpoint_folder_path,
                 phrase_grounder_checkpoint_folder_path,
                 device='GPU'):
        
        import torch

        # device
        print_bold('device = ', device)
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')

        # Load phrase grounder
        print_bold('Load phrase grounder')
        self.metadata = load_metadata(phrase_grounder_checkpoint_folder_path)
        model_kwargs = self.metadata['model_kwargs']
        from medvqa.models.phrase_grounding.phrase_grounder import PhraseGrounder
        self.phrase_grounder = PhraseGrounder(**model_kwargs)
        self.phrase_grounder = self.phrase_grounder.to(self.device)

        # Load model weights
        print_bold('Load model weights')
        model_checkpoint_path = get_checkpoint_filepath(phrase_grounder_checkpoint_folder_path)
        print('model_checkpoint_path = ', model_checkpoint_path)
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        self.phrase_grounder.load_state_dict(checkpoint['model'])

        # Load text embedding model
        print_bold('Load text embedding model')
        from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
        self.cached_text_embedding_extractor = CachedTextEmbeddingExtractor(
            model_name=text_embedding_model_name,
            model_checkpoint_folder_path=text_embedding_model_checkpoint_folder_path,
        )

        # Load image transform
        print_bold('Load image transform')
        from medvqa.datasets.image_transforms_factory import create_image_transforms
        # try:
        self.train_image_transform_kwargs = self.metadata['train_image_transform_kwargs']
        self.val_image_transform_kwargs = self.metadata['val_image_transform_kwargs']
        if DATASET_NAMES.MIMICCXR in self.train_image_transform_kwargs:
            self.mimiccxr_train_image_transform = create_image_transforms(**self.train_image_transform_kwargs[DATASET_NAMES.MIMICCXR])
            self.mimiccxr_val_image_transform = create_image_transforms(**self.val_image_transform_kwargs[DATASET_NAMES.MIMICCXR])
        if DATASET_NAMES.VINBIG in self.train_image_transform_kwargs:
            self.vinbig_train_image_transform = create_image_transforms(**self.train_image_transform_kwargs[DATASET_NAMES.VINBIG])
            self.vinbig_val_image_transform = create_image_transforms(**self.val_image_transform_kwargs[DATASET_NAMES.VINBIG])
        self.image_mean = next(iter(self.train_image_transform_kwargs.values()))['image_mean']
        self.image_std = next(iter(self.train_image_transform_kwargs.values()))['image_std']

    def visualize_phrase_grounding(self, phrases, image_path, bbox_figsize=(10, 10), attention_figsize=(3, 3), attention_factor=1.0,
                                   mimiccxr_forward=False, vinbig_forward=False, yolov8_detection_layer_index=None,
                                   run_also_in_training_mode=False):
        
        assert sum([mimiccxr_forward, vinbig_forward]) == 1

        import torch

        # Load image
        print(f'image_path = {image_path}')
        if mimiccxr_forward:
            image_transform = self.mimiccxr_val_image_transform
        elif vinbig_forward:
            image_transform = self.vinbig_val_image_transform
        else: assert False
        image, image_size_before, image_size_after = image_transform(image_path, return_image_size=True)
        print(f'image.shape = {image.shape}')
        print(f'image_size_before = {image_size_before}')
        print(f'image_size_after = {image_size_after}')

        # Obtain text embeddings
        print_bold('Obtain text embeddings')
        text_embeddings = self.cached_text_embedding_extractor.compute_text_embeddings(phrases, update_cache_on_disk=False)
        print(f'text_embeddings.shape = {text_embeddings.shape}')
        
        # Run phrase grounder in inference mode
        print_bold('Run phrase grounder in inference mode')
        with torch.set_grad_enabled(False):
            self.phrase_grounder.eval()
            print('self.phrase_grounder.training = ', self.phrase_grounder.training)
            image = image.to(self.device)
            print(f'image.shape = {image.shape}')
            text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32).to(self.device)
            output = self.phrase_grounder(
                raw_images=image.unsqueeze(0),
                phrase_embeddings=text_embeddings.unsqueeze(0),
                mimiccxr_forward=mimiccxr_forward,
                vinbig_forward=vinbig_forward,
                yolov8_detection_layer_index=yolov8_detection_layer_index,
            )
            print(f'output.keys() = {output.keys()}')
            print('local_feat:')
            print(output['local_feat'])
            yolov8_predictions = output['yolov8_predictions'][0].detach().cpu()
            yolov8_predictions[:, :4] /= torch.tensor([image_size_after[1], image_size_after[0], image_size_after[1], image_size_after[0]], dtype=torch.float32)
            pred_coords = yolov8_predictions[:, :4].numpy()
            pred_classes = yolov8_predictions[:, 5].numpy().astype(int)
            print(f'pred_coords.shape = {pred_coords.shape}')
            print(f'pred_classes.shape = {pred_classes.shape}')
            sigmoid_attention = output['sigmoid_attention'][0].detach().cpu().numpy()
            sigmoid_attention = sigmoid_attention.reshape(-1, self.phrase_grounder.regions_height, self.phrase_grounder.regions_width)
            print(f'sigmoid_attention.shape = {sigmoid_attention.shape}')
        
        
        # Visualize bbox predictions
        if mimiccxr_forward:
            from medvqa.datasets.chest_imagenome import CHEST_IMAGENOME_BBOX_NAMES
            bbox_class_names = CHEST_IMAGENOME_BBOX_NAMES
        elif vinbig_forward:
            from medvqa.datasets.vinbig import VINBIG_BBOX_NAMES
            bbox_class_names = VINBIG_BBOX_NAMES
        else: assert False
        from medvqa.evaluation.plots import visualize_predicted_bounding_boxes__yolo
        print_bold('Visualize bbox predictions')
        visualize_predicted_bounding_boxes__yolo(
            image_path=image_path,
            pred_coords=pred_coords,
            pred_classes=pred_classes,
            class_names=bbox_class_names,
            figsize=bbox_figsize,
            format='xyxy',
        )

        # Visualize attention maps
        print_bold('Visualize attention maps')
        from medvqa.evaluation.plots import visualize_attention_maps
        n_rows = int(np.ceil(len(phrases) / 3))
        n_cols = min(len(phrases), 3)
        figsize = (attention_figsize[0] * n_cols, attention_figsize[1] * n_rows)
        visualize_attention_maps(
            image_path=image_path,
            attention_maps=sigmoid_attention,
            figsize=figsize,
            titles=phrases,
            attention_factor=attention_factor,
            max_cols=3,
        )

        # Run phrase grounder in training mode
        if run_also_in_training_mode:
            print_bold('Run phrase grounder in training mode')
            with torch.set_grad_enabled(False):
                self.phrase_grounder.train(True)
                print('self.phrase_grounder.training = ', self.phrase_grounder.training)
                print(f'image.shape = {image.shape}')
                output = self.phrase_grounder(
                    raw_images=image.unsqueeze(0),
                    phrase_embeddings=text_embeddings.unsqueeze(0),
                    mimiccxr_forward=mimiccxr_forward,
                    vinbig_forward=vinbig_forward,
                    yolov8_detection_layer_index=yolov8_detection_layer_index,
                )
                print(f'output.keys() = {output.keys()}')
                print('local_feat:')
                print(output['local_feat'])
                sigmoid_attention = output['sigmoid_attention'][0].detach().cpu().numpy()
                sigmoid_attention = sigmoid_attention.reshape(-1, self.phrase_grounder.regions_height, self.phrase_grounder.regions_width)
                print(f'sigmoid_attention.shape = {sigmoid_attention.shape}')

                # Visualize attention maps
                print_bold('Visualize attention maps')
                visualize_attention_maps(
                    image_path=image_path,
                    attention_maps=sigmoid_attention,
                    figsize=figsize,
                    titles=phrases,
                    attention_factor=attention_factor,
                    max_cols=3,
                )

    def visualize_phrase_grounding_bbox_mode(
            self, phrases, image_path, mimiccxr_forward=False, vinbig_forward=False, subfigsize=(3, 3),
            gt_phrases_to_highlight=None, phrases_and_embeddings_file_path=None, show_heatmaps=False,
            apply_data_augmentation=False, iou_threshold=0.5, conf_threshold=0.5, max_det_per_class=100):
        
        assert sum([mimiccxr_forward, vinbig_forward]) == 1

        from medvqa.utils.common import activate_determinism
        activate_determinism() # for reproducibility

        import torch

        # Load image
        print(f'image_path = {image_path}')
        if mimiccxr_forward:
            image_transform = (self.mimiccxr_train_image_transform
                               if apply_data_augmentation else self.mimiccxr_val_image_transform)
        elif vinbig_forward:
            image_transform = (self.vinbig_train_image_transform
                               if apply_data_augmentation else self.vinbig_val_image_transform)
        else: assert False
        if apply_data_augmentation:
            # Break determinism by using OS-level entropy or system time
            import random
            seed = int.from_bytes(os.urandom(4), "big")  # Generate a random seed using OS-level entropy
            np.random.seed(seed)
            random.seed(seed)
        if apply_data_augmentation:
            image = image_transform(image_path, bboxes=[], bbox_labels=[])['pixel_values']
        else:
            image = image_transform(image_path)['pixel_values']
        print(f'image.shape = {image.shape}')
            
        from PIL import Image
        denorm_image = image.clone()
        denorm_image.mul_(torch.tensor(self.image_std).view(3, 1, 1).to(image.device))
        denorm_image.add_(torch.tensor(self.image_mean).view(3, 1, 1).to(image.device))
        image_from_tensor = Image.fromarray((denorm_image.permute(1,2,0) * 255).numpy().astype(np.uint8))

        # Obtain text embeddings
        if phrases_and_embeddings_file_path is not None:
            print_bold('Load precomputed text embeddings')
            phrases_and_embeddings = load_pickle(phrases_and_embeddings_file_path)
            print(f'loaded phrases = {phrases_and_embeddings["phrases"]}')
            print(f'loaded embeddings\'s shape = {phrases_and_embeddings["phrase_embeddings"].shape}')
            p2e = {p:e for p, e in zip(phrases_and_embeddings['phrases'], phrases_and_embeddings['phrase_embeddings'])}
            text_embeddings = [p2e[p] for p in phrases]
            text_embeddings = np.array(text_embeddings)
        else:
            print_bold('Computing text embeddings')
            text_embeddings = self.cached_text_embedding_extractor.compute_text_embeddings(phrases, update_cache_on_disk=False)
        print(f'text_embeddings.shape = {text_embeddings.shape}')
        
        # Run phrase grounder in inference mode
        print_bold('Run phrase grounder in inference mode')
        with torch.set_grad_enabled(False):
            self.phrase_grounder.eval()
            print('self.phrase_grounder.training = ', self.phrase_grounder.training)
            image = image.to(self.device)
            print(f'image.shape = {image.shape}')
            text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32).to(self.device)
            output = self.phrase_grounder(
                raw_images=image.unsqueeze(0),
                phrase_embeddings=text_embeddings.unsqueeze(0),
                only_compute_features=True,
                predict_bboxes=True,
                apply_nms=True,
                iou_threshold=iou_threshold,
                conf_threshold=conf_threshold,
                max_det_per_class=max_det_per_class,
                return_sigmoid_attention=show_heatmaps,
            )
            print(f'output.keys() = {output.keys()}')
            predicted_bboxes = output['predicted_bboxes']
            phrase_classifier_logits = output['phrase_classifier_logits']
            phrase_classifier_probs = torch.sigmoid(phrase_classifier_logits)
            if show_heatmaps:
                sigmoid_attention = output['sigmoid_attention']

        # Visualize bounding boxes
        print_bold('Visualize bounding boxes')
        from medvqa.evaluation.plots import visualize_visual_grounding_as_bboxes
        num_subplots = len(phrases)+1 # +1 for the image
        n_rows = int(np.ceil(num_subplots / 3))
        n_cols = min(num_subplots, 3)
        figsize = (subfigsize[0] * n_cols, subfigsize[1] * n_rows)
        print(f'figsize = {figsize}')
        visualize_visual_grounding_as_bboxes(
            image=image_from_tensor,
            phrases=phrases,
            gt_phrases_to_highlight=gt_phrases_to_highlight,
            phrase_classifier_probs=phrase_classifier_probs[0].cpu().numpy(),
            bbox_coords=predicted_bboxes[0][0].cpu().numpy(),
            bbox_probs=predicted_bboxes[0][1].cpu().numpy(),
            phrase_ids=predicted_bboxes[0][2].cpu().numpy(),
            show_heatmaps=show_heatmaps,
            heatmaps=sigmoid_attention[0].cpu().numpy() if show_heatmaps else None,
            figsize=figsize,
            max_cols=3,
            bbox_format=self.phrase_grounder.visual_grounding_bbox_regressor.bbox_format,
            display_raw_image=True,
        )

class YOLOv11Visualizer:
    def __init__(self, model_checkpoint_folder_path: str, device: str = 'cuda', fact_conditioned: bool = False,
                 vinbig_phrase_embeddings_filepath=None):

        import torch

        self.fact_conditioned = fact_conditioned

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        print_bold('self.device = ', self.device)

        # Load model
        print_bold('Load model')
        self.metadata = load_metadata(model_checkpoint_folder_path)
        model_kwargs = self.metadata['model_kwargs']
        if fact_conditioned:
            from medvqa.models.phrase_grounding.phrase_grounder import PhraseGrounder
            self.model = PhraseGrounder(**model_kwargs, device=self.device)
        else:
            from medvqa.models.vision.visual_modules import MultiPurposeVisualModule
            self.model = MultiPurposeVisualModule(**model_kwargs, device=self.device)
        self.model = self.model.to(self.device)

        # Load model weights
        print_bold('Load model weights')
        model_checkpoint_path = get_checkpoint_filepath(model_checkpoint_folder_path)
        print('model_checkpoint_path = ', model_checkpoint_path)
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

        # Load image transform
        print_bold('Load image transform')
        from medvqa.datasets.image_processing import get_image_transform
        self.image_transform_kwargs = self.metadata['val_image_transform_kwargs']
        if DATASET_NAMES.MIMICCXR in self.image_transform_kwargs:
            self.mimiccxr_image_transform = get_image_transform(**self.image_transform_kwargs[DATASET_NAMES.MIMICCXR])
        if DATASET_NAMES.VINBIG in self.image_transform_kwargs:
            self.vinbig_image_transform = get_image_transform(**self.image_transform_kwargs[DATASET_NAMES.VINBIG])

        # Ground truth bounding boxes
        self._cig_image_id_2_gt_bboxes = None
        self._vinbig_image_id_2_gt_bboxes = None

        # Load VinBigData fact embeddings
        if vinbig_phrase_embeddings_filepath is not None:
            self.vinbig_phrase_embeddings = load_pickle(vinbig_phrase_embeddings_filepath)

    @property
    def cig_image_id_2_gt_bboxes(self):
        if self._cig_image_id_2_gt_bboxes is None:
            from medvqa.datasets.chest_imagenome.chest_imagenome_dataset_management import load_chest_imagenome_silver_bboxes
            self._cig_image_id_2_gt_bboxes = load_chest_imagenome_silver_bboxes()
        return self._cig_image_id_2_gt_bboxes
    
    @property
    def vinbig_image_id_2_gt_bboxes(self):
        if self._vinbig_image_id_2_gt_bboxes is None:
            from medvqa.datasets.vinbig.vinbig_dataset_management import load_train_image_id_2_bboxes, load_test_image_id_2_bboxes
            train_image_id_2_bboxes = load_train_image_id_2_bboxes(for_training=True, normalize=True)
            test_image_id_2_bboxes = load_test_image_id_2_bboxes(for_training=True, normalize=True)
            self._vinbig_image_id_2_gt_bboxes = {**train_image_id_2_bboxes, **test_image_id_2_bboxes}
        return self._vinbig_image_id_2_gt_bboxes

    def visualize_yolov11_predictions(self, image_path: str, dataset_name: str,
                 conf_thres: float = 0.5, iou_thres: float = 0.5, max_det: int = 100,
                 figsize: tuple = (10, 10), format: str = 'xyxy', show_gt: bool = True):
        
        if not self.fact_conditioned:
            assert dataset_name in self.model.raw_image_encoder.detect # make sure the model has a detection layer for the dataset
        
        if dataset_name == 'cig': # Chest ImaGenome
            class_names = CHEST_IMAGENOME_BBOX_NAMES
            image_transform = self.mimiccxr_image_transform
            if self.fact_conditioned:
                raise NotImplementedError
        elif dataset_name == 'vinbig': # VinBigData
            class_names = VINBIG_BBOX_NAMES
            image_transform = self.vinbig_image_transform
            if self.fact_conditioned:
                assert self.vinbig_phrase_embeddings is not None
                phrase_embeddings = self.vinbig_phrase_embeddings['phrase_embeddings']
                phrases = self.vinbig_phrase_embeddings['phrases']
                assert len(phrase_embeddings) == len(phrases)
                assert len(phrases) > len(class_names) # only 22 out of 28 phrases have bounding boxes
                phrases = phrases[:len(class_names)] # keep only phrases with bounding boxes
                phrase_embeddings = phrase_embeddings[:len(class_names)]
        else: assert False
        
        import torch

        # Load image
        image = image_transform(image_path)
        print(f'image_path = {image_path}')
        print(f'image.shape = {image.shape}')

        # Run model in inference mode
        print_bold('Run model in inference mode')
        with torch.set_grad_enabled(False):
            self.model.eval()
            print('self.model.training = ', self.model.training)
            image = image.to(self.device)
            print(f'image.shape = {image.shape}')
            if self.fact_conditioned:
                phrase_embeddings = torch.tensor(phrase_embeddings, dtype=torch.float32).to(self.device)
                phrase_embeddings = phrase_embeddings.unsqueeze(0) # add batch dimension (1, num_phrases, embedding_dim)
                output = self.model(
                    raw_images=image.unsqueeze(0),
                    phrase_embeddings=phrase_embeddings,
                    apply_nms=True,
                    conf_threshold=conf_thres,
                    iou_threshold=iou_thres,
                    max_det=max_det,
                )
                print(f'output.keys() = {output.keys()}')
                classification_logits = output['classification_logits']
                detection = output['detection']
                assert len(classification_logits) == 1
                assert len(detection) == 1
                classification_logits = classification_logits[0]
                classification_probs = classification_logits.sigmoid().cpu().numpy()
                sorted_indexes = classification_probs.argsort()[::-1]
                print_bold('Classification probabilities:')
                for i in sorted_indexes:
                    print(f'\t{phrases[i]}: {classification_probs[i]:.4f}')
                detection = detection[0]
                assert len(detection) == len(class_names)
                print_bold('Detection:')
                print(detection)
                pred_coords = []
                pred_confs = []
                pred_classes = []
                for i in range(len(class_names)):
                    if detection[i] is not None and len(detection[i]) > 0:
                        detection_i = detection[i].cpu().numpy()
                        pred_coords.append(detection_i[:, :4])
                        pred_confs.append(detection_i[:, 4])
                        assert np.all(detection_i[:, 5] == 0) # all detections should have class 0
                        pred_classes.append(np.full(len(detection_i), i))
                pred_coords = np.concatenate(pred_coords, axis=0) if len(pred_coords) > 0 else np.empty((0, 4))
                pred_confs = np.concatenate(pred_confs) if len(pred_confs) > 0 else np.empty(0)
                pred_classes = np.concatenate(pred_classes) if len(pred_classes) > 0 else np.empty(0)
            else:
                output = self.model(
                    raw_images=image.unsqueeze(0),
                    yolov11_detection_tasks=dataset_name,
                    apply_nms=True,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    max_det=max_det,
                )
                print(f'output.keys() = {output.keys()}')
                predictions = output['detection'][dataset_name]
                assert len(predictions) == 1
                predictions = predictions[0].cpu().numpy()
                print(f'predictions.shape = {predictions.shape}')
                pred_coords = predictions[:, :4]
                pred_confs = predictions[:, 4]
                pred_classes = predictions[:, 5].astype(int)

        if show_gt:
            # Obtain ground truth bounding boxes
            if dataset_name == 'cig':
                key = os.path.basename(image_path).split('.')[0] # e.g., 'b1b1aa2a-7e114d6d-0879953c-b69a25f6-e7d82b54'
                out = self.cig_image_id_2_gt_bboxes[key]
                coords, presence = out['coords'], out['presence']
                coords = coords.reshape(-1, 4)
                gt_bbox_coords = [[] for _ in class_names]
                for class_id in range(len(class_names)):
                    if presence[class_id] == 1:
                        gt_bbox_coords[class_id].append(coords[class_id])
            elif dataset_name == 'vinbig':
                key = os.path.basename(image_path).split('.')[0] # e.g., '4f737958fc5a7d9805f4a94f70cfc5a2'
                coords_list, class_list = self.vinbig_image_id_2_gt_bboxes[key]
                gt_bbox_coords = [[] for _ in class_names]
                for coords, class_id in zip(coords_list, class_list):
                    gt_bbox_coords[class_id].append(coords)
            else: assert False
        else:
            gt_bbox_coords = None

        # Visualize bbox predictions
        from medvqa.evaluation.plots import visualize_predicted_bounding_boxes__yolo
        print_bold('Visualize bbox predictions')
        visualize_predicted_bounding_boxes__yolo(
            image_path=image_path,
            pred_coords=pred_coords,
            pred_confs=pred_confs,
            pred_classes=pred_classes,
            gt_bbox_coords=gt_bbox_coords,
            class_names=class_names,
            figsize=figsize,
            format=format,
        )