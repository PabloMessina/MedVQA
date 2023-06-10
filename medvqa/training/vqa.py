import bs4
import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.models.vqa.open_ended_vqa import QuestionEncoding
from medvqa.models.common import AnswerDecoding
from medvqa.losses import get_binary_multilabel_loss
from medvqa.losses.optimizers import GradientAccumulator
from medvqa.utils.constants import (
    CXR14_DATASET_ID,
    CHEXPERT_TASKS,
    IUXRAY_DATASET_ID,
    IUXRAY_DATASET_ID__CHEXPERT_MODE,
    MIMICCXR_DATASET_ID,
    CHEXPERT_DATASET_ID,
    MIMICCXR_DATASET_ID__CHEST_IMAGENOME__DETECTRON2_MODE,
    MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE,
    MIMICCXR_DATASET_ID__CHEXPERT_MODE,
    VINBIG_DATASET_ID,
    PADCHEST_DATASET_ID,
    MetricNames,
)
from medvqa.utils.logging import print_magenta

def get_step_fn(model, optimizer, nlg_criterion, tokenizer, training, device,
        use_visual_module_only=False,
        question_encoding=QuestionEncoding.BILSTM,
        answer_decoding=AnswerDecoding.TEACHER_FORCING,
        shift_answer=False,
        beam_search_k=None,
        include_answer=True,
        include_image=True,
        include_visual_features=False,
        max_answer_length=None,
        use_merged_findings=False,
        iters_to_accumulate=1, # for gradient accumulation
        # automatic mixed precision
        use_amp=False,
        # tags aux task
        classify_tags=False,
        tags_criterion=None,
        # orientation aux task
        classify_orientation=False,
        iuxray_orientation_criterion=None,
        mimiccxr_orientation_criterion=None,
        # gender aux task
        classify_gender=False,
        chest_imagenome_gender_criterion=None,
        # chexpert aux task
        classify_chexpert=False,
        chexpert_criterion=None,
        # question auxiliary task
        classify_questions=False,
        question_criterion=None,
        # chexpert dataset
        chexpert_aux_criterion=None,
        chexpert_mode=None,
        # cxr14 dataset
        cxr14_criterion=None,
        # vinbig dataset
        vinbig_criterion=None,
        predict_bboxes_vinbig=False,
        # padchest dataset
        padchest_multilabel_criterion=None,
        padchest_singlelabel_criterion=None,
        # chest imagenome dataset
        classify_chest_imagenome=False,
        chest_imagenome_multilabel_criterion=None,
        predict_bboxes_chest_imagenome=False,
        chest_imagenome_bbox_coords_criterion=None,
        chest_imagenome_bbox_presence_criterion=None,
        chest_imagenome_bbox_loss_weight=1.0,
        pass_pred_bbox_coords_as_input=False,
        valid_chest_imagenome_label_indices=None,
        # detectron2
        detectron2_includes_rpn=False,
        # yolov8
        using_yolov8=False,
        yolov8_criterion=None,
        yolov8_use_multiple_detection_layers=False,
        # batchwise learning rate updates
        update_lr_batchwise=False,
        lr_scheduler=None,
    ):

    scaler = GradScaler(enabled=use_amp)

    if not use_visual_module_only:
        verbose_question = question_encoding != QuestionEncoding.ONE_HOT
        # teacher forcing only during training
        assert training == (answer_decoding == AnswerDecoding.TEACHER_FORCING)
        use_beam_search = answer_decoding == AnswerDecoding.BEAM_SEARCH    
        if use_beam_search:
            assert beam_search_k is not None
        if not include_answer:
            assert max_answer_length is not None
        use_chexpert_vqa = chexpert_mode == CHEXPERT_TASKS.VQA

    if update_lr_batchwise:
        assert lr_scheduler is not None

    if yolov8_use_multiple_detection_layers:
        print('Using multiple detection layers in yolov8')
        mimiccxr_yolov8_index = 0
        vinbig_yolov8_index = 1

    if using_yolov8 and training:
        assert yolov8_criterion is not None
        if yolov8_use_multiple_detection_layers:
            mimiccxr_yolov8_criterion = lambda *args: yolov8_criterion(*args, mimiccxr_yolov8_index)
            vinbig_yolov8_criterion = lambda *args: yolov8_criterion(*args, vinbig_yolov8_index)
        else:
            mimiccxr_yolov8_criterion = vinbig_yolov8_criterion = yolov8_criterion

    if predict_bboxes_vinbig:
        assert using_yolov8

    if training:
        gradient_accumulator = GradientAccumulator(optimizer, scaler, iters_to_accumulate)
    
    _mimiccxr_dataset_ids = [MIMICCXR_DATASET_ID, MIMICCXR_DATASET_ID__CHEXPERT_MODE,
                             MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE]

    if predict_bboxes_chest_imagenome:
        print('chest_imagenome_bbox_loss_weight: ', chest_imagenome_bbox_loss_weight)

    # DEBUG = True

    def step_fn__mimiccxr_chest_imagenome_detectron2(batch):

        from detectron2.utils.events import EventStorage

        # Extract elements from batch
        dataset_id = batch['dataset_id']
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'detectron2_forward': True,
                'detectron2_input': batch['batch'],
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                with EventStorage() as storage:
                    model_output = model(**model_kwargs)

                if training: 
                    # Compute losses
                    loss_cls = model_output['loss_cls']
                    loss_box_reg = model_output['loss_box_reg']
                    batch_loss = loss_cls + loss_box_reg
                    if detectron2_includes_rpn:
                        loss_rpn_cls = model_output['loss_rpn_cls']
                        loss_rpn_loc = model_output['loss_rpn_loc']
                        batch_loss += loss_rpn_cls + loss_rpn_loc

                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss)

        # Prepare output
        output = {
            'dataset_id': dataset_id,
        }
        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
            output[MetricNames.DETECTRON2_CLS_LOSS] = loss_cls.detach()
            output[MetricNames.DETECTRON2_BOX_REG_LOSS] = loss_box_reg.detach()
            if detectron2_includes_rpn:
                output[MetricNames.DETECTRON2_RPN_CLS_LOSS] = loss_rpn_cls.detach()
                output[MetricNames.DETECTRON2_RPN_LOC_LOSS] = loss_rpn_loc.detach()
        else:
            # print('model_output: ', model_output)
            output['pred_boxes'] = [x['instances'].pred_boxes.tensor.detach().cpu() for x in model_output]
            output['pred_classes'] = [x['instances'].pred_classes.detach().cpu() for x in model_output]
            output['scores'] = [x['instances'].scores.detach().cpu() for x in model_output]
            output['bbox_coords'] = [x['bbox_coords'] for x in batch['batch']]
            output['bbox_presence'] = [x['bbox_presence'] for x in batch['batch']]

            # nonlocal DEBUG
            # if DEBUG:
            #     print('output[bbox_coords]: ', output['bbox_coords'])
            #     print('output[bbox_presence]: ', output['bbox_presence'])
            #     print('output[pred_boxes]: ', output['pred_boxes'])
            #     print('output[pred_classes]: ', output['pred_classes'])
            #     print('output[scores]: ', output['scores'])
            #     DEBUG = False

        return output
    
    def step_fn__mimiccxr_iuxray(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']        
        if include_image:
            if using_yolov8:
                images = batch['img'].to(device)
            else:
                images = batch['i'].to(device)
        if include_visual_features:
            visual_features = batch['vf'].to(device)
        if not use_visual_module_only:
            questions = batch['q'].to(device)
            if verbose_question:
                question_lengths = batch['ql']
            if include_answer:
                answers = batch['a'].to(device)
                if shift_answer:
                    answers_start = answers[:, :-1]
                    answers_end = answers[:, 1:]

        is_mimiccxr = dataset_id in _mimiccxr_dataset_ids

        findings_name = 'findings' if use_merged_findings else 'chexpert'
        
        if classify_tags:
            tags = batch['tags'].to(device)
        if classify_orientation:
            orientation = batch['orientation'].to(device)
        if classify_chexpert:
            chexpert = batch['chexpert'].to(device)
        if classify_questions:
            question_labels = batch['qlabels'].to(device)
        if classify_gender:
            genders = batch['gender'].to(device)
        if classify_chest_imagenome:
            chest_imagenome = batch['chest_imagenome'].to(device)
        if predict_bboxes_chest_imagenome:
            if (using_yolov8 and not training) or not using_yolov8:
                chest_imagenome_bbox_coords = batch['chest_imagenome_bbox_coords'].to(device)
                chest_imagenome_bbox_presence = batch['chest_imagenome_bbox_presence'].to(device)
        if pass_pred_bbox_coords_as_input:
            predicted_bbox_coords = batch['pred_bbox_coords'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {                
                'device': device,
                'mode': 'train' if training else 'eval',
            }            
            if include_image:
                model_kwargs['raw_images'] = images
                # print(f'step_fn__mimiccxr_iuxray: images.shape: {images.shape}')
                # assert len(images.shape) == 4
            if include_visual_features:
                model_kwargs['visual_features'] = visual_features
            if is_mimiccxr:
                model_kwargs['mimiccxr_forward'] = True
            else:
                model_kwargs['iuxray_forward'] = True
            if pass_pred_bbox_coords_as_input:
                model_kwargs['pred_bbox_coords'] = predicted_bbox_coords
                model_kwargs['refine_bbox_coords'] = predict_bboxes_chest_imagenome
            if is_mimiccxr and using_yolov8 and yolov8_use_multiple_detection_layers:
                model_kwargs['yolov8_detection_layer_index'] = mimiccxr_yolov8_index

            if not use_visual_module_only:
                model_kwargs['questions'] = questions
                if verbose_question:
                    model_kwargs['question_lengths'] = question_lengths
                if training:
                    if shift_answer:
                        model_kwargs['answers'] = answers_start
                    else:
                        model_kwargs['answers'] = answers
                else:
                    if use_beam_search:
                        model_kwargs['beam_search_k'] = beam_search_k
                    if include_answer:
                        model_kwargs['max_answer_length'] = answers.size(1)
                    else:                    
                        model_kwargs['max_answer_length'] = max_answer_length

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                
                if classify_tags:
                    pred_tags_logits = model_output['pred_tags']            
                if classify_orientation:
                    if is_mimiccxr:
                        pred_orientation_logits = model_output['mimiccxr_pred_orientation']
                    else:
                        pred_orientation_logits = model_output['iuxray_pred_orientation']
                if classify_chexpert:
                    pred_chexpert_logits = model_output[f'pred_{findings_name}']
                    pred_chexpert_probs = model_output[f'pred_{findings_name}_probs']
                if classify_questions:
                    pred_qlabels_logits = model_output['pred_qlabels']
                if classify_gender:
                    pred_gender_logits = model_output['pred_gender']
                if classify_chest_imagenome:
                    pred_chest_imagenome_logits = model_output['pred_chest_imagenome']
                    pred_chest_imagenome_probs = model_output['pred_chest_imagenome_probs']
                    if valid_chest_imagenome_label_indices is not None:
                        pred_chest_imagenome_logits = pred_chest_imagenome_logits[:, valid_chest_imagenome_label_indices]
                        pred_chest_imagenome_probs = pred_chest_imagenome_probs[:, valid_chest_imagenome_label_indices]
                if predict_bboxes_chest_imagenome:
                    if using_yolov8:
                        yolov8_features = model_output['yolov8_features']
                        if not training:
                            yolov8_predictions = model_output['yolov8_predictions']
                    else:
                        pred_chest_imagenome_bbox_coords = model_output['pred_chest_imagenome_bbox_coords']
                        bs = pred_chest_imagenome_bbox_coords.size(0)
                        pred_chest_imagenome_bbox_coords = pred_chest_imagenome_bbox_coords.view(bs, -1, 4)
                        pred_chest_imagenome_bbox_presence = model_output['pred_chest_imagenome_bbox_presence']

                if not use_visual_module_only:
                    if training:
                        pred_answer_logits = model_output['pred_answers']
                    else:
                        pred_answers = model_output['pred_answers']
                    
                    if verbose_question:
                        pred_question_logits = model_output['pred_questions']

                if training:
                    # Compute losses
                    losses = []
                    if classify_tags:
                        tags_loss = tags_criterion(pred_tags_logits, tags.float())
                        losses.append(tags_loss)
                    if classify_orientation:
                        if is_mimiccxr:
                            orientation_loss = mimiccxr_orientation_criterion(pred_orientation_logits, orientation)
                        else:
                            orientation_loss = iuxray_orientation_criterion(pred_orientation_logits, orientation)
                        losses.append(orientation_loss)
                    if classify_chexpert:
                        chexpert_loss = chexpert_criterion(pred_chexpert_logits, chexpert.float())
                        losses.append(chexpert_loss)
                    if classify_questions:
                        qlabels_loss = question_criterion(pred_qlabels_logits, question_labels.float())
                        losses.append(qlabels_loss)
                    if classify_gender:
                        gender_loss = chest_imagenome_gender_criterion(pred_gender_logits, genders)
                        losses.append(gender_loss)
                    if classify_chest_imagenome:
                        chest_imagenome_loss = chest_imagenome_multilabel_criterion(pred_chest_imagenome_logits, chest_imagenome.float())
                        losses.append(chest_imagenome_loss)
                    if predict_bboxes_chest_imagenome:
                        if using_yolov8:
                            batch_size = images.shape[0]
                            assert batch_size == yolov8_features[0].shape[0]
                            chest_imagenome_yolov8_loss, yolov8_loss_items = mimiccxr_yolov8_criterion(yolov8_features, batch)
                            chest_imagenome_yolov8_loss /= batch_size
                            losses.append(chest_imagenome_yolov8_loss)
                        else:
                            chest_imagenome_bbox_coords_loss = chest_imagenome_bbox_coords_criterion(
                                pred_chest_imagenome_bbox_coords, chest_imagenome_bbox_coords.float())
                            chest_imagenome_bbox_presence_loss = chest_imagenome_bbox_presence_criterion(
                                pred_chest_imagenome_bbox_presence, chest_imagenome_bbox_presence.float())
                            chest_imagenome_bbox_loss = chest_imagenome_bbox_coords_loss + chest_imagenome_bbox_presence_loss
                            chest_imagenome_bbox_loss = chest_imagenome_bbox_loss * chest_imagenome_bbox_loss_weight # weight the loss
                            losses.append(chest_imagenome_bbox_loss)
                    
                    if not use_visual_module_only:
                        if verbose_question:
                            question_loss = nlg_criterion(pred_question_logits.view(-1, pred_question_logits.shape[-1]), questions.view(-1))            
                            losses.append(question_loss)
                        if include_answer:                        
                            if shift_answer:
                                answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers_end.reshape(-1))
                            else:
                                answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers.view(-1))
                            losses.append(answer_loss)

                    if len(losses) > 0:
                        batch_loss = sum(losses)
                    else:
                        batch_loss = None

                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss)

        # Prepare output
        output = {
            'idxs': idxs,            
            'dataset_id': dataset_id,
        }

        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        if classify_tags:
            output['tags'] = tags.detach().cpu()
            output['pred_tags'] = (pred_tags_logits.detach() > 0).cpu()
            if training:
                output['tags_loss'] = tags_loss.detach()
        if classify_orientation:
            output['orientation'] = orientation.detach()
            output['pred_orientation'] = pred_orientation_logits.argmax(-1).detach()            
            if training:
                output['orientation_loss'] = orientation_loss.detach()
        if classify_chexpert:
            output['chexpert'] = chexpert.detach().cpu()
            output[f'pred_chexpert'] = (pred_chexpert_logits.detach() > 0).cpu()
            output[f'pred_chexpert_probs'] = pred_chexpert_probs.detach().cpu()
            if training:
                output[f'chexpert_loss'] = chexpert_loss.detach()
        if classify_questions:
            output['qlabels'] = question_labels.detach().cpu()
            output['pred_qlabels'] = (pred_qlabels_logits.detach() > 0).cpu()
            if training:
                output['qlabels_loss'] = qlabels_loss.detach()
        if classify_gender:
            output['gender'] = genders.detach()
            output['pred_gender'] = pred_gender_logits.argmax(-1).detach()
            if training:
                output['gender_loss'] = gender_loss.detach()
        if classify_chest_imagenome:
            output['chest_imagenome'] = chest_imagenome.detach().cpu()
            output[f'pred_chest_imagenome'] = (pred_chest_imagenome_logits.detach() > 0).cpu()
            output[f'pred_chest_imagenome_probs'] = pred_chest_imagenome_probs.detach().cpu()
            if training:
                output[MetricNames.CHEST_IMAGENOME_LABEL_LOSS] = chest_imagenome_loss.detach()
        if predict_bboxes_chest_imagenome:
            if using_yolov8:
                if training:
                    output[MetricNames.YOLOV8_LOSS] = chest_imagenome_yolov8_loss.detach()
                    output[MetricNames.YOLOV8_BOX_LOSS] = yolov8_loss_items[0]
                    output[MetricNames.YOLOV8_CLS_LOSS] = yolov8_loss_items[1]
                    output[MetricNames.YOLOV8_DFL_LOSS] = yolov8_loss_items[2]
                else:
                    # normalize yolov8 predictions
                    resized_shapes = batch['resized_shape']
                    assert len(resized_shapes) == len(yolov8_predictions)
                    for i in range(len(resized_shapes)):
                        resized_shape = resized_shapes[i]
                        pred = yolov8_predictions[i].detach().cpu()
                        pred[:, :4] /= torch.tensor([resized_shape[1], resized_shape[0], resized_shape[1], resized_shape[0]], dtype=torch.float32)
                        yolov8_predictions[i] = pred
                    output['yolov8_predictions'] = yolov8_predictions
                    output['chest_imagenome_bbox_coords'] = chest_imagenome_bbox_coords.detach().cpu()
                    output['chest_imagenome_bbox_presence'] = chest_imagenome_bbox_presence.detach().cpu()
            else:
                output['chest_imagenome_bbox_coords'] = chest_imagenome_bbox_coords.detach().cpu()
                output['chest_imagenome_bbox_presence'] = chest_imagenome_bbox_presence.detach().cpu()
                output['pred_chest_imagenome_bbox_coords'] = pred_chest_imagenome_bbox_coords.detach().cpu()
                output['pred_chest_imagenome_bbox_presence'] = pred_chest_imagenome_bbox_presence.detach().cpu()
                if training:
                    output[MetricNames.CHEST_IMAGENOME_BBOX_LOSS] = chest_imagenome_bbox_loss.detach()

        if not use_visual_module_only:
            # Compute predicted Q & A
            if training:
                pred_answers = pred_answer_logits.argmax(-1)
            if verbose_question:
                pred_questions = pred_question_logits.argmax(-1)            
            output['pred_answers'] = tokenizer.clean_batch(pred_answers.detach())
            if verbose_question:
                output['questions'] = tokenizer.clean_batch(questions.detach())
                output['pred_questions'] = tokenizer.clean_batch(pred_questions.detach())
                if training:
                    output['question_loss'] = question_loss.detach()
            else:
                output['questions'] = questions.detach()
            if include_answer:
                output['answers'] = tokenizer.clean_batch(answers.detach())
                if training:
                    output['answer_loss'] = answer_loss.detach()

        return output
    
    def step_fn__chexpert_cxr14(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        use_labels = (dataset_id == CHEXPERT_DATASET_ID and classify_chexpert) or (dataset_id == CXR14_DATASET_ID)
        use_vqa_mode = not use_visual_module_only and (use_chexpert_vqa or dataset_id == CXR14_DATASET_ID)
        dataset_name = 'chexpert' if dataset_id == CHEXPERT_DATASET_ID else 'cxr14'
        findings_name = 'findings' if use_merged_findings else dataset_name
        findings_criterion = chexpert_criterion if dataset_id == CHEXPERT_DATASET_ID else cxr14_criterion

        if classify_orientation:
            orientations = batch['o'].to(device)
        if classify_gender:
            genders = batch['g'].to(device)
        if use_labels:
            labels = batch['l'].to(device)
        if include_image:
            images = batch['i'].to(device)
        if include_visual_features:
            visual_features = batch['vf'].to(device)
        if use_vqa_mode:
            questions = batch['q'].to(device)
            answers = batch['a'].to(device)
            if shift_answer:
                answers_start = answers[:, :-1]
                answers_end = answers[:, 1:]
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                f'{dataset_name}_forward': True,
                'device': device,
            }

            if include_image:
                model_kwargs['raw_images'] = images
            if include_visual_features:
                model_kwargs['visual_features'] = visual_features

            if use_vqa_mode:
                model_kwargs['questions'] = questions
                if training:
                    model_kwargs['mode'] = 'train'
                    if shift_answer:
                        model_kwargs['answers'] = answers_start
                    else:
                        model_kwargs['answers'] = answers
                else:
                    model_kwargs['mode'] = 'eval'
                    if use_beam_search:
                        model_kwargs['beam_search_k'] = beam_search_k
                    if include_answer:
                        model_kwargs['max_answer_length'] = answers.size(1)
                    else:                    
                        model_kwargs['max_answer_length'] = max_answer_length

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision

                model_output = model(**model_kwargs)

                if use_labels:                
                    pred_labels_logits = model_output[f'pred_{findings_name}']
                    pred_labels_probs = model_output[f'pred_{findings_name}_probs']
                if classify_orientation:
                    pred_orientation_logits = model_output['pred_orientation']
                if classify_gender:
                    pred_gender_logits = model_output['pred_gender']
                if use_vqa_mode:
                    if training:
                        pred_answer_logits = model_output['pred_answers']
                    else:
                        pred_answers = model_output['pred_answers']

                if training:                    
                    # Compute losses
                    losses = []
                    if use_labels:
                        labels_loss = findings_criterion(pred_labels_logits, labels.float())
                        losses.append(labels_loss)
                    if classify_orientation:
                        orientation_loss = chexpert_aux_criterion(pred_orientation_logits, orientations)
                        losses.append(orientation_loss)
                    if classify_gender:
                        gender_loss = chexpert_aux_criterion(pred_gender_logits, genders)
                        losses.append(gender_loss)                    
                    if use_vqa_mode:
                        # answer_loss = nlg_criterion(pred_answer_logits.view(-1, pred_answer_logits.shape[-1]), answers.view(-1))
                        if shift_answer:
                            answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers_end.reshape(-1))
                        else:
                            answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers.view(-1))
                        losses.append(answer_loss)                    
                    # Backward pass + optimizer step
                    assert len(losses) > 0
                    batch_loss = sum(losses)
                    gradient_accumulator.step(batch_loss)
        
        output = {
            'idxs': idxs,
            'dataset_id': dataset_id,
        }            
        if training:
            output['loss'] = batch_loss.detach()
            
        # dataset-specific labels
        if use_labels:
            output[dataset_name] = labels.detach().cpu()
            output[f'pred_{dataset_name}'] = (pred_labels_logits.detach() > 0).cpu()
            output[f'pred_{dataset_name}_probs'] = pred_labels_probs.detach().cpu()
            if training:
                output[f'{dataset_name}_loss'] = labels_loss.detach()

        # orientation
        if classify_orientation:
            output['orientation'] = orientations.detach()
            output['pred_orientation'] = pred_orientation_logits.argmax(-1).detach()        
            if training:
                output['orientation_loss'] = orientation_loss.detach()

        # gender
        if classify_gender:
            output['gender'] = genders.detach()
            output['pred_gender'] = pred_gender_logits.argmax(-1).detach()
            if training:
                output['gender_loss'] = gender_loss.detach()

        # answers (vqa)
        if use_vqa_mode:
            if training:
                pred_answers = pred_answer_logits.argmax(-1)        
                output['answer_loss'] = answer_loss.detach()            
            output['pred_answers'] = tokenizer.clean_batch(pred_answers.detach())
            output['answers'] = tokenizer.clean_batch(answers.detach())                

        return output

    def step_fn__vinbig(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        vinbig_labels = batch['l'].to(device)
        if include_image:
            if using_yolov8:
                images = batch['img'].to(device)
            else:
                images = batch['i'].to(device)
        if include_visual_features:
            visual_features = batch['vf'].to(device)
        if not use_visual_module_only:
            questions = batch['q'].to(device)
            answers = batch['a'].to(device)
            if shift_answer:
                answers_start = answers[:, :-1]
                answers_end = answers[:, 1:]

        findings_name = 'findings' if use_merged_findings else 'vinbig'

        if predict_bboxes_vinbig:
            assert using_yolov8
            if not training:
                vinbig_bbox_coords = batch['bboxes']
                vinbig_bbox_classes = batch['classes']
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'vinbig_forward': True,
                'device': device,
                'mode': 'train' if training else 'eval',
            }

            if include_image:
                model_kwargs['raw_images'] = images
            if include_visual_features:
                model_kwargs['visual_features'] = visual_features
            if using_yolov8 and yolov8_use_multiple_detection_layers:
                model_kwargs['yolov8_detection_layer_index'] = vinbig_yolov8_index

            if not use_visual_module_only:
                model_kwargs['questions'] = questions
                if training:
                    if shift_answer:
                        model_kwargs['answers'] = answers_start
                    else:
                        model_kwargs['answers'] = answers
                else:
                    if use_beam_search:
                        model_kwargs['beam_search_k'] = beam_search_k
                    if include_answer:
                        model_kwargs['max_answer_length'] = answers.size(1)
                    else:                    
                        model_kwargs['max_answer_length'] = max_answer_length

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision

                model_output = model(**model_kwargs)
                
                pred_vinbig_logits = model_output[f'pred_{findings_name}']
                pred_vinbig_probs = model_output[f'pred_{findings_name}_probs']
                if predict_bboxes_vinbig:
                    assert using_yolov8
                    yolov8_features = model_output['yolov8_features']
                    if not training:
                        yolov8_predictions = model_output['yolov8_predictions']

                if not use_visual_module_only:
                    if training:
                        pred_answer_logits = model_output['pred_answers']
                    else:
                        pred_answers = model_output['pred_answers']

                if training:
                    # Compute losses
                    vinbig_label_loss = vinbig_criterion(pred_vinbig_logits, vinbig_labels.float())                    
                    batch_loss = vinbig_label_loss
                    
                    if predict_bboxes_vinbig:
                        assert using_yolov8
                        batch_size = images.shape[0]
                        assert batch_size == yolov8_features[0].shape[0]
                        vinbig_yolov8_loss, yolov8_loss_items = vinbig_yolov8_criterion(yolov8_features, batch)
                        vinbig_yolov8_loss /= batch_size
                        batch_loss += vinbig_yolov8_loss

                    if not use_visual_module_only:
                        if shift_answer:
                            answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers_end.reshape(-1))
                        else:
                            answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers.view(-1))
                        batch_loss += answer_loss
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss)
        
        # Prepare output
        output = {
            'idxs': idxs,
            'dataset_id': dataset_id,
        }            
        if training:
            output['loss'] = batch_loss.detach()
        # vinbig labels
        output[f'vinbig_labels'] = vinbig_labels.detach().cpu()
        output[f'pred_vinbig_labels'] = (pred_vinbig_logits.detach() > 0).cpu()
        output[f'pred_vinbig_probs'] = pred_vinbig_probs.detach().cpu()
        if training:
            output[f'vinbig_label_loss'] = vinbig_label_loss.detach()
        if predict_bboxes_vinbig:
            assert using_yolov8
            if training:
                output[MetricNames.YOLOV8_LOSS] = vinbig_yolov8_loss.detach()
                output[MetricNames.YOLOV8_BOX_LOSS] = yolov8_loss_items[0]
                output[MetricNames.YOLOV8_CLS_LOSS] = yolov8_loss_items[1]
                output[MetricNames.YOLOV8_DFL_LOSS] = yolov8_loss_items[2]
            else:
                # normalize yolov8 predictions
                resized_shapes = batch['resized_shape']
                assert len(resized_shapes) == len(yolov8_predictions)
                for i in range(len(resized_shapes)):
                    resized_shape = resized_shapes[i]
                    pred = yolov8_predictions[i].detach().cpu()
                    pred[:, :4] /= torch.tensor([resized_shape[1], resized_shape[0], resized_shape[1], resized_shape[0]], dtype=torch.float32)
                    yolov8_predictions[i] = pred
                output['yolov8_predictions'] = yolov8_predictions
                output['vinbig_bbox_coords'] = vinbig_bbox_coords
                output['vinbig_bbox_classes'] = vinbig_bbox_classes

        # answers (vqa)
        if not use_visual_module_only:
            if training:
                pred_answers = pred_answer_logits.argmax(-1)        
                output['answer_loss'] = answer_loss.detach()            
            output['pred_answers'] = tokenizer.clean_batch(pred_answers.detach())
            output['answers'] = tokenizer.clean_batch(answers.detach())                

        return output

    def step_fn__padchest(batch):
            
        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        padchest_labels = batch['l'].to(device)
        padchest_loc = batch['loc'].to(device)
        if classify_orientation:
            orientations = batch['o'].to(device)
        if classify_gender:
            genders = batch['g'].to(device)
        if not use_visual_module_only:
            questions = batch['q'].to(device)
            answers = batch['a'].to(device)
            if shift_answer:
                answers_start = answers[:, :-1]
                answers_end = answers[:, 1:]
            if include_image:
                images = batch['i'].to(device)

        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'padchest_forward': True,
                'device': device,
                'mode': 'train' if training else 'eval',
            }

            if include_image:
                model_kwargs['raw_images'] = images

            if not use_visual_module_only:            
                model_kwargs['questions'] = questions
                if training:
                    if shift_answer:
                        model_kwargs['answers'] = answers_start
                    else:
                        model_kwargs['answers'] = answers
                else:
                    if use_beam_search:
                        model_kwargs['beam_search_k'] = beam_search_k
                    if include_answer:
                        model_kwargs['max_answer_length'] = answers.size(1)
                    else:                    
                        model_kwargs['max_answer_length'] = max_answer_length

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision

                model_output = model(**model_kwargs)
                
                pred_padchest_labels_logits = model_output['pred_padchest_labels']
                pred_padchest_labels_probs = model_output['pred_padchest_labels_probs']
                pred_padchest_loc_logits = model_output['pred_padchest_loc']
                pred_padchest_loc_probs = model_output['pred_padchest_loc_probs']
                if classify_orientation:
                    pred_orientation_logits = model_output['pred_orientation']
                if classify_gender:
                    pred_gender_logits = model_output['pred_gender']
                if not use_visual_module_only:
                    if training:
                        pred_answer_logits = model_output['pred_answers']
                    else:
                        pred_answers = model_output['pred_answers']

                if training:
                    # Compute losses
                    losses = []
                    padchest_label_loss = padchest_multilabel_criterion(pred_padchest_labels_logits, padchest_labels.float())
                    losses.append(padchest_label_loss)
                    padchest_loc_loss = padchest_multilabel_criterion(pred_padchest_loc_logits, padchest_loc.float())
                    losses.append(padchest_loc_loss)
                    if classify_orientation:
                        orientation_loss = padchest_singlelabel_criterion(pred_orientation_logits, orientations)
                        losses.append(orientation_loss)
                    if classify_gender:
                        gender_loss = padchest_singlelabel_criterion(pred_gender_logits, genders)
                        losses.append(gender_loss)
                    if not use_visual_module_only:                    
                        if shift_answer:
                            answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers_end.reshape(-1))
                        else:
                            answer_loss = nlg_criterion(pred_answer_logits.reshape(-1, pred_answer_logits.shape[-1]), answers.view(-1))
                        losses.append(answer_loss)
                    # Backward pass + optimizer step
                    assert len(losses) > 0
                    batch_loss = sum(losses)
                    gradient_accumulator.step(batch_loss)

        output = {
            'idxs': idxs,
            'dataset_id': dataset_id,
        }            
        if training:
            output['loss'] = batch_loss.detach()
            
        # padchest labels and localizations
        output['padchest_labels'] = padchest_labels.detach().cpu()
        output['pred_padchest_labels'] = (pred_padchest_labels_logits.detach() > 0).cpu()
        output['pred_padchest_probs'] = pred_padchest_labels_probs.detach().cpu()
        output['padchest_loc'] = padchest_loc.detach().cpu()
        output['pred_padchest_loc'] = (pred_padchest_loc_logits.detach() > 0).cpu()
        output['pred_padchest_loc_probs'] = pred_padchest_loc_probs.detach().cpu()
        if training:
            output['padchest_label_loss'] = padchest_label_loss.detach()
            output['padchest_loc_loss'] = padchest_loc_loss.detach()
        if classify_orientation:
            output['orientation'] = orientations.detach()
            output['pred_orientation'] = pred_orientation_logits.argmax(-1).detach()
            if training:
                output['orientation_loss'] = orientation_loss.detach()
        if classify_gender:
            output['gender'] = genders.detach()
            output['pred_gender'] = pred_gender_logits.argmax(-1).detach()
            if training:
                output['gender_loss'] = gender_loss.detach()
        # answers (vqa)
        if not use_visual_module_only:
            if training:
                pred_answers = pred_answer_logits.argmax(-1)
                output['answer_loss'] = answer_loss.detach()
            output['pred_answers'] = tokenizer.clean_batch(pred_answers.detach())
            output['answers'] = tokenizer.clean_batch(answers.detach())                

        return output

    _mim_iu_datasets = [MIMICCXR_DATASET_ID, IUXRAY_DATASET_ID,
             MIMICCXR_DATASET_ID__CHEXPERT_MODE, IUXRAY_DATASET_ID__CHEXPERT_MODE,
             MIMICCXR_DATASET_ID__CHEST_IMAGENOME_MODE]

    _chexpert_cxr14_datsets = [CHEXPERT_DATASET_ID, CXR14_DATASET_ID]
    
    def step_fn(unused_engine, batch):
        dataset_id = batch['dataset_id']
        # print(f"step_fn(dataset_id={dataset_id})")
        if dataset_id in _mim_iu_datasets:
            output = step_fn__mimiccxr_iuxray(batch)
        elif dataset_id in _chexpert_cxr14_datsets:
            output = step_fn__chexpert_cxr14(batch)
        elif dataset_id == VINBIG_DATASET_ID:
            output = step_fn__vinbig(batch)
        elif dataset_id == PADCHEST_DATASET_ID:
            output = step_fn__padchest(batch)
        elif dataset_id == MIMICCXR_DATASET_ID__CHEST_IMAGENOME__DETECTRON2_MODE:
            output = step_fn__mimiccxr_chest_imagenome_detectron2(batch)
        else: assert False, f'Unknown dataset_id {dataset_id}'
        # update learning rate batchwise
        if update_lr_batchwise:
            lr_scheduler.step()
        return output
    
    return step_fn

def _get_dataset_masks(dataset_id, labels_remapper, n_labels, device):
    mask = [0] * n_labels
    new_labels = labels_remapper[dataset_id]
    for i in range(len(new_labels)):
        mask[new_labels[i]] = 1
    mask = torch.tensor(mask).to(device)
    return mask

def get_engine(model, classify_tags, classify_orientation, classify_gender,
                classify_chexpert, classify_questions, classify_chest_imagenome,
                predict_bboxes_chest_imagenome, pass_pred_bbox_coords_as_input,
                predict_bboxes_vinbig,
                device,
                tokenizer=None,
                question_encoding=None,
                answer_decoding=None,
                iters_to_accumulate=1,
                binary_loss_name='bce',
                focal_loss_weight=None,
                bce_loss_weight=None,
                wbce_loss_weight=None,
                include_image=True, include_visual_features=False,
                shift_answer=False, include_answer=True,
                beam_search_k=None, max_answer_length=None,
                use_amp=False,
                training=False,
                use_chexpert_dataset=False,
                use_cxr14_dataset=False,
                chexpert_mode=None,
                use_vinbig_dataset=False,
                use_padchest_dataset=False,
                chest_imagenome_bbox_loss_weight=1.0,
                valid_chest_imagenome_label_indices=None,
                optimizer=None,
                update_lr_batchwise=False, lr_scheduler=None,
                use_merged_findings=False, findings_remapper=None, n_findings=None,
                use_visual_module_only=False,
                detectron2_includes_rpn=False,
                using_yolov8=False,
                yolov8_use_multiple_detection_layers=False,
                model_for_yolov8=None,
                **unused_kwargs,
            ):
    
    print(f'get_engine(): shift_answer={shift_answer}')
    
    # Criterion
    if not use_visual_module_only:
        nlg_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding in loss
    else:
        nlg_criterion = None

    if use_merged_findings and training:
        assert binary_loss_name == 'wbce-c'
        assert findings_remapper is not None
        assert n_findings is not None
    
    if binary_loss_name == 'focal+bce+wbce-c':
        assert focal_loss_weight is not None
        assert bce_loss_weight is not None
        assert wbce_loss_weight is not None
        binary_loss_kwargs = {
            'focal_weight': focal_loss_weight,
            'bce_weight': bce_loss_weight,
            'wbce_weight': wbce_loss_weight,
        }
        print('Using focal+bce+wbce-c loss')
        print('binary_loss_kwargs:', binary_loss_kwargs)
    else:
        binary_loss_kwargs = {}
    
    # Auxiliary tasks
    if training and classify_tags:
        tags_criterion = get_binary_multilabel_loss(binary_loss_name, **binary_loss_kwargs)
    else:
        tags_criterion = None
    
    if training and classify_orientation:
        iuxray_orientation_criterion = nn.CrossEntropyLoss()
        mimiccxr_orientation_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore unknown
    else:
        iuxray_orientation_criterion = None
        mimiccxr_orientation_criterion = None
    
    if training and classify_questions:
        question_criterion = get_binary_multilabel_loss(binary_loss_name, **binary_loss_kwargs)
    else:
        question_criterion = None
    
    if training and classify_gender:
        chest_imagenome_gender_criterion = nn.CrossEntropyLoss(ignore_index=2) # ignore unknown
    else:
        chest_imagenome_gender_criterion = None

    if training and (use_chexpert_dataset or use_cxr14_dataset or use_cxr14_dataset):
        chexpert_aux_criterion = nn.CrossEntropyLoss()
        assert chexpert_mode is not None
    else:
        chexpert_aux_criterion = None

    if training and classify_chexpert:
        if use_merged_findings:
            chexpert_mask = _get_dataset_masks(CHEXPERT_DATASET_ID, findings_remapper, n_findings, device)
            chexpert_criterion = get_binary_multilabel_loss(binary_loss_name, classes_mask=chexpert_mask, **binary_loss_kwargs)
        else:
            chexpert_criterion = get_binary_multilabel_loss(binary_loss_name, **binary_loss_kwargs)
    else:
        chexpert_criterion = None

    if training and use_cxr14_dataset:
        if use_merged_findings:
            cxr14_mask = _get_dataset_masks(CXR14_DATASET_ID, findings_remapper, n_findings, device)
            cxr14_criterion = get_binary_multilabel_loss(binary_loss_name, classes_mask=cxr14_mask, **binary_loss_kwargs)
        else:
            cxr14_criterion = get_binary_multilabel_loss(binary_loss_name, **binary_loss_kwargs)
    else:
        cxr14_criterion = None
    
    if training and use_vinbig_dataset:
        if use_merged_findings:
            vinbig_mask = _get_dataset_masks(VINBIG_DATASET_ID, findings_remapper, n_findings, device)
            vinbig_criterion = get_binary_multilabel_loss(binary_loss_name, classes_mask=vinbig_mask, **binary_loss_kwargs)
        else:
            vinbig_criterion = get_binary_multilabel_loss(binary_loss_name, **binary_loss_kwargs)
    else:
        vinbig_criterion = None

    if training and use_padchest_dataset:
        padchest_multilabel_criterion = get_binary_multilabel_loss(binary_loss_name, **binary_loss_kwargs)
        padchest_singlelabel_criterion = nn.CrossEntropyLoss()
    else:
        padchest_multilabel_criterion = None
        padchest_singlelabel_criterion = None

    if training and classify_chest_imagenome:
        chest_imagenome_multilabel_criterion = get_binary_multilabel_loss(binary_loss_name, **binary_loss_kwargs)
    else:
        chest_imagenome_multilabel_criterion = None

    if predict_bboxes_chest_imagenome:
        chest_imagenome_bbox_coords_criterion = nn.MSELoss()
        chest_imagenome_bbox_presence_criterion = nn.BCEWithLogitsLoss()
    else:
        chest_imagenome_bbox_coords_criterion = None
        chest_imagenome_bbox_presence_criterion = None

    if training and using_yolov8:
        assert model_for_yolov8 is not None        
        from ultralytics.yolo.utils.torch_utils import de_parallel
        if yolov8_use_multiple_detection_layers:
            print_magenta('Using YOLOv8MultiDetectionLayersLoss', bold=True)
            from medvqa.losses.yolov8_custom_loss import YOLOV8MultiDetectionLayersLoss
            yolov8_criterion = YOLOV8MultiDetectionLayersLoss(de_parallel(model_for_yolov8))
        else:
            from ultralytics.yolo.v8.detect.train import Loss
            yolov8_criterion = Loss(de_parallel(model_for_yolov8))
    else:
        yolov8_criterion = None

    # Create engine
    step_fn = get_step_fn(model, optimizer, nlg_criterion, tokenizer,
                            use_visual_module_only=use_visual_module_only,
                            include_visual_features=include_visual_features,
                            include_image=include_image, include_answer=include_answer,
                            max_answer_length=max_answer_length,
                            training=training,
                            device=device, use_amp=use_amp,
                            question_encoding=question_encoding,
                            answer_decoding=answer_decoding,
                            beam_search_k=beam_search_k,
                            shift_answer=shift_answer,
                            use_merged_findings=use_merged_findings,
                            iters_to_accumulate=iters_to_accumulate,
                            # tags auxiliary task
                            classify_tags=classify_tags,
                            tags_criterion=tags_criterion,
                            # orientation auxiliary task
                            classify_orientation=classify_orientation,
                            iuxray_orientation_criterion=iuxray_orientation_criterion,
                            mimiccxr_orientation_criterion=mimiccxr_orientation_criterion,
                            # gender auxiliary task
                            classify_gender=classify_gender,
                            chest_imagenome_gender_criterion=chest_imagenome_gender_criterion,
                            # chexpert auxiliary task
                            classify_chexpert=classify_chexpert,
                            chexpert_criterion=chexpert_criterion,
                            # question auxiliary task
                            classify_questions=classify_questions,
                            question_criterion=question_criterion,
                            # chexpert dataset
                            chexpert_aux_criterion=chexpert_aux_criterion,
                            chexpert_mode=chexpert_mode,
                            # cxr14 dataset
                            cxr14_criterion=cxr14_criterion,
                            # vinbig dataset
                            vinbig_criterion=vinbig_criterion,
                            predict_bboxes_vinbig=predict_bboxes_vinbig,
                            # padchest dataset
                            padchest_multilabel_criterion=padchest_multilabel_criterion,
                            padchest_singlelabel_criterion=padchest_singlelabel_criterion,
                            # chest imagenome dataset
                            classify_chest_imagenome=classify_chest_imagenome,
                            chest_imagenome_multilabel_criterion=chest_imagenome_multilabel_criterion,
                            predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
                            chest_imagenome_bbox_coords_criterion=chest_imagenome_bbox_coords_criterion,
                            chest_imagenome_bbox_presence_criterion=chest_imagenome_bbox_presence_criterion,
                            chest_imagenome_bbox_loss_weight=chest_imagenome_bbox_loss_weight,
                            pass_pred_bbox_coords_as_input=pass_pred_bbox_coords_as_input,
                            valid_chest_imagenome_label_indices=valid_chest_imagenome_label_indices,
                            # detectron2
                            detectron2_includes_rpn=detectron2_includes_rpn,
                            # yolov8
                            using_yolov8=using_yolov8,
                            yolov8_criterion=yolov8_criterion,
                            yolov8_use_multiple_detection_layers=yolov8_use_multiple_detection_layers,
                            # batchwise learning rate updates
                            update_lr_batchwise=update_lr_batchwise,
                            lr_scheduler=lr_scheduler,
                        )
    engine = Engine(step_fn)
    return engine