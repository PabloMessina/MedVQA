import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.datasets.vinbig.vinbig_dataset_management import VinBigPhraseTaskMode
from medvqa.losses import BinaryMultiLabelClassificationLossNames, get_binary_multilabel_loss
from medvqa.losses.custom_bbox_loss import compute_bbox_loss
from medvqa.losses.nt_xent_loss import NTXentLoss
from medvqa.losses.optimizers import GradientAccumulator
from medvqa.losses.segmentation_loss import compute_balanced_segmentation_loss
from medvqa.losses.threshold_loss import ThresholdLoss
from medvqa.losses.wbce import NegativePositiveBalancedBCELoss, WeightedNegativePositiveBalancedBCELoss
from medvqa.losses.weakly_supervised_presence_loss import WeaklySupervisedPresenceLoss
import logging

logger = logging.getLogger(__name__)

def get_step_fn(model, optimizer, training, validating, testing, device,
                mimiccxr_phrase_classifier_criterion,
                binary_multilabel_classification_criterion,
                generic_phrase_classifier_criterion,
                contrastive_phrase_grounding_criterion,
                global_image_phrase_contrastive_criterion,
                balanced_binary_cross_entropy_criterion,
                weighted_balanced_binary_cross_entropy_criterion,
                threshold_criterion,
                weakly_supervised_presence_criterion,
                neg_area_prior,
                gradient_accumulation_steps=1, # for gradient accumulation
                max_grad_norm=None,
                # automatic mixed precision
                use_amp=False,
                # yolov8
                using_yolov8=False,
                yolov8_criterion=None,
                yolov8_use_multiple_detection_layers=False,
                # batchwise learning rate updates
                update_lr_batchwise=False,
                lr_scheduler=None,
                # loss weights
                attention_supervision_loss_weight=1.0,
                visual_grounding_confidence_loss_weight=1.0,
                phrase_classifier_loss_weight=1.0,
                foreground_loss_weight=1.0,
                background_loss_weight=1.0,
                # other loss args
                use_weighted_phrase_classifier_loss=False,
                use_attention_regularization_loss=False,
                use_contrastive_phrase_grounding_loss=False,
                use_global_image_phrase_contrastive_loss=False,
                # other args
                skip_nms=False,
                vinbig_task_mode=None,
                mscxr_do_grounding_only=False,
                ):

    scaler = GradScaler(enabled=use_amp)

    # using_yolo = using_yolov8 or using_yolov11

    if update_lr_batchwise:
        assert lr_scheduler is not None

    if yolov8_use_multiple_detection_layers:
        logger.info('Using multiple detection layers in yolov8')
        # mimiccxr_yolov8_index = 0
        # vinbig_yolov8_index = 1

    if using_yolov8 and training:
        assert yolov8_criterion is not None
        # if yolov8_use_multiple_detection_layers:
        #     mimiccxr_yolov8_criterion = lambda *args: yolov8_criterion(*args, mimiccxr_yolov8_index)
        #     vinbig_yolov8_criterion = lambda *args: yolov8_criterion(*args, vinbig_yolov8_index)
        # else:
        #     mimiccxr_yolov8_criterion = vinbig_yolov8_criterion = yolov8_criterion

    if training:
        gradient_accumulator = GradientAccumulator(optimizer, scaler, gradient_accumulation_steps, max_grad_norm)

    # if use_vinbig_with_modified_labels:
    #     vinbig_num_bbox_classes = len(VINBIG_BBOX_NAMES__MODIFIED) # 23
    # else:
    #     vinbig_num_bbox_classes = VINBIG_NUM_BBOX_CLASSES # 22

    def _step_fn__grounding_only__one_phrase_per_image(batch):

        # Extract elements from batch
        images = batch['i'].to(device) # (batch_size, 3, H, W)
        phrase_embeddings = batch['pe'].to(device) # (batch_size, embedding_dim)
        if training:
            target_bbox_coords = batch['tbc'].to(device) # (batch_size, num_regions, 4)
            target_bbox_presence = batch['tbp'].to(device) # (batch_size, num_regions)
            target_prob_mask = batch['tpm'].to(device) # (batch_size, num_regions)
        else:
            bboxes = batch['bboxes']
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'phrase_embeddings': phrase_embeddings.unsqueeze(1), # (batch_size, 1, embedding_dim), add a singleton dimension for num_facts
                'predict_bboxes': True,
                'only_compute_features': True,
                'apply_nms': not training and not skip_nms, # apply NMS during validation/testing
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                
                if training or skip_nms:
                    visual_grounding_bbox_logits = model_output['visual_grounding_bbox_logits'] # (batch_size, 1, num_regions, 4)
                    visual_grounding_bbox_logits = visual_grounding_bbox_logits.squeeze(1) # (batch_size, num_regions, 4)
                    visual_grounding_confidence_logits = model_output['visual_grounding_confidence_logits'] # (batch_size, 1, num_regions, 1)
                    visual_grounding_confidence_logits = visual_grounding_confidence_logits.view(-1, visual_grounding_confidence_logits.shape[-2]) # (batch_size, num_regions)
                else:
                    predicted_bboxes_ = model_output['predicted_bboxes']
                    predicted_bboxes = []
                    for preds in predicted_bboxes_:
                        assert len(preds) == 3 # coords, confs, classes
                        coords, confs, classes = preds
                        assert (classes == 0).all() # only one class
                        predicted_bboxes.append((coords, confs))
                        
                if training:
                    # Compute losses
                    losses = []

                    # 1. Visual grounding confidence loss
                    assert visual_grounding_confidence_logits.shape == target_bbox_presence.shape, (
                        f'{visual_grounding_confidence_logits.shape} != {target_bbox_presence.shape}'
                    )
                    visual_grounding_confidence_loss = weighted_balanced_binary_cross_entropy_criterion(
                        output=visual_grounding_confidence_logits,
                        target=target_bbox_presence,
                        weights=target_prob_mask
                    )
                    visual_grounding_confidence_loss *= visual_grounding_confidence_loss_weight # weight
                    losses.append(visual_grounding_confidence_loss)

                    # 2. Visual grounding bbox regression loss
                    assert visual_grounding_bbox_logits.shape == target_bbox_coords.shape, (
                        f'{visual_grounding_bbox_logits.shape} != {target_bbox_coords.shape}'
                    )
                    visual_grounding_bbox_loss = compute_bbox_loss(
                        pred_bbox_logits=visual_grounding_bbox_logits,
                        gt_bbox_coords=target_bbox_coords,
                        weights=target_prob_mask,
                    )
                    losses.append(visual_grounding_bbox_loss)

                    if len(losses) > 0:
                        batch_loss = sum(losses)
                    else:
                        batch_loss = None

                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

        # Prepare output
        output = {}

        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        if training:
            output['visual_grounding_bbox_loss'] = visual_grounding_bbox_loss.detach()
            output['visual_grounding_confidence_loss'] = visual_grounding_confidence_loss.detach()
        else:
            if skip_nms:
                output['pred_bbox_probs'] = visual_grounding_confidence_logits.detach().sigmoid()
                output['pred_bbox_coords'] = visual_grounding_bbox_logits.detach()
            else:
                output['predicted_bboxes'] = predicted_bboxes
            output['bbox_coords'] = bboxes

        return output
    
    def _step_fn__fact_grounding(batch, phrase_classifier_criterion, use_weighted_phrase_classifier_loss):

        # Extract elements from batch
        if using_yolov8:
            images = batch['img'].to(device)
        else:
            images = batch['i'].to(device)
        phrase_embeddings = batch['pe'].to(device)
        gt_labels = batch['l'].to(device)
        if use_weighted_phrase_classifier_loss:
            phrase_weights = batch['pw'].to(device)

        pos_indices = gt_labels == 1
        neg_indices = gt_labels == 0

        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'phrase_embeddings': phrase_embeddings,
                'only_compute_features': True,
                'compute_global_alignment': use_global_image_phrase_contrastive_loss,
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                phrase_classifier_logits = model_output['phrase_classifier_logits'] # (batch_size, num_facts)
                if use_contrastive_phrase_grounding_loss:
                    phrase_grounding_similarity = model_output['phrase_grounding_similarity'] # (batch_size, num_facts)
                if use_global_image_phrase_contrastive_loss:
                    global_alignment_similarity = model_output['global_alignment_similarity'] # (batch_size, num_facts)
                
                # Compute losses

                # 1. phrase classification loss
                if use_weighted_phrase_classifier_loss:
                    phrase_classifier_loss = phrase_classifier_criterion(phrase_classifier_logits, gt_labels.float(), phrase_weights)
                else:
                    phrase_classifier_loss = phrase_classifier_criterion(phrase_classifier_logits, gt_labels.float())
                phrase_classifier_loss *= phrase_classifier_loss_weight # apply weight

                # 2. contrastive phrase grounding loss
                if use_contrastive_phrase_grounding_loss:
                    phrase_grounding_similarity_pos = phrase_grounding_similarity[pos_indices] # (num_positives)
                    phrase_grounding_similarity_neg = phrase_grounding_similarity[neg_indices] # (num_negatives)
                    phrase_grounding_similarity_pos = phrase_grounding_similarity_pos.view(-1)
                    phrase_grounding_similarity_neg = phrase_grounding_similarity_neg.view(-1)
                    if len(phrase_grounding_similarity_pos) == 0 or len(phrase_grounding_similarity_neg) == 0:
                        contrastive_phrase_grounding_loss = torch.tensor(0.0, device=device) # no positives or negatives
                    else:
                        contrastive_phrase_grounding_loss = contrastive_phrase_grounding_criterion(phrase_grounding_similarity_pos,
                                                                                                phrase_grounding_similarity_neg)
                        
                # 3. global image-phrase contrastive loss
                if use_global_image_phrase_contrastive_loss:
                    global_alignment_similarity_pos = global_alignment_similarity[pos_indices] # (num_positives)
                    global_alignment_similarity_neg = global_alignment_similarity[neg_indices] # (num_negatives)
                    global_alignment_similarity_pos = global_alignment_similarity_pos.view(-1)
                    global_alignment_similarity_neg = global_alignment_similarity_neg.view(-1)
                    if len(global_alignment_similarity_pos) == 0 or len(global_alignment_similarity_neg) == 0:
                        global_alignment_contrastive_loss = torch.tensor(0.0, device=device) # no positives or negatives
                    else:
                        global_alignment_contrastive_loss = global_image_phrase_contrastive_criterion(global_alignment_similarity_pos,
                                                                                                      global_alignment_similarity_neg)

                # 4. attention regularization loss
                if use_attention_regularization_loss:
                    sigmoid_attention = model_output['sigmoid_attention']
                    areas = sigmoid_attention.mean(dim=-1)
                    areas_pos = areas[pos_indices]
                    areas_neg = areas[neg_indices]
                    if len(areas_pos) == 0:
                        attention_regularization_loss_pos = torch.tensor(0.0, device=device)
                    else:
                        attention_regularization_loss_pos = threshold_criterion(areas_pos)
                    if len(areas_neg) == 0:
                        attention_regularization_loss_neg = torch.tensor(0.0, device=device)
                    else:
                        attention_regularization_loss_neg = (areas_neg - neg_area_prior).abs().mean()
                    attention_regularization_loss = attention_regularization_loss_pos + attention_regularization_loss_neg

                if training:
                    losses = []
                    losses.append(phrase_classifier_loss)
                    if use_contrastive_phrase_grounding_loss:
                        losses.append(contrastive_phrase_grounding_loss)
                    if use_global_image_phrase_contrastive_loss:
                        losses.append(global_alignment_contrastive_loss)
                    if use_attention_regularization_loss:
                        losses.append(attention_regularization_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)
                else:
                    batch_loss = None

        # Prepare output
        output = {}

        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        
        output['phrase_classifier_loss'] = phrase_classifier_loss.detach()
        output['classifier_sigmoids'] = phrase_classifier_logits.detach().sigmoid().view(-1)
        output['gt_labels'] = gt_labels.detach().view(-1)
        if use_contrastive_phrase_grounding_loss:
            output['contrastive_phrase_grounding_loss'] = contrastive_phrase_grounding_loss.detach()
        if use_global_image_phrase_contrastive_loss:
            output['global_alignment_contrastive_loss'] = global_alignment_contrastive_loss.detach()
        if use_attention_regularization_loss:
            output['attention_regularization_loss'] = attention_regularization_loss.detach()

        return output
    
    def _step_fn__standard_multilabel_classification(batch):
        # Extract elements from batch
        phrase_embeddings = batch['pe'].to(device)
        phrase_classification_labels = batch['pcl'].to(device)
        images = batch['i'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'phrase_embeddings': phrase_embeddings,
                'only_compute_features': True,
                'compute_global_alignment': use_global_image_phrase_contrastive_loss,
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                sigmoid_attention = model_output['sigmoid_attention'] # (batch_size, num_facts, HxW)
                phrase_classifier_logits = model_output['phrase_classifier_logits']
                if use_contrastive_phrase_grounding_loss:
                    phrase_grounding_similarity = model_output['phrase_grounding_similarity'] # (batch_size, num_facts)
                if use_global_image_phrase_contrastive_loss:
                    global_alignment_similarity = model_output['global_alignment_similarity'] # (batch_size, num_facts)

                if use_contrastive_phrase_grounding_loss or use_global_image_phrase_contrastive_loss or use_attention_regularization_loss:
                    pos_indices = phrase_classification_labels == 1
                    neg_indices = phrase_classification_labels == 0

                # Compute losses
                    
                # 1. phrase classification loss
                phrase_classifier_loss = binary_multilabel_classification_criterion(phrase_classifier_logits, phrase_classification_labels.float())
                phrase_classifier_loss *= phrase_classifier_loss_weight # weight

                # 2. contrastive phrase grounding loss
                if use_contrastive_phrase_grounding_loss:
                    phrase_grounding_similarity_pos = phrase_grounding_similarity[pos_indices] # (num_positives)
                    phrase_grounding_similarity_neg = phrase_grounding_similarity[neg_indices] # (num_negatives)
                    phrase_grounding_similarity_pos = phrase_grounding_similarity_pos.view(-1)
                    phrase_grounding_similarity_neg = phrase_grounding_similarity_neg.view(-1)
                    if len(phrase_grounding_similarity_pos) == 0 or len(phrase_grounding_similarity_neg) == 0:
                        contrastive_phrase_grounding_loss = torch.tensor(0.0, device=device) # no positives or negatives
                    else:
                        contrastive_phrase_grounding_loss = contrastive_phrase_grounding_criterion(phrase_grounding_similarity_pos,
                                                                                                phrase_grounding_similarity_neg)
                        
                # 3. global image-phrase contrastive loss
                if use_global_image_phrase_contrastive_loss:
                    global_alignment_similarity_pos = global_alignment_similarity[pos_indices] # (num_positives)
                    global_alignment_similarity_neg = global_alignment_similarity[neg_indices] # (num_negatives)
                    global_alignment_similarity_pos = global_alignment_similarity_pos.view(-1)
                    global_alignment_similarity_neg = global_alignment_similarity_neg.view(-1)
                    if len(global_alignment_similarity_pos) == 0 or len(global_alignment_similarity_neg) == 0:
                        global_alignment_contrastive_loss = torch.tensor(0.0, device=device) # no positives or negatives
                    else:
                        global_alignment_contrastive_loss = global_image_phrase_contrastive_criterion(global_alignment_similarity_pos,
                                                                                                      global_alignment_similarity_neg)
                
                # 4. attention regularization loss
                if use_attention_regularization_loss:
                    areas = sigmoid_attention.mean(dim=-1)
                    areas_pos = areas[pos_indices]
                    areas_neg = areas[neg_indices]
                    if len(areas_pos) == 0:
                        attention_regularization_loss_pos = torch.tensor(0.0, device=device)
                    else:
                        attention_regularization_loss_pos = threshold_criterion(areas_pos)
                    if len(areas_neg) == 0:
                        attention_regularization_loss_neg = torch.tensor(0.0, device=device)
                    else:
                        attention_regularization_loss_neg = (areas_neg - neg_area_prior).abs().mean()
                    attention_regularization_loss = attention_regularization_loss_pos + attention_regularization_loss_neg
                
                if training:
                    losses = []
                    losses.append(phrase_classifier_loss)
                    if use_contrastive_phrase_grounding_loss:
                        losses.append(contrastive_phrase_grounding_loss)
                    if use_global_image_phrase_contrastive_loss:
                        losses.append(global_alignment_contrastive_loss)
                    if use_attention_regularization_loss:
                        losses.append(attention_regularization_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)
                else:
                    batch_loss = None

        # Prepare output
        output = {}

        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        
        output['phrase_classifier_loss'] = phrase_classifier_loss.detach()
        output['classifier_sigmoids'] = phrase_classifier_logits.detach().sigmoid()
        output['gt_labels'] = phrase_classification_labels.detach()
        if use_contrastive_phrase_grounding_loss:
            output['contrastive_phrase_grounding_loss'] = contrastive_phrase_grounding_loss.detach()
        if use_global_image_phrase_contrastive_loss:
            output['global_alignment_contrastive_loss'] = global_alignment_contrastive_loss.detach()
        if use_attention_regularization_loss:
            output['attention_regularization_loss'] = attention_regularization_loss.detach()

        return output
    
    def step_fn__mimiccxr_fact_grounding(batch):
        return _step_fn__fact_grounding(batch, mimiccxr_phrase_classifier_criterion, use_weighted_phrase_classifier_loss)
    
    def step_fn__iuxray_fact_grounding(batch):
        return _step_fn__fact_grounding(batch, generic_phrase_classifier_criterion, False)
    
    def step_fn__phrase_grounding(batch):

        # Extract elements from batch
        if using_yolov8:
            images = batch['img'].to(device)
        else:
            images = batch['i'].to(device)
        phrase_embeddings = batch['pe'].to(device)
        phrase_grounding_masks = batch['pgm'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'phrase_embeddings': phrase_embeddings,
                'only_compute_features': True,
                'skip_phrase_classifier': True,
                'mimiccxr_forward': True,
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                sigmoid_attention = model_output['sigmoid_attention']
                
                if training:
                    # Compute losses
                    losses = []

                    # attention supervision loss
                    attention_supervision_loss = compute_balanced_segmentation_loss(sigmoid_attention, phrase_grounding_masks,
                                                                                    foreground_loss_weight, background_loss_weight)
                    attention_supervision_loss *= attention_supervision_loss_weight # weight
                    losses.append(attention_supervision_loss)

                    if len(losses) > 0:
                        batch_loss = sum(losses)
                    else:
                        batch_loss = None

                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

                else:

                    # Compute attention supervision loss for validation/testing
                    attention_supervision_loss = compute_balanced_segmentation_loss(sigmoid_attention, phrase_grounding_masks,
                                                                                    foreground_loss_weight, background_loss_weight)

        # Prepare output
        output = {}

        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        output['attention_supervision_loss'] = attention_supervision_loss.detach()
        output['pred_mask'] = sigmoid_attention.detach()
        output['gt_mask'] = phrase_grounding_masks.detach()

        return output
    
    def step_fn__chest_imagenome_phrase_grounding(batch):

        # Extract elements from batch
        images = batch['i'].to(device) # (B, 3, H, W)
        phrase_embeddings = batch['pe'].to(device) # (B, embedding_dim)
        target_bbox_presence = batch['tbp'].to(device) # (B, num_regions)
        target_area_ratio = batch['tar'].to(device) # (B)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'phrase_embeddings': phrase_embeddings.unsqueeze(1), # (B, 1, embedding_dim), add a singleton dimension for num_facts
                'only_compute_features': True,
                'skip_phrase_classifier': True,
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                
                visual_grounding_confidence_logits = model_output['visual_grounding_confidence_logits'] # (B, 1, num_regions, 1)
                visual_grounding_confidence_logits = visual_grounding_confidence_logits.view(-1, visual_grounding_confidence_logits.shape[-2]) # (B, num_regions)
                visual_grounding_confidence_probs = visual_grounding_confidence_logits.sigmoid() # (B, num_regions)

                # Weakly supervised presence loss
                assert visual_grounding_confidence_logits.shape == target_bbox_presence.shape, (
                    f'{visual_grounding_confidence_logits.shape} != {target_bbox_presence.shape}'
                )
                visual_grounding_confidence_loss = weakly_supervised_presence_criterion(
                    visual_grounding_confidence_logits=visual_grounding_confidence_logits,
                    visual_grounding_confidence_probs=visual_grounding_confidence_probs,
                    target_presence=target_bbox_presence,
                    target_area_ratio=target_area_ratio,
                )
                visual_grounding_confidence_loss *= visual_grounding_confidence_loss_weight # weight

                if training:
                    batch_loss = visual_grounding_confidence_loss
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

        # Prepare output
        output = {}

        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        output['visual_grounding_confidence_loss'] = visual_grounding_confidence_loss.detach()

        return output
    
    def step_fn__chest_imagenome_anatomical_location_grounding(batch):

        # Extract elements from batch
        images = batch['i'].to(device)  # (B, 3, H, W)
        phrase_embeddings = batch['pe'].to(device) # (B, 36, embedding_dim)
        if training:
            target_bbox_coords = batch['tbc'].to(device) # (num_phrases_with_grounding, num_regions, 4)
            target_bbox_presence = batch['tbp'].to(device) # (num_phrases_with_grounding, num_regions)
            target_prob_mask = batch['tpm'].to(device) # (num_phrases_with_grounding, num_regions)
            gidxs = batch['gidxs'] # (num_phrases_with_grounding)
        else:
            bbox_coords = batch['bboxes'] # List of list of bboxes
            bbox_classes = batch['classes'] # List of list of classes
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'phrase_embeddings': phrase_embeddings,
                'only_compute_features': True,
                'skip_phrase_classifier': True,
                'apply_nms': not training and not skip_nms, # apply NMS during validation/testing
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                
                if training or skip_nms:
                    visual_grounding_bbox_logits = model_output['visual_grounding_bbox_logits'] # (B, 36, num_regions, 4)
                    visual_grounding_confidence_logits = model_output['visual_grounding_confidence_logits'] # (B, 36, num_regions, 1)
                else:
                    predicted_bboxes_ = model_output['predicted_bboxes']
                    predicted_bboxes = []
                    for preds in predicted_bboxes_:
                        assert len(preds) == 3 # coords, confs, classes
                        coords, confs, classes = preds
                        predicted_bboxes.append((coords, confs, classes))
                        
                if training:
                    # Compute losses
                    losses = []

                    # 1. Visual grounding confidence loss
                    visual_grounding_confidence_logits = visual_grounding_confidence_logits.reshape(
                        -1, visual_grounding_confidence_logits.shape[-2]) # (B*36, num_regions)
                    visual_grounding_confidence_logits = visual_grounding_confidence_logits[gidxs] # (num_phrases_with_grounding, num_regions)
                    assert visual_grounding_confidence_logits.shape == target_bbox_presence.shape, (
                        f'{visual_grounding_confidence_logits.shape} != {target_bbox_presence.shape}'
                    )
                    visual_grounding_confidence_loss = weighted_balanced_binary_cross_entropy_criterion(
                        output=visual_grounding_confidence_logits,
                        target=target_bbox_presence,
                        weights=target_prob_mask
                    )
                    visual_grounding_confidence_loss *= visual_grounding_confidence_loss_weight # weight
                    losses.append(visual_grounding_confidence_loss)

                    # 2. Visual grounding bbox regression loss
                    visual_grounding_bbox_logits = visual_grounding_bbox_logits.reshape(
                        -1, visual_grounding_bbox_logits.shape[-2], visual_grounding_bbox_logits.shape[-1]) # (B*36, num_regions, 4)
                    visual_grounding_bbox_logits = visual_grounding_bbox_logits[gidxs] # (num_phrases_with_grounding, num_regions, 4)
                    assert visual_grounding_bbox_logits.shape == target_bbox_coords.shape, (
                        f'{visual_grounding_bbox_logits.shape} != {target_bbox_coords.shape}'
                    )
                    visual_grounding_bbox_loss = compute_bbox_loss(visual_grounding_bbox_logits, target_bbox_coords, target_bbox_presence)
                    losses.append(visual_grounding_bbox_loss)

                    if len(losses) > 0:
                        batch_loss = sum(losses)
                    else:
                        batch_loss = None

                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)
                
                else:
                    # Prepare output for validation/testing
                    if skip_nms:
                        visual_grounding_confidence_logits = visual_grounding_confidence_logits.squeeze(-1) # (B, 36, num_regions)
        

        # Prepare output
        output = {}

        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        if training:
            output['visual_grounding_bbox_loss'] = visual_grounding_bbox_loss.detach()
            output['visual_grounding_confidence_loss'] = visual_grounding_confidence_loss.detach()
        else:
            if skip_nms:
                output['pred_bbox_probs'] = visual_grounding_confidence_logits.detach().sigmoid()
                output['pred_bbox_coords'] = visual_grounding_bbox_logits.detach()
            else:
                output['predicted_bboxes'] = predicted_bboxes
            output['bbox_coords'] = bbox_coords
            output['bbox_classes'] = bbox_classes

        return output
    

    def step_fn__vinbig(batch):
        if vinbig_task_mode == VinBigPhraseTaskMode.CLASSIFICATION.value:
            raise NotImplementedError('Classification only mode is not implemented')
        elif vinbig_task_mode == VinBigPhraseTaskMode.GROUNDING.value:
            return _step_fn__grounding_only__one_phrase_per_image(batch)
        elif vinbig_task_mode == VinBigPhraseTaskMode.CLASSIFICATION_AND_GROUNDING.value:
            raise NotImplementedError('Classification and grounding mode is not implemented')
        else:
            raise ValueError(f'Unknown vinbig_task_mode: {vinbig_task_mode}')
        
    def step_fn__padchest_grounding(batch):
        return _step_fn__grounding_only__one_phrase_per_image(batch)
    
    # TODO: Review and integrate or delete this function
    # def step_fn__vinbig_bbox_grounding(batch):

    #     # Extract elements from batch
    #     images = batch['i'].to(device)
    #     phrase_embeddings = batch['pe'].to(device)
    #     phrase_classification_labels = batch['pcl'].to(device)
    #     if do_visual_grounding_with_bbox_regression:
    #         if training:
    #             if not using_yolo:
    #                 target_coords = batch['btc'].to(device) # (batch_size, num_boxes, num_regions, 4)
    #                 target_presence = batch['btp'].to(device) # (batch_size, num_boxes, num_regions)
    #                 assert target_coords.shape[1] == vinbig_num_bbox_classes
    #                 assert vinbig_num_bbox_classes < phrase_classification_labels.shape[1] # some classes don't have bounding boxes
    #         else:
    #             vinbig_bbox_coords = batch['bboxes']
    #             vinbig_bbox_classes = batch['classes']
    #     elif do_visual_grounding_with_segmentation:
    #         phrase_grounding_masks = batch['pgm'].to(device)
        
    #     with torch.set_grad_enabled(training):

    #         model.train(training)

    #         # Prepare args for model forward
    #         model_kwargs = {
    #             'raw_images': images,
    #             'phrase_embeddings': phrase_embeddings,
    #             'vinbig_forward': True,
    #             'predict_bboxes': do_visual_grounding_with_bbox_regression,
    #             'apply_nms': (do_visual_grounding_with_bbox_regression and not training) and not skip_nms, # apply NMS during validation/testing
    #             'compute_global_alignment': use_global_image_phrase_contrastive_loss,
    #             'batch': batch, # used by YOLO
    #         }
    #         if using_yolo:
    #             model_kwargs['use_first_n_facts_for_detection'] = vinbig_num_bbox_classes # only 22 out of 28 classes have bounding boxes

    #         # Forward pass
    #         with autocast(enabled=use_amp): # automatic mixed precision
    #             model_output = model(**model_kwargs)
    #             if using_yolo:
    #                 phrase_classifier_logits = model_output['classification_logits']
    #             else:
    #                 phrase_classifier_logits = model_output['phrase_classifier_logits']
                
    #             if 'sigmoid_attention' in model_output:
    #                 sigmoid_attention = model_output['sigmoid_attention'] # (batch_size, num_facts, HxW)
                
    #             if do_visual_grounding_with_bbox_regression:
    #                 if using_yolo:
    #                     detect_output = model_output['detection']
    #                 else:
    #                     if training or skip_nms:
    #                         visual_grounding_bbox_logits = model_output['visual_grounding_bbox_logits']
    #                         visual_grounding_binary_logits = model_output['visual_grounding_binary_logits']
    #                     else:
    #                         predicted_bboxes_ = model_output['predicted_bboxes']
    #                         predicted_bboxes = []
    #                         for preds in predicted_bboxes_:
    #                             assert len(preds) == 3 # coords, confs, classes
    #                             coords, confs, classes = preds
    #                             mask = classes < vinbig_num_bbox_classes # only first vinbig_num_bbox_classes classes have bounding boxes
    #                             predicted_bboxes.append((coords[mask], confs[mask], classes[mask]))

    #             elif do_visual_grounding_with_segmentation:
    #                 assert sigmoid_attention.shape[1] > vinbig_num_bbox_classes # some classes don't have bounding boxes
    #                 sigmoid_attention_with_mask = sigmoid_attention[:, :vinbig_num_bbox_classes]
                
    #             if training:
    #                 # Compute losses
    #                 losses = []
                    
    #                 # 1. phrase classification loss
    #                 phrase_classifier_loss = binary_multilabel_classification_criterion(phrase_classifier_logits, phrase_classification_labels)
    #                 phrase_classifier_loss *= phrase_classifier_loss_weight # weight
    #                 losses.append(phrase_classifier_loss)

    #                 # 2.1 attention supervision loss
    #                 if not using_yolo:
    #                     if do_visual_grounding_with_bbox_regression:
    #                         assert visual_grounding_binary_logits.shape[1] > vinbig_num_bbox_classes # only 22 out of 28 classes have bounding boxes
    #                         visual_grounding_binary_logits = visual_grounding_binary_logits[:, :vinbig_num_bbox_classes] # only first num_boxes classes have bounding boxes
    #                         visual_grounding_binary_logits = visual_grounding_binary_logits.contiguous() # (batch_size, num_boxes, num_regions)
    #                         attention_supervision_loss = balanced_binary_cross_entropy_criterion(visual_grounding_binary_logits, target_presence)
    #                         attention_supervision_loss *= attention_supervision_loss_weight # weight
    #                         losses.append(attention_supervision_loss)
    #                     elif do_visual_grounding_with_segmentation:
    #                         attention_supervision_loss = compute_balanced_segmentation_loss(sigmoid_attention_with_mask, phrase_grounding_masks,
    #                                                                                         foreground_loss_weight, background_loss_weight)
    #                         attention_supervision_loss *= attention_supervision_loss_weight # weight
    #                         losses.append(attention_supervision_loss)
                    
    #                 # 2.2 attention regularization loss
    #                 if not using_yolo:
    #                     if use_attention_regularization_loss:
    #                         sigmoid_attention_without_mask = sigmoid_attention[:, vinbig_num_bbox_classes:]
    #                         areas = sigmoid_attention_without_mask.mean(dim=-1)
    #                         labels_without_mask = phrase_classification_labels[:, vinbig_num_bbox_classes:]
    #                         pos_indices = labels_without_mask == 1
    #                         neg_indices = labels_without_mask == 0
    #                         areas_pos = areas[pos_indices]
    #                         areas_neg = areas[neg_indices]
    #                         if len(areas_pos) == 0:
    #                             attention_regularization_loss_pos = torch.tensor(0.0, device=device)
    #                         else:
    #                             attention_regularization_loss_pos = threshold_criterion(areas_pos)
    #                         if len(areas_neg) == 0:
    #                             attention_regularization_loss_neg = torch.tensor(0.0, device=device)
    #                         else:
    #                             attention_regularization_loss_neg = (areas_neg - neg_area_prior).abs().mean()
    #                         attention_regularization_loss = attention_regularization_loss_pos + attention_regularization_loss_neg
    #                         losses.append(attention_regularization_loss)

    #                 # 3. visual grounding bbox regression loss
    #                 if not using_yolo:
    #                     if do_visual_grounding_with_bbox_regression:
    #                         assert visual_grounding_bbox_logits.shape[1] > vinbig_num_bbox_classes # only 22 out of 28 classes have bounding boxes
    #                         visual_grounding_bbox_logits = visual_grounding_bbox_logits[:, :vinbig_num_bbox_classes] # only first num_boxes classes have bounding boxes
    #                         visual_grounding_bbox_loss = compute_bbox_loss(visual_grounding_bbox_logits, target_coords, target_presence)
    #                         losses.append(visual_grounding_bbox_loss)

    #                 # 4. global image-phrase contrastive loss
    #                 if use_global_image_phrase_contrastive_loss:
    #                     global_alignment_similarity = model_output['global_alignment_similarity']
    #                     global_alignment_similarity_pos = global_alignment_similarity[phrase_classification_labels == 1]
    #                     global_alignment_similarity_neg = global_alignment_similarity[phrase_classification_labels == 0]
    #                     if len(global_alignment_similarity_pos) == 0 or len(global_alignment_similarity_neg) == 0:
    #                         global_alignment_contrastive_loss = torch.tensor(0.0, device=device)
    #                     else:
    #                         global_alignment_contrastive_loss = global_image_phrase_contrastive_criterion(global_alignment_similarity_pos,
    #                                                                                                       global_alignment_similarity_neg)
    #                     losses.append(global_alignment_contrastive_loss)

    #                 # 5. YOLO-specific losses
    #                 if using_yolo:
    #                     yolo_loss = detect_output['loss']
    #                     yolo_loss_items = detect_output['loss_items']
    #                     losses.append(yolo_loss)

    #                 if len(losses) > 0:
    #                     batch_loss = sum(losses)
    #                 else:
    #                     batch_loss = None

    #                 # Backward pass + optimizer step if training
    #                 gradient_accumulator.step(batch_loss, model)

    #             else:
                    
    #                 if do_visual_grounding_with_segmentation:
    #                     # Compute attention supervision loss for validation/testing
    #                     attention_supervision_loss = compute_balanced_segmentation_loss(sigmoid_attention_with_mask, phrase_grounding_masks,
    #                                                                                     foreground_loss_weight, background_loss_weight)


    #     # Prepare output
    #     output = {}

    #     if training and batch_loss is not None:
    #         output['loss'] = batch_loss.detach()
    #     if training:
    #         output['phrase_classifier_loss'] = phrase_classifier_loss.detach()
    #         if use_attention_regularization_loss:
    #             output['attention_regularization_loss'] = attention_regularization_loss.detach()
    #         if use_global_image_phrase_contrastive_loss:
    #             output['global_alignment_contrastive_loss'] = global_alignment_contrastive_loss.detach()
    #     if do_visual_grounding_with_bbox_regression:
    #         if training:
    #             if using_yolo:
    #                 output['yolo_loss'] = yolo_loss.detach()
    #                 output['yolo_box_loss'] = yolo_loss_items[0]
    #                 output['yolo_cls_loss'] = yolo_loss_items[1]
    #                 output['yolo_dfl_loss'] = yolo_loss_items[2]
    #             else:
    #                 output['visual_grounding_bbox_loss'] = visual_grounding_bbox_loss.detach()
    #                 output['attention_supervision_loss'] = attention_supervision_loss.detach()
    #         else:
    #             if using_yolo:
    #                 predictions = detect_output
    #                 if skip_nms:
    #                     resized_shapes = batch['resized_shape']
    #                     assert len(resized_shapes) * vinbig_num_bbox_classes == len(predictions)
    #                     output['resized_shape'] = resized_shapes # needed for NMS later
    #                 output['yolo_predictions'] = predictions
    #             else:
    #                 if skip_nms:
    #                     output['pred_bbox_probs'] = visual_grounding_binary_logits[:, :vinbig_num_bbox_classes].detach().sigmoid()
    #                     output['pred_bbox_coords'] = visual_grounding_bbox_logits[:, :vinbig_num_bbox_classes].detach()
    #                 else:
    #                     output['predicted_bboxes'] = predicted_bboxes
    #             output['vinbig_bbox_coords'] = vinbig_bbox_coords
    #             output['vinbig_bbox_classes'] = vinbig_bbox_classes
    #     elif do_visual_grounding_with_segmentation:
    #         output['attention_supervision_loss'] = attention_supervision_loss.detach()
    #         output['pred_mask'] = sigmoid_attention_with_mask.detach()
    #         output['gt_mask'] = phrase_grounding_masks.detach()
    #     output['pred_probs'] = phrase_classifier_logits.detach().sigmoid()
    #     output['gt_labels'] = phrase_classification_labels.detach()

    #     return output
    
    def _step_fn__mscxr_grounding_and_classification(batch):

        # Extract elements from batch
        images = batch['i'].to(device)
        phrase_embeddings = batch['pe'].to(device)
        phrase_classification_labels = batch['pcl'].to(device)
        if training:
            target_coords = batch['btc'].to(device) # (num_phrases_with_grounding, num_regions, 4)
            target_presence = batch['btp'].to(device) # (num_phrases_with_grounding, num_regions)
            gidxs = batch['gidxs'] # (num_phrases_with_grounding)
        else:
            bbox_coords = batch['bboxes']
            bbox_classes = batch['classes']
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'phrase_embeddings': phrase_embeddings,
                'predict_bboxes': True,
                'only_compute_features': True,
                'apply_nms': not training and not skip_nms, # apply NMS during validation/testing
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                phrase_classifier_logits = model_output['phrase_classifier_logits'] # (batch_size, num_facts)
                
                if training or skip_nms:
                    visual_grounding_bbox_logits = model_output['visual_grounding_bbox_logits'] # (batch_size, num_facts, num_regions, 4)
                    visual_grounding_binary_logits = model_output['visual_grounding_binary_logits'] # (batch_size, num_facts, num_regions, 1)
                else:
                    predicted_bboxes_ = model_output['predicted_bboxes']
                    predicted_bboxes = []
                    for preds in predicted_bboxes_:
                        assert len(preds) == 3 # coords, confs, classes
                        coords, confs, classes = preds
                        predicted_bboxes.append((coords, confs, classes))
                        
                if training:
                    # Compute losses
                    losses = []
                    
                    # 1. phrase classification loss
                    phrase_classifier_loss = generic_phrase_classifier_criterion(phrase_classifier_logits, phrase_classification_labels.float())
                    phrase_classifier_loss *= phrase_classifier_loss_weight # weight
                    losses.append(phrase_classifier_loss)

                    # 2. attention supervision loss
                    visual_grounding_binary_logits = visual_grounding_binary_logits.reshape(-1, visual_grounding_binary_logits.shape[-2]) # (batch_size*num_facts, num_regions)
                    visual_grounding_binary_logits = visual_grounding_binary_logits[gidxs] # (num_phrases_with_grounding, num_regions)
                    assert visual_grounding_binary_logits.shape == target_presence.shape, f'{visual_grounding_binary_logits.shape} != {target_presence.shape}'
                    attention_supervision_loss = balanced_binary_cross_entropy_criterion(visual_grounding_binary_logits, target_presence)
                    attention_supervision_loss *= attention_supervision_loss_weight # weight
                    losses.append(attention_supervision_loss)

                    # 3. visual grounding bbox regression loss
                    visual_grounding_bbox_logits = visual_grounding_bbox_logits.reshape(-1, visual_grounding_bbox_logits.shape[-2], visual_grounding_bbox_logits.shape[-1]) # (batch_size*num_facts, num_regions, 4)
                    visual_grounding_bbox_logits = visual_grounding_bbox_logits[gidxs] # (num_phrases_with_grounding, num_regions, 4)
                    assert visual_grounding_bbox_logits.shape == target_coords.shape, f'{visual_grounding_bbox_logits.shape} != {target_coords.shape}'
                    visual_grounding_bbox_loss = compute_bbox_loss(visual_grounding_bbox_logits, target_coords, target_presence)
                    losses.append(visual_grounding_bbox_loss)

                    if len(losses) > 0:
                        batch_loss = sum(losses)
                    else:
                        batch_loss = None

                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

        # Prepare output
        output = {}

        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        if training:
            output['phrase_classifier_loss'] = phrase_classifier_loss.detach()
            output['visual_grounding_bbox_loss'] = visual_grounding_bbox_loss.detach()
            output['attention_supervision_loss'] = attention_supervision_loss.detach()
        else:
            if skip_nms:
                output['pred_bbox_probs'] = visual_grounding_binary_logits.detach().sigmoid()
                output['pred_bbox_coords'] = visual_grounding_bbox_logits.detach()
            else:
                output['predicted_bboxes'] = predicted_bboxes
            output['bbox_coords'] = bbox_coords
            output['bbox_classes'] = bbox_classes
        output['pred_probs'] = phrase_classifier_logits.detach().sigmoid().view(-1)
        output['gt_labels'] = phrase_classification_labels.detach().view(-1)

        return output
    
    def step_fn__mscxr_grounding(batch):
        if mscxr_do_grounding_only:
            return _step_fn__grounding_only__one_phrase_per_image(batch)
        else:
            return _step_fn__mscxr_grounding_and_classification(batch)
    
    def step_fn__chexlocalize(batch):

        # Extract elements from batch
        images = batch['i'].to(device)
        phrase_embeddings = batch['pe'].to(device)
        phrase_classification_labels = batch['pcl'].to(device)
        phrase_grounding_masks = batch['pgm'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'phrase_embeddings': phrase_embeddings,
                'only_compute_features': True,
                'compute_global_alignment': use_global_image_phrase_contrastive_loss,
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                sigmoid_attention = model_output['sigmoid_attention']
                phrase_classifier_logits = model_output['phrase_classifier_logits']
                if use_global_image_phrase_contrastive_loss:
                    global_alignment_similarity = model_output['global_alignment_similarity']
                
                if training:
                    # Compute losses
                    losses = []
                    
                    # 1. phrase classification loss
                    phrase_classifier_loss = binary_multilabel_classification_criterion(phrase_classifier_logits, phrase_classification_labels)
                    phrase_classifier_loss *= phrase_classifier_loss_weight # weight
                    losses.append(phrase_classifier_loss)

                    # 2. attention supervision loss
                    attention_supervision_loss = compute_balanced_segmentation_loss(sigmoid_attention, phrase_grounding_masks,
                                                                                    foreground_loss_weight, background_loss_weight)
                    attention_supervision_loss *= attention_supervision_loss_weight # weight
                    losses.append(attention_supervision_loss)

                    # 3. global image-phrase contrastive loss
                    if use_global_image_phrase_contrastive_loss:
                        global_alignment_similarity_pos = global_alignment_similarity[phrase_classification_labels == 1]
                        global_alignment_similarity_neg = global_alignment_similarity[phrase_classification_labels == 0]
                        if len(global_alignment_similarity_pos) == 0 or len(global_alignment_similarity_neg) == 0:
                            global_alignment_contrastive_loss = torch.tensor(0.0, device=device)
                        else:
                            global_alignment_contrastive_loss = global_image_phrase_contrastive_criterion(global_alignment_similarity_pos,
                                                                                                          global_alignment_similarity_neg)
                        losses.append(global_alignment_contrastive_loss)

                    if len(losses) > 0:
                        batch_loss = sum(losses)
                    else:
                        batch_loss = None

                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

                else:
                    # Compute attention supervision loss for validation/testing
                    attention_supervision_loss = compute_balanced_segmentation_loss(sigmoid_attention, phrase_grounding_masks,
                                                                                    foreground_loss_weight, background_loss_weight)

        # Prepare output
        output = {}

        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        if training:
            output['phrase_classifier_loss'] = phrase_classifier_loss.detach()
            if use_global_image_phrase_contrastive_loss:
                output['global_alignment_contrastive_loss'] = global_alignment_contrastive_loss.detach()
        output['attention_supervision_loss'] = attention_supervision_loss.detach()
        output['pred_mask'] = sigmoid_attention.detach()
        output['gt_mask'] = phrase_grounding_masks.detach()
        output['pred_probs'] = phrase_classifier_logits.detach().sigmoid().view(-1)
        output['pred_labels'] = (phrase_classifier_logits.detach() > 0).view(-1)
        output['gt_labels'] = phrase_classification_labels.detach().view(-1)

        return output
    
    def step_fn__chexpert_phrase_grounding(batch):
        return _step_fn__standard_multilabel_classification(batch)

    def step_fn__iuxray_fact_grounding(batch):

        # Extract elements from batch
        images = batch['i'].to(device)
        phrase_embeddings = batch['pe'].to(device)
        gt_labels = batch['l'].to(device)

        pos_indices = gt_labels == 1
        neg_indices = gt_labels == 0

        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'phrase_embeddings': phrase_embeddings,
                'only_compute_features': True,
                'compute_global_alignment': use_global_image_phrase_contrastive_loss,
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                sigmoid_attention = model_output['sigmoid_attention']
                phrase_classifier_logits = model_output['phrase_classifier_logits'] # (batch_size, num_facts)
                if use_contrastive_phrase_grounding_loss:
                    phrase_grounding_similarity = model_output['phrase_grounding_similarity'] # (batch_size, num_facts)
                if use_global_image_phrase_contrastive_loss:
                    global_alignment_similarity = model_output['global_alignment_similarity']
                
                # Compute losses

                # 1. phrase classification loss
                phrase_classifier_loss = generic_phrase_classifier_criterion(phrase_classifier_logits, gt_labels.float())
                phrase_classifier_loss *= phrase_classifier_loss_weight # apply weight

                # 2. contrastive phrase grounding loss
                if use_contrastive_phrase_grounding_loss:
                    phrase_grounding_similarity_pos = phrase_grounding_similarity[pos_indices] # (num_positives)
                    phrase_grounding_similarity_neg = phrase_grounding_similarity[neg_indices] # (num_negatives)
                    phrase_grounding_similarity_pos = phrase_grounding_similarity_pos.view(-1)
                    phrase_grounding_similarity_neg = phrase_grounding_similarity_neg.view(-1)
                    if len(phrase_grounding_similarity_pos) == 0 or len(phrase_grounding_similarity_neg) == 0:
                        contrastive_phrase_grounding_loss = torch.tensor(0.0, device=device) # no positives or negatives
                    else:
                        contrastive_phrase_grounding_loss = contrastive_phrase_grounding_criterion(phrase_grounding_similarity_pos,
                                                                                                phrase_grounding_similarity_neg)
                        
                # 3. global image-phrase contrastive loss
                if use_global_image_phrase_contrastive_loss:
                    global_alignment_similarity_pos = global_alignment_similarity[pos_indices] # (num_positives)
                    global_alignment_similarity_neg = global_alignment_similarity[neg_indices] # (num_negatives)
                    global_alignment_similarity_pos = global_alignment_similarity_pos.view(-1)
                    global_alignment_similarity_neg = global_alignment_similarity_neg.view(-1)
                    if len(global_alignment_similarity_pos) == 0 or len(global_alignment_similarity_neg) == 0:
                        global_alignment_contrastive_loss = torch.tensor(0.0, device=device) # no positives or negatives
                    else:
                        global_alignment_contrastive_loss = global_image_phrase_contrastive_criterion(global_alignment_similarity_pos,
                                                                                                      global_alignment_similarity_neg)

                # 4. attention regularization loss
                if use_attention_regularization_loss:
                    areas = sigmoid_attention.mean(dim=-1)
                    areas_pos = areas[pos_indices]
                    areas_neg = areas[neg_indices]
                    if len(areas_pos) == 0:
                        attention_regularization_loss_pos = torch.tensor(0.0, device=device)
                    else:
                        attention_regularization_loss_pos = threshold_criterion(areas_pos)
                    if len(areas_neg) == 0:
                        attention_regularization_loss_neg = torch.tensor(0.0, device=device)
                    else:
                        attention_regularization_loss_neg = (areas_neg - neg_area_prior).abs().mean()
                    attention_regularization_loss = attention_regularization_loss_pos + attention_regularization_loss_neg

                if training:
                    losses = []
                    losses.append(phrase_classifier_loss)
                    if use_contrastive_phrase_grounding_loss:
                        losses.append(contrastive_phrase_grounding_loss)
                    if use_global_image_phrase_contrastive_loss:
                        losses.append(global_alignment_contrastive_loss)
                    if use_attention_regularization_loss:
                        losses.append(attention_regularization_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)
                else:
                    batch_loss = None

        # Prepare output
        output = {}

        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        
        output['phrase_classifier_loss'] = phrase_classifier_loss.detach()
        output['classifier_sigmoids'] = phrase_classifier_logits.detach().sigmoid().view(-1)
        output['gt_labels'] = gt_labels.detach().view(-1)
        if use_contrastive_phrase_grounding_loss:
            output['contrastive_phrase_grounding_loss'] = contrastive_phrase_grounding_loss.detach()
        if use_global_image_phrase_contrastive_loss:
            output['global_alignment_contrastive_loss'] = global_alignment_contrastive_loss.detach()
        if use_attention_regularization_loss:
            output['attention_regularization_loss'] = attention_regularization_loss.detach()

        return output
    
    def step_fn__cxrlt2024_custom_labels(batch):

        # Extract elements from batch
        if using_yolov8:
            images = batch['img'].to(device)
        else:
            images = batch['i'].to(device)
        phrase_embeddings = batch['pe'].to(device)
        phrase_indices = batch['pi']
        phrase_classification_labels = batch['pcl'].to(device)

        pos_indices = phrase_classification_labels == 1
        neg_indices = phrase_classification_labels == 0

        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'phrase_embeddings': phrase_embeddings,
                'only_compute_features': True,
                'compute_global_alignment': use_global_image_phrase_contrastive_loss,
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                sigmoid_attention = model_output['sigmoid_attention']
                phrase_classifier_logits = model_output['phrase_classifier_logits'] # (batch_size, num_facts)
                if use_global_image_phrase_contrastive_loss:
                    global_alignment_similarity = model_output['global_alignment_similarity']
                
                # Compute losses

                # 1. phrase classification loss
                phrase_classifier_loss = generic_phrase_classifier_criterion(phrase_classifier_logits, phrase_classification_labels.float())
                phrase_classifier_loss *= phrase_classifier_loss_weight # apply weight

                # 2. attention regularization loss
                if use_attention_regularization_loss:
                    areas = sigmoid_attention.mean(dim=-1)
                    areas_pos = areas[pos_indices]
                    areas_neg = areas[neg_indices]
                    if len(areas_pos) == 0:
                        attention_regularization_loss_pos = torch.tensor(0.0, device=device)
                    else:
                        attention_regularization_loss_pos = threshold_criterion(areas_pos)
                    if len(areas_neg) == 0:
                        attention_regularization_loss_neg = torch.tensor(0.0, device=device)
                    else:
                        attention_regularization_loss_neg = (areas_neg - neg_area_prior).abs().mean()
                    attention_regularization_loss = attention_regularization_loss_pos + attention_regularization_loss_neg

                # 3. global image-phrase contrastive loss
                if use_global_image_phrase_contrastive_loss:
                    global_alignment_similarity_pos = global_alignment_similarity[pos_indices] # (num_positives)
                    global_alignment_similarity_neg = global_alignment_similarity[neg_indices] # (num_negatives)
                    global_alignment_similarity_pos = global_alignment_similarity_pos.view(-1)
                    global_alignment_similarity_neg = global_alignment_similarity_neg.view(-1)
                    if len(global_alignment_similarity_pos) == 0 or len(global_alignment_similarity_neg) == 0:
                        global_alignment_contrastive_loss = torch.tensor(0.0, device=device) # no positives or negatives
                    else:
                        global_alignment_contrastive_loss = global_image_phrase_contrastive_criterion(global_alignment_similarity_pos,
                                                                                                      global_alignment_similarity_neg)

                if training:
                    losses = []
                    losses.append(phrase_classifier_loss)
                    if use_attention_regularization_loss:
                        losses.append(attention_regularization_loss)
                    if use_global_image_phrase_contrastive_loss:
                        losses.append(global_alignment_contrastive_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)
                else:
                    batch_loss = None

        # Prepare output
        output = {}

        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        
        output['phrase_classifier_loss'] = phrase_classifier_loss.detach()
        output['classifier_sigmoids'] = phrase_classifier_logits.detach().sigmoid().view(-1)
        output['gt_labels'] = phrase_classification_labels.detach().view(-1)
        output['phrase_indices'] = phrase_indices.flatten()
        if use_attention_regularization_loss:
            output['attention_regularization_loss'] = attention_regularization_loss.detach()
        if use_global_image_phrase_contrastive_loss:
            output['global_alignment_contrastive_loss'] = global_alignment_contrastive_loss.detach()

        return output
    
    def step_fn__cxrlt2024_official_labels(batch):
        return _step_fn__standard_multilabel_classification(batch)
    
    dataset_name_to_step_fn = {
        'mimfg': step_fn__mimiccxr_fact_grounding, # mimiccxr fact grounding (facts extracted from radiology reports)
        'iufg': step_fn__iuxray_fact_grounding, # iuxray fact grounding (facts extracted from radiology reports)
        'pg': step_fn__phrase_grounding, # phrase grounding (this assumes ground truth masks are available)
        'mscxr': step_fn__mscxr_grounding, # MS-CXR phrase grounding
        'chest-imagenome-pg': step_fn__chest_imagenome_phrase_grounding, # Chest Imagenome phrase grounding
        'chest-imagenome-alg': step_fn__chest_imagenome_anatomical_location_grounding, # Chest Imagenome anatomical location grounding
        'vinbig': step_fn__vinbig, # vinbig bbox grounding
        'padchest_gr': step_fn__padchest_grounding, # padchest grounding
        'cl': step_fn__chexlocalize, # chexlocalize
        'chxp': step_fn__chexpert_phrase_grounding, # chexpert phrase grounding
        'cxrlt2024c': step_fn__cxrlt2024_custom_labels, # MICCAI CXR-LT 2024 challenge with custom labels
        'cxrlt2024o': step_fn__cxrlt2024_official_labels, # MICCAI CXR-LT 2024 challenge with official labels
    }
    
    def step_fn(unused_engine, batch):
        dataset_name = batch['dataset_name']
        output = dataset_name_to_step_fn[dataset_name](batch)
        output['dataset_name'] = dataset_name # propagate dataset name to output
        if update_lr_batchwise: # update learning rate batchwise
            lr_scheduler.step()
        return output
    
    return step_fn

def get_engine(model, device, gradient_accumulation_steps=1,
               use_amp=False, training=False, validating=False,
               testing=False, optimizer=None,
               update_lr_batchwise=False, lr_scheduler=None,
               using_yolov8=False,
               yolov8_use_multiple_detection_layers=False,
               model_for_yolov8=None,
               pos_area_prior=0.2, neg_area_prior=0.0,
               max_grad_norm=None,
               attention_supervision_loss_weight=1.0,
               visual_grounding_confidence_loss_weight=1.0,
               phrase_classifier_loss_weight=1.0,
               foreground_loss_weight=1.0,
               background_loss_weight=1.0,
               binary_multilabel_classif_loss_name='bce',
               focal_loss_weight=None,
               bce_loss_weight=None,
               wbce_loss_weight=None,
               use_attention_regularization_loss=False,
               use_contrastive_phrase_grounding_loss=False,
               use_global_image_phrase_contrastive_loss=False,
               nt_xent_temperature=0.1,
               skip_nms=False,
               vinbig_task_mode=None,
               mscxr_do_grounding_only=False,
               **unused_kwargs,
            ):
    
    if unused_kwargs:
        logger.warning(f'unused_kwargs: {unused_kwargs}')

    # Create multiple criterion objects

    # Binary multi-label loss for mimic-cxr phrase classifier
    assert binary_multilabel_classif_loss_name in [
        BinaryMultiLabelClassificationLossNames.BCE,
        BinaryMultiLabelClassificationLossNames.WBCE,
        BinaryMultiLabelClassificationLossNames.FOCAL,
        BinaryMultiLabelClassificationLossNames.FOCAL_BCE,
        BinaryMultiLabelClassificationLossNames.FOCAL_BCE_WBCE,
        BinaryMultiLabelClassificationLossNames.FOCAL_BCE_NPBBCE,
    ]
    if binary_multilabel_classif_loss_name == BinaryMultiLabelClassificationLossNames.FOCAL_BCE_WBCE:
        assert focal_loss_weight is not None
        assert bce_loss_weight is not None
        assert wbce_loss_weight is not None
        binary_loss_kwargs = {
            'focal_weight': focal_loss_weight,
            'bce_weight': bce_loss_weight,
            'wbce_weight': wbce_loss_weight,
        }
        print('binary_loss_kwargs:', binary_loss_kwargs)
    else:
        binary_loss_kwargs = {}
    if binary_multilabel_classif_loss_name in [
        BinaryMultiLabelClassificationLossNames.WBCE,
        BinaryMultiLabelClassificationLossNames.FOCAL_BCE_WBCE,
    ]:
        use_weighted_phrase_classifier_loss = True
    else:
        use_weighted_phrase_classifier_loss = False
    mimiccxr_phrase_classifier_criterion = get_binary_multilabel_loss(binary_multilabel_classif_loss_name, **binary_loss_kwargs)

    # Generic phrase classifier loss
    generic_phrase_classifier_criterion = get_binary_multilabel_loss(BinaryMultiLabelClassificationLossNames.FOCAL_BCE_NPBBCE)

    # Standard binary multi-label classification loss
    binary_multilabel_classification_criterion = get_binary_multilabel_loss(BinaryMultiLabelClassificationLossNames.FOCAL_BCE_WBCBCE)

    # Contrastive phrase grounding loss
    contrastive_phrase_grounding_criterion = NTXentLoss(temperature=nt_xent_temperature, device=device)

    # Global image-phrase contrastive loss
    global_image_phrase_contrastive_criterion = NTXentLoss(temperature=nt_xent_temperature, device=device)

    # Threshold loss
    threshold_criterion = ThresholdLoss(pos_area_prior)

    # Balanced binary cross-entropy loss
    balanced_binary_cross_entropy_criterion = NegativePositiveBalancedBCELoss()

    # Weighted binary cross-entropy loss
    weighted_balanced_binary_cross_entropy_criterion = WeightedNegativePositiveBalancedBCELoss()

    # Weakly supervised presence loss
    weakly_supervised_presence_criterion = WeaklySupervisedPresenceLoss()

    if training and using_yolov8:
        assert model_for_yolov8 is not None
        from ultralytics.yolo.utils.torch_utils import de_parallel
        if yolov8_use_multiple_detection_layers:
            logger.info('Using YOLOv8MultiDetectionLayersLoss')
            from medvqa.losses.yolov8_custom_loss import YOLOV8MultiDetectionLayersLoss
            yolov8_criterion = YOLOV8MultiDetectionLayersLoss(de_parallel(model_for_yolov8))
        else:
            from ultralytics.yolo.v8.detect.train import Loss
            yolov8_criterion = Loss(de_parallel(model_for_yolov8))
    else:
        yolov8_criterion = None

    logger.info(f'foreground_loss_weight: {foreground_loss_weight}')
    logger.info(f'background_loss_weight: {background_loss_weight}')

    # Create engine
    step_fn = get_step_fn(model, optimizer, training, validating, testing, device,
                            mimiccxr_phrase_classifier_criterion=mimiccxr_phrase_classifier_criterion,
                            binary_multilabel_classification_criterion=binary_multilabel_classification_criterion,
                            generic_phrase_classifier_criterion=generic_phrase_classifier_criterion,
                            contrastive_phrase_grounding_criterion=contrastive_phrase_grounding_criterion,
                            global_image_phrase_contrastive_criterion=global_image_phrase_contrastive_criterion,
                            balanced_binary_cross_entropy_criterion=balanced_binary_cross_entropy_criterion,
                            weighted_balanced_binary_cross_entropy_criterion=weighted_balanced_binary_cross_entropy_criterion,
                            threshold_criterion=threshold_criterion,
                            weakly_supervised_presence_criterion=weakly_supervised_presence_criterion,
                            neg_area_prior=neg_area_prior,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            max_grad_norm=max_grad_norm, use_amp=use_amp,
                            # yolov8
                            using_yolov8=using_yolov8,
                            yolov8_criterion=yolov8_criterion,
                            yolov8_use_multiple_detection_layers=yolov8_use_multiple_detection_layers,
                            # batchwise learning rate updates
                            update_lr_batchwise=update_lr_batchwise,
                            lr_scheduler=lr_scheduler,
                            # loss weights
                            attention_supervision_loss_weight=attention_supervision_loss_weight,
                            visual_grounding_confidence_loss_weight=visual_grounding_confidence_loss_weight,
                            phrase_classifier_loss_weight=phrase_classifier_loss_weight,
                            foreground_loss_weight=foreground_loss_weight,
                            background_loss_weight=background_loss_weight,
                            # other loss args
                            use_weighted_phrase_classifier_loss=use_weighted_phrase_classifier_loss,
                            use_attention_regularization_loss=use_attention_regularization_loss,
                            use_contrastive_phrase_grounding_loss=use_contrastive_phrase_grounding_loss,
                            use_global_image_phrase_contrastive_loss=use_global_image_phrase_contrastive_loss,
                            skip_nms=skip_nms,
                            vinbig_task_mode=vinbig_task_mode,
                            mscxr_do_grounding_only=mscxr_do_grounding_only,
                        )
    engine = Engine(step_fn)
    return engine