import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.losses import BinaryMultiLabelClassificationLossNames, get_binary_multilabel_loss
from medvqa.losses.nt_xent_loss import NTXentLoss
from medvqa.losses.optimizers import GradientAccumulator
from medvqa.losses.segmentation_loss import compute_balanced_segmentation_loss
from medvqa.losses.threshold_loss import ThresholdLoss
from medvqa.utils.constants import MetricNames
from medvqa.utils.logging import print_magenta

def get_step_fn(model, optimizer, training, validating, testing, device,
                mimiccxr_phrase_classifier_criterion,
                binary_multilabel_classification_criterion,
                generic_phrase_classifier_criterion,
                contrastive_phrase_grounding_criterion,
                global_image_phrase_contrastive_criterion,
                threshold_criterion,
                neg_area_prior,
                gradient_accumulation_steps=1, # for gradient accumulation
                max_grad_norm=None,
                # automatic mixed precision
                use_amp=False,
                # chest imagenome dataset
                predict_bboxes_chest_imagenome=False,
                # vinbig dataset
                predict_bboxes_vinbig=False,
                # yolov8
                using_yolov8=False,
                yolov8_criterion=None,
                yolov8_use_multiple_detection_layers=False,
                # batchwise learning rate updates
                update_lr_batchwise=False,
                lr_scheduler=None,
                # loss weights
                attention_supervision_loss_weight=1.0,
                phrase_classifier_loss_weight=1.0,
                foreground_loss_weight=1.0,
                background_loss_weight=1.0,
                # other loss args
                use_weighted_phrase_classifier_loss=False,
                use_attention_regularization_loss=False,
                use_contrastive_phrase_grounding_loss=False,
                use_global_image_phrase_contrastive_loss=False,
                ):

    scaler = GradScaler(enabled=use_amp)

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

    if training:
        gradient_accumulator = GradientAccumulator(optimizer, scaler, gradient_accumulation_steps, max_grad_norm)
    
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
                sigmoid_attention = model_output['sigmoid_attention']
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
    
    def step_fn__chest_imagenome_bbox_grounding(batch):
        
        # assert using_yolov8

        # Extract elements from batch
        if using_yolov8:
            images = batch['img'].to(device)
        else:
            images = batch['i'].to(device)
        phrase_embeddings = batch['pe'].to(device)
        phrase_grounding_masks = batch['pgm'].to(device)
        if not training and using_yolov8:
            chest_imagenome_bbox_coords = batch['bc']
            chest_imagenome_bbox_presence = batch['bp']
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'phrase_embeddings': phrase_embeddings,
                'mimiccxr_forward': True,
            }
            if using_yolov8 and yolov8_use_multiple_detection_layers:
                model_kwargs['yolov8_detection_layer_index'] = mimiccxr_yolov8_index

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                sigmoid_attention = model_output['sigmoid_attention']
                if using_yolov8:
                    yolov8_features = model_output['yolov8_features']
                    if not training:
                        yolov8_predictions = model_output['yolov8_predictions']
                
                if training:
                    # Compute losses
                    losses = []

                    # 1. attention supervision loss
                    attention_supervision_loss = compute_balanced_segmentation_loss(sigmoid_attention, phrase_grounding_masks,
                                                                                    foreground_loss_weight, background_loss_weight)
                    attention_supervision_loss *= attention_supervision_loss_weight # weight
                    losses.append(attention_supervision_loss)

                    # 2. yolov8 loss
                    if using_yolov8 and predict_bboxes_chest_imagenome:
                        batch_size = images.shape[0]
                        assert batch_size == yolov8_features[0].shape[0]
                        chest_imagenome_yolov8_loss, yolov8_loss_items = mimiccxr_yolov8_criterion(yolov8_features, batch)
                        chest_imagenome_yolov8_loss /= batch_size
                        losses.append(chest_imagenome_yolov8_loss)

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
            if using_yolov8:
                output[MetricNames.YOLOV8_LOSS] = chest_imagenome_yolov8_loss.detach()
                output[MetricNames.YOLOV8_BOX_LOSS] = yolov8_loss_items[0]
                output[MetricNames.YOLOV8_CLS_LOSS] = yolov8_loss_items[1]
                output[MetricNames.YOLOV8_DFL_LOSS] = yolov8_loss_items[2]
        else:
            if using_yolov8:
                # normalize yolov8 predictions
                resized_shapes = batch['resized_shape']
                assert len(resized_shapes) == len(yolov8_predictions)
                for i in range(len(resized_shapes)):
                    resized_shape = resized_shapes[i]
                    pred = yolov8_predictions[i].detach().cpu()
                    pred[:, :4] /= torch.tensor([resized_shape[1], resized_shape[0], resized_shape[1], resized_shape[0]], dtype=torch.float32)
                    yolov8_predictions[i] = pred
                output['yolov8_predictions'] = yolov8_predictions
                output['chest_imagenome_bbox_coords'] = chest_imagenome_bbox_coords
                output['chest_imagenome_bbox_presence'] = chest_imagenome_bbox_presence
        output['attention_supervision_loss'] = attention_supervision_loss.detach()
        output['pred_mask'] = sigmoid_attention.detach()
        output['gt_mask'] = phrase_grounding_masks.detach()

        return output
    
    def step_fn__vinbig_bbox_grounding(batch):

        # Extract elements from batch
        phrase_embeddings = batch['pe'].to(device)
        phrase_classification_labels = batch['pcl'].to(device)
        phrase_grounding_masks = batch['pgm'].to(device)
        if using_yolov8:
            images = batch['img'].to(device)
            if not training:
                vinbig_bbox_coords = batch['bboxes']
                vinbig_bbox_classes = batch['classes']
        else:
            images = batch['i'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'phrase_embeddings': phrase_embeddings,
                'vinbig_forward': True,
                'compute_global_alignment': use_global_image_phrase_contrastive_loss,
            }
            if using_yolov8 and yolov8_use_multiple_detection_layers:
                model_kwargs['yolov8_detection_layer_index'] = vinbig_yolov8_index

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                sigmoid_attention = model_output['sigmoid_attention'] # (batch_size, num_facts, HxW)
                assert sigmoid_attention.dim() == 3
                n = phrase_grounding_masks.shape[1] # number of facts with masks
                assert sigmoid_attention.shape[1] > n # NOTE: some facts don't have masks
                sigmoid_attention_with_mask = sigmoid_attention[:, :n] # first n facts will be supervised with ground truth masks
                if use_attention_regularization_loss:                    
                    sigmoid_attention_without_mask = sigmoid_attention[:, n:] # remaining facts will be supervised with attention regularization loss
                phrase_classifier_logits = model_output['phrase_classifier_logits']
                if using_yolov8:
                    yolov8_features = model_output['yolov8_features']
                    if not training:
                        yolov8_predictions = model_output['yolov8_predictions']
                if use_global_image_phrase_contrastive_loss:
                    global_alignment_similarity = model_output['global_alignment_similarity']
                
                if training:
                    # Compute losses
                    losses = []
                    
                    # 1. phrase classification loss
                    phrase_classifier_loss = binary_multilabel_classification_criterion(phrase_classifier_logits, phrase_classification_labels)
                    phrase_classifier_loss *= phrase_classifier_loss_weight # weight
                    losses.append(phrase_classifier_loss)

                    # 2.1 attention supervision loss
                    attention_supervision_loss = compute_balanced_segmentation_loss(sigmoid_attention_with_mask, phrase_grounding_masks,
                                                                                    foreground_loss_weight, background_loss_weight)
                    attention_supervision_loss *= attention_supervision_loss_weight # weight
                    losses.append(attention_supervision_loss)
                    
                    # 2.2 attention regularization loss
                    if use_attention_regularization_loss:
                        areas = sigmoid_attention_without_mask.mean(dim=-1)
                        labels_without_mask = phrase_classification_labels[:, n:]
                        pos_indices = labels_without_mask == 1
                        neg_indices = labels_without_mask == 0
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
                        losses.append(attention_regularization_loss)

                    # 3. yolov8 loss
                    if using_yolov8 and predict_bboxes_vinbig:
                        batch_size = images.shape[0]
                        assert batch_size == yolov8_features[0].shape[0]
                        vinbig_yolov8_loss, yolov8_loss_items = vinbig_yolov8_criterion(yolov8_features, batch)
                        vinbig_yolov8_loss /= batch_size
                        losses.append(vinbig_yolov8_loss)

                    # 4. global image-phrase contrastive loss
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
                    attention_supervision_loss = compute_balanced_segmentation_loss(sigmoid_attention_with_mask, phrase_grounding_masks,
                                                                                    foreground_loss_weight, background_loss_weight)


        # Prepare output
        output = {}

        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        if training:
            output['phrase_classifier_loss'] = phrase_classifier_loss.detach()
            if use_attention_regularization_loss:
                output['attention_regularization_loss'] = attention_regularization_loss.detach()
            if using_yolov8:
                output[MetricNames.YOLOV8_LOSS] = vinbig_yolov8_loss.detach()
                output[MetricNames.YOLOV8_BOX_LOSS] = yolov8_loss_items[0]
                output[MetricNames.YOLOV8_CLS_LOSS] = yolov8_loss_items[1]
                output[MetricNames.YOLOV8_DFL_LOSS] = yolov8_loss_items[2]
            if use_global_image_phrase_contrastive_loss:
                output['global_alignment_contrastive_loss'] = global_alignment_contrastive_loss.detach()
        else:
            if using_yolov8:
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
        output['attention_supervision_loss'] = attention_supervision_loss.detach()
        output['pred_mask'] = sigmoid_attention_with_mask.detach()
        output['gt_mask'] = phrase_grounding_masks.detach()
        output['pred_probs'] = phrase_classifier_logits.detach().sigmoid()
        output['gt_labels'] = phrase_classification_labels.detach()

        return output
    
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
    
    flag_to_step_fn = {
        'mimfg': step_fn__mimiccxr_fact_grounding, # mimiccxr fact grounding (facts extracted from radiology reports)
        'iufg': step_fn__iuxray_fact_grounding, # iuxray fact grounding (facts extracted from radiology reports)
        'pg': step_fn__phrase_grounding, # phrase grounding (this assumes ground truth masks are available)
        'cibg': step_fn__chest_imagenome_bbox_grounding, # chest imagenome bbox grounding
        'vbg': step_fn__vinbig_bbox_grounding, # vinbig bbox grounding
        'cl': step_fn__chexlocalize, # chexlocalize
        'chxp': step_fn__chexpert_phrase_grounding, # chexpert phrase grounding
        'cxrlt2024c': step_fn__cxrlt2024_custom_labels, # MICCAI CXR-LT 2024 challenge with custom labels
        'cxrlt2024o': step_fn__cxrlt2024_official_labels, # MICCAI CXR-LT 2024 challenge with official labels
    }
    
    def step_fn(unused_engine, batch):
        # print(f'step_fn: flag={batch["flag"]}')
        flag = batch['flag']
        output = flag_to_step_fn[flag](batch)
        output['flag'] = flag # propagate flag
        if update_lr_batchwise: # update learning rate batchwise
            lr_scheduler.step()
        return output
    
    return step_fn

def get_engine(model, device, gradient_accumulation_steps=1,
               use_amp=False, training=False, validating=False,
               testing=False, optimizer=None,
               predict_bboxes_chest_imagenome=False,
               predict_bboxes_vinbig=False,
               update_lr_batchwise=False, lr_scheduler=None,
               using_yolov8=False,
               yolov8_use_multiple_detection_layers=False,
               model_for_yolov8=None,
               pos_area_prior=0.2, neg_area_prior=0.0,
               max_grad_norm=None,
               attention_supervision_loss_weight=1.0,
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
            #    **unused_kwargs,
            ):

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

    print(f'foreground_loss_weight: {foreground_loss_weight}')
    print(f'background_loss_weight: {background_loss_weight}')

    # Create engine
    step_fn = get_step_fn(model, optimizer, training, validating, testing, device,
                            mimiccxr_phrase_classifier_criterion=mimiccxr_phrase_classifier_criterion,
                            binary_multilabel_classification_criterion=binary_multilabel_classification_criterion,
                            generic_phrase_classifier_criterion=generic_phrase_classifier_criterion,
                            contrastive_phrase_grounding_criterion=contrastive_phrase_grounding_criterion,
                            global_image_phrase_contrastive_criterion=global_image_phrase_contrastive_criterion,
                            threshold_criterion=threshold_criterion,
                            neg_area_prior=neg_area_prior,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            max_grad_norm=max_grad_norm, use_amp=use_amp,
                            # chest imagenome dataset
                            predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
                            # vinbig dataset
                            predict_bboxes_vinbig=predict_bboxes_vinbig,
                            # yolov8
                            using_yolov8=using_yolov8,
                            yolov8_criterion=yolov8_criterion,
                            yolov8_use_multiple_detection_layers=yolov8_use_multiple_detection_layers,
                            # batchwise learning rate updates
                            update_lr_batchwise=update_lr_batchwise,
                            lr_scheduler=lr_scheduler,
                            # loss weights
                            attention_supervision_loss_weight=attention_supervision_loss_weight,
                            phrase_classifier_loss_weight=phrase_classifier_loss_weight,
                            foreground_loss_weight=foreground_loss_weight,
                            background_loss_weight=background_loss_weight,
                            # other loss args
                            use_weighted_phrase_classifier_loss=use_weighted_phrase_classifier_loss,
                            use_attention_regularization_loss=use_attention_regularization_loss,
                            use_contrastive_phrase_grounding_loss=use_contrastive_phrase_grounding_loss,
                            use_global_image_phrase_contrastive_loss=use_global_image_phrase_contrastive_loss,
                        )
    engine = Engine(step_fn)
    return engine