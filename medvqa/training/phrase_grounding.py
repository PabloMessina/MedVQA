import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.losses.nt_xent_loss import NTXentLoss
from medvqa.losses.optimizers import GradientAccumulator
from medvqa.losses.segmentation_loss import compute_balanced_segmentation_loss
from medvqa.utils.constants import MetricNames
from medvqa.utils.logging import print_magenta

def get_step_fn(model, optimizer, training, validating, testing, device,
                phrase_classifier_criterion,
                contrastive_phrase_grounding_criterion,
                pos_area_prior, neg_area_prior,
                iters_to_accumulate=1, # for gradient accumulation
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
        gradient_accumulator = GradientAccumulator(optimizer, scaler, iters_to_accumulate, max_grad_norm)
    
    def step_fn__fact_grounding(batch):

        assert training

        # Extract elements from batch
        if using_yolov8:
            images = batch['img'].to(device)
        else:
            images = batch['i'].to(device)
        pos_phrase_embeddings = batch['pe'].to(device)
        neg_phrase_embeddings = batch['ne'].to(device)

        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'pos_phrase_embeddings': pos_phrase_embeddings,
                'neg_phrase_embeddings': neg_phrase_embeddings,
                'only_compute_features': True,
                'mimiccxr_forward': True,
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                sigmoid_attention_pos = model_output['sigmoid_attention_pos']
                sigmoid_attention_neg = model_output['sigmoid_attention_neg']
                phrase_classifier_output_pos = model_output['phrase_classifier_output_pos']
                phrase_classifier_output_neg = model_output['phrase_classifier_output_neg']
                phrase_grounding_similarity_pos = model_output['phrase_grounding_similarity_pos'] # (batch_size, max_phrase_length)
                phrase_grounding_similarity_neg = model_output['phrase_grounding_similarity_neg']
                
                if training:
                    # Compute losses
                    losses = []

                    # 1. phrase classification loss
                    phrase_classifier_loss_pos = phrase_classifier_criterion(phrase_classifier_output_pos, torch.ones_like(phrase_classifier_output_pos))
                    phrase_classifier_loss_neg = phrase_classifier_criterion(phrase_classifier_output_neg, torch.zeros_like(phrase_classifier_output_neg))
                    phrase_classifier_loss = phrase_classifier_loss_pos + phrase_classifier_loss_neg
                    phrase_classifier_loss *= phrase_classifier_loss_weight # weight
                    losses.append(phrase_classifier_loss)

                    # 2. contrastive phrase grounding loss
                    contrastive_phrase_grounding_loss = contrastive_phrase_grounding_criterion(phrase_grounding_similarity_pos,
                                                                                               phrase_grounding_similarity_neg)
                    losses.append(contrastive_phrase_grounding_loss)

                    # 3. attention regularization loss
                    area_pos = sigmoid_attention_pos.mean(dim=-1)
                    area_neg = sigmoid_attention_neg.mean(dim=-1)
                    attention_regularization_loss_pos = (area_pos - pos_area_prior).abs().mean()
                    attention_regularization_loss_neg = (area_neg - neg_area_prior).abs().mean()
                    attention_regularization_loss = attention_regularization_loss_pos + attention_regularization_loss_neg
                    losses.append(attention_regularization_loss)

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
            output['contrastive_phrase_grounding_loss'] = contrastive_phrase_grounding_loss.detach()
            output['attention_regularization_loss'] = attention_regularization_loss.detach()

        return output
    
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
        
        assert using_yolov8

        # Extract elements from batch
        if using_yolov8:
            images = batch['img'].to(device)
        else:
            images = batch['i'].to(device)
        phrase_embeddings = batch['pe'].to(device)
        phrase_grounding_masks = batch['pgm'].to(device)
        if not training:
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
                    if predict_bboxes_chest_imagenome:
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
            output['chest_imagenome_bbox_coords'] = chest_imagenome_bbox_coords
            output['chest_imagenome_bbox_presence'] = chest_imagenome_bbox_presence
        output['attention_supervision_loss'] = attention_supervision_loss.detach()
        output['pred_mask'] = sigmoid_attention.detach()
        output['gt_mask'] = phrase_grounding_masks.detach()

        return output
    
    def step_fn__vinbig_bbox_grounding(batch):
        
        assert using_yolov8

        # Extract elements from batch
        if using_yolov8:
            images = batch['img'].to(device)
        else:
            images = batch['i'].to(device)
        phrase_embeddings = batch['pe'].to(device)
        phrase_classification_labels = batch['pcl'].to(device)
        phrase_grounding_masks = batch['pgm'].to(device)
        if not training:
            vinbig_bbox_coords = batch['bboxes']
            vinbig_bbox_classes = batch['classes']
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'phrase_embeddings': phrase_embeddings,
                'vinbig_forward': True,
            }
            if using_yolov8 and yolov8_use_multiple_detection_layers:
                model_kwargs['yolov8_detection_layer_index'] = vinbig_yolov8_index

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                sigmoid_attention = model_output['sigmoid_attention']
                phrase_classifier_output = model_output['phrase_classifier_output']
                yolov8_features = model_output['yolov8_features']
                if not training:
                    yolov8_predictions = model_output['yolov8_predictions']
                
                if training:
                    # Compute losses
                    losses = []
                    
                    # 1. phrase classification loss
                    phrase_classifier_loss = phrase_classifier_criterion(phrase_classifier_output.view(-1), phrase_classification_labels.view(-1))
                    phrase_classifier_loss *= phrase_classifier_loss_weight # weight
                    losses.append(phrase_classifier_loss)

                    # 2. attention supervision loss
                    attention_supervision_loss = compute_balanced_segmentation_loss(sigmoid_attention, phrase_grounding_masks,
                                                                                    foreground_loss_weight, background_loss_weight)
                    attention_supervision_loss *= attention_supervision_loss_weight # weight
                    losses.append(attention_supervision_loss)

                    # 3. yolov8 loss
                    if predict_bboxes_vinbig:
                        batch_size = images.shape[0]
                        assert batch_size == yolov8_features[0].shape[0]
                        vinbig_yolov8_loss, yolov8_loss_items = vinbig_yolov8_criterion(yolov8_features, batch)
                        vinbig_yolov8_loss /= batch_size
                        losses.append(vinbig_yolov8_loss)

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
        output['attention_supervision_loss'] = attention_supervision_loss.detach()
        output['pred_mask'] = sigmoid_attention.detach()
        output['gt_mask'] = phrase_grounding_masks.detach()
        output['pred_phrase_labels'] = (phrase_classifier_output.detach() > 0).view(-1)
        output['gt_phrase_labels'] = phrase_classification_labels.detach().view(-1)

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
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                sigmoid_attention = model_output['sigmoid_attention']
                phrase_classifier_output = model_output['phrase_classifier_output']
                
                if training:
                    # Compute losses
                    losses = []
                    
                    # 1. phrase classification loss
                    phrase_classifier_loss = phrase_classifier_criterion(phrase_classifier_output.view(-1), phrase_classification_labels.view(-1))
                    phrase_classifier_loss *= phrase_classifier_loss_weight # weight
                    losses.append(phrase_classifier_loss)

                    # 2. attention supervision loss
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
        if training:
            output['phrase_classifier_loss'] = phrase_classifier_loss.detach()
        output['attention_supervision_loss'] = attention_supervision_loss.detach()
        output['pred_mask'] = sigmoid_attention.detach()
        output['gt_mask'] = phrase_grounding_masks.detach()
        output['pred_phrase_labels'] = (phrase_classifier_output.detach() > 0).view(-1)
        output['gt_phrase_labels'] = phrase_classification_labels.detach().view(-1)

        return output
    
    def step_fn(unused_engine, batch):
        flag = batch['flag']
        print(f'step_fn(): flag={flag}')
        if flag == 'fg': # fact grounding (facts extracted from radiology reports)
            output = step_fn__fact_grounding(batch)
        elif flag == 'pg': # phrase grounding (this assumes ground truth masks are available)
            output = step_fn__phrase_grounding(batch)
        elif flag == 'cibg': # chest imagenome bbox grounding
            output = step_fn__chest_imagenome_bbox_grounding(batch)
        elif flag == 'vbg': # vinbig bbox grounding
            output = step_fn__vinbig_bbox_grounding(batch)
        elif flag == 'cl': # chexlocalize
            output = step_fn__chexlocalize(batch)
        else:
            raise ValueError(f'Invalid flag: {flag}')
        output['flag'] = flag # propagate flag
        # update learning rate batchwise
        if update_lr_batchwise:
            lr_scheduler.step()
        return output
    
    return step_fn

def get_engine(model, device, iters_to_accumulate=1,
               use_amp=False, training=False, validating=False,
               testing=False, optimizer=None,
               predict_bboxes_chest_imagenome=False,
               predict_bboxes_vinbig=False,
               update_lr_batchwise=False, lr_scheduler=None,
               using_yolov8=False,
               yolov8_use_multiple_detection_layers=False,
               model_for_yolov8=None,
               pos_area_prior=0.4, neg_area_prior=0.0,
               max_grad_norm=None,
               attention_supervision_loss_weight=1.0,
               phrase_classifier_loss_weight=1.0,
               foreground_loss_weight=1.0,
               background_loss_weight=1.0,
            #    **unused_kwargs,
            ):

    # Create criterion
    if training:
        phrase_classifier_criterion = nn.BCEWithLogitsLoss()
        contrastive_phrase_grounding_criterion = NTXentLoss(temperature=0.1, device=device)
    else:
        phrase_classifier_criterion = contrastive_phrase_grounding_criterion = None

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
                            phrase_classifier_criterion=phrase_classifier_criterion,
                            contrastive_phrase_grounding_criterion=contrastive_phrase_grounding_criterion,
                            pos_area_prior=pos_area_prior, neg_area_prior=neg_area_prior,
                            iters_to_accumulate=iters_to_accumulate,
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
                        )
    engine = Engine(step_fn)
    return engine