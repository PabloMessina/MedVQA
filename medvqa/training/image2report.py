import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.losses import get_binary_multilabel_loss
from medvqa.losses.optimizers import GradientAccumulator
from medvqa.utils.constants import MIMICCXR_DATASET_ID, MetricNames

def get_step_fn(model, optimizer, tokenizer, training, device,
        shift_tokens_for_transformer=True,
        include_report=True,
        max_report_length=None,
        iters_to_accumulate=1, # for gradient accumulation
        nlg_criterion=None, # for report generation
        use_visual_module_only=False,
        # automatic mixed precision
        use_amp=False,
        # gender aux task
        classify_gender=False,
        chest_imagenome_gender_criterion=None,
        # chexpert aux task
        classify_chexpert=False,
        chexpert_criterion=None,
        # chest imagenome dataset
        classify_chest_imagenome=False,
        chest_imagenome_multilabel_criterion=None,
        predict_bboxes_chest_imagenome=False,
        chest_imagenome_bbox_coords_criterion=None,
        chest_imagenome_bbox_presence_criterion=None,
        chest_imagenome_bbox_loss_weight=1.0,
        valid_chest_imagenome_label_indices=None,
        # yolov8
        using_yolov8=False,
        yolov8_criterion=None,
        # batchwise learning rate updates
        update_lr_batchwise=False,
        lr_scheduler=None,
    ):

    scaler = GradScaler(enabled=use_amp)

    assert tokenizer is not None, 'tokenizer is required'

    if training:
        gradient_accumulator = GradientAccumulator(optimizer, scaler, iters_to_accumulate)
        if not use_visual_module_only:
            assert include_report, 'Training requires report'
            assert nlg_criterion is not None, 'Training requires report loss'
    
    if classify_chexpert:
        if training:
            assert chexpert_criterion is not None, 'Training requires chexpert loss'

    if classify_chest_imagenome:
        if training:
            assert chest_imagenome_multilabel_criterion is not None, 'Training requires chest imagenome loss'

    if update_lr_batchwise:
        assert lr_scheduler is not None

    if using_yolov8 and training:
        assert yolov8_criterion is not None

    if predict_bboxes_chest_imagenome:
        print('chest_imagenome_bbox_loss_weight: ', chest_imagenome_bbox_loss_weight)
    
    def step_fn__mimiccxr(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        if using_yolov8:
            images = batch['img'].to(device)
        else:
            images = batch['i'].to(device)
        if not use_visual_module_only:
            if include_report:
                reports = batch['report'].to(device)
                if shift_tokens_for_transformer:
                    reports_start = reports[:, :-1]
                    reports_end = reports[:, 1:]
        if classify_chexpert:
            chexpert = batch['chexpert'].to(device)
        if classify_chest_imagenome:
            chest_imagenome = batch['chest_imagenome'].to(device)
        if classify_gender:
            genders = batch['gender'].to(device)
        if predict_bboxes_chest_imagenome:
            if (using_yolov8 and not training) or not using_yolov8:
                chest_imagenome_bbox_coords = batch['chest_imagenome_bbox_coords'].to(device)
                chest_imagenome_bbox_presence = batch['chest_imagenome_bbox_presence'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {                
                'device': device,
                'mode': 'train' if training else 'eval',
            }            
            model_kwargs['raw_images'] = images
            model_kwargs['mimiccxr_forward'] = True

            if not use_visual_module_only:
                if training:
                    if shift_tokens_for_transformer:
                        model_kwargs['reports'] = reports_start
                    else:
                        model_kwargs['reports'] = reports
                else:
                    if include_report:
                        model_kwargs['max_report_length'] = reports.size(1)
                    else:                    
                        model_kwargs['max_report_length'] = max_report_length

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                
                if classify_gender:
                    pred_gender_logits = model_output['pred_gender']
                if classify_chexpert:
                    pred_chexpert_logits = model_output[f'pred_chexpert']
                    pred_chexpert_probs = model_output[f'pred_chexpert_probs']
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
                        pred_chest_imagenome_bbox_presence = model_output['pred_chest_imagenome_bbox_presence']

                if not use_visual_module_only:
                    if training:
                        pred_report_logits = model_output['pred_reports']
                        pred_reports = pred_report_logits.argmax(dim=-1)
                    else:
                        pred_reports = model_output['pred_reports']

                if training:
                    # Compute losses
                    losses = []
                    if classify_gender:
                        gender_loss = chest_imagenome_gender_criterion(pred_gender_logits, genders)
                        losses.append(gender_loss)
                    if classify_chexpert:
                        chexpert_loss = chexpert_criterion(pred_chexpert_logits, chexpert.float())
                        losses.append(chexpert_loss)
                    if classify_chest_imagenome:
                        chest_imagenome_loss = chest_imagenome_multilabel_criterion(pred_chest_imagenome_logits, chest_imagenome.float())
                        losses.append(chest_imagenome_loss)
                    if predict_bboxes_chest_imagenome:
                        if using_yolov8:
                            batch_size = images.shape[0]
                            assert batch_size == yolov8_features[0].shape[0]
                            chest_imagenome_yolov8_loss, yolov8_loss_items = yolov8_criterion(yolov8_features, batch)
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
                        if include_report:
                            if shift_tokens_for_transformer:
                                report_loss = nlg_criterion(pred_report_logits.reshape(-1, pred_report_logits.shape[-1]), reports_end.reshape(-1))
                            else:
                                report_loss = nlg_criterion(pred_report_logits.reshape(-1, pred_report_logits.shape[-1]), reports.view(-1))
                        losses.append(report_loss)

                    assert len(losses) > 0
                    batch_loss = sum(losses)

                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss)

        # Prepare output
        output = {
            'idxs': idxs,            
            'dataset_id': dataset_id,
        }

        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        if classify_gender:
            output['gender'] = genders.detach()
            output['pred_gender'] = pred_gender_logits.argmax(-1).detach()
            if training:
                output['gender_loss'] = gender_loss.detach()
        if classify_chexpert:
            output['chexpert'] = chexpert.detach().cpu()
            output[f'pred_chexpert'] = (pred_chexpert_logits.detach() > 0).cpu()
            output[f'pred_chexpert_probs'] = pred_chexpert_probs.detach().cpu()
            if training:
                output[f'chexpert_loss'] = chexpert_loss.detach()
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
            output['pred_reports'] = tokenizer.clean_batch(pred_reports.detach())
            if include_report:
                output['reports'] = tokenizer.clean_batch(reports.detach())
                if training:
                    output['report_loss'] = report_loss.detach()
        return output
    
    def step_fn(unused_engine, batch):
        dataset_id = batch['dataset_id']
        if dataset_id == MIMICCXR_DATASET_ID:
            output = step_fn__mimiccxr(batch)
        else: assert False, f'Unknown dataset_id {dataset_id}'
        # update learning rate batchwise
        if update_lr_batchwise:
            lr_scheduler.step()
        return output
    
    return step_fn

def get_engine(model, classify_gender, classify_chexpert, classify_chest_imagenome,
                predict_bboxes_chest_imagenome, device,
                include_report=True,
                tokenizer=None,
                shift_tokens_for_transformer=True,
                max_report_length=None,
                iters_to_accumulate=1,
                binary_loss_name='bce',
                focal_loss_weight=None,
                bce_loss_weight=None,
                wbce_loss_weight=None,
                use_amp=False,
                training=False,
                chest_imagenome_bbox_loss_weight=1.0,
                valid_chest_imagenome_label_indices=None,
                optimizer=None,
                update_lr_batchwise=False, lr_scheduler=None,
                use_visual_module_only=False,
                using_yolov8=False,
                model_for_yolov8=None,
            ):
    
    # Criterion
    if not use_visual_module_only and include_report:
        nlg_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding in loss
    else:
        nlg_criterion = None
    
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
    
    if training and classify_gender:
        chest_imagenome_gender_criterion = nn.CrossEntropyLoss(ignore_index=2) # ignore unknown
    else:
        chest_imagenome_gender_criterion = None

    if training and classify_chexpert:
        chexpert_criterion = get_binary_multilabel_loss(binary_loss_name, **binary_loss_kwargs)
    else:
        chexpert_criterion = None

    if training and classify_chest_imagenome:
        chest_imagenome_multilabel_criterion = get_binary_multilabel_loss(binary_loss_name, **binary_loss_kwargs)
    else:
        chest_imagenome_multilabel_criterion = None

    if training and predict_bboxes_chest_imagenome and not using_yolov8:
        chest_imagenome_bbox_coords_criterion = nn.MSELoss()
        chest_imagenome_bbox_presence_criterion = nn.BCEWithLogitsLoss()
    else:
        chest_imagenome_bbox_coords_criterion = None
        chest_imagenome_bbox_presence_criterion = None

    if training and using_yolov8:
        assert model_for_yolov8 is not None
        from ultralytics.yolo.v8.detect.train import Loss
        from ultralytics.yolo.utils.torch_utils import de_parallel
        yolov8_criterion = Loss(de_parallel(model_for_yolov8))
    else:
        yolov8_criterion = None

    # Create engine
    step_fn = get_step_fn(model=model, optimizer=optimizer, tokenizer=tokenizer, training=training, device=device,
                        shift_tokens_for_transformer=shift_tokens_for_transformer,
                        include_report=include_report,
                        max_report_length=max_report_length,
                        iters_to_accumulate=iters_to_accumulate,
                        nlg_criterion=nlg_criterion,
                        use_visual_module_only=use_visual_module_only,
                        use_amp=use_amp,
                        # gender auxiliary task
                        classify_gender=classify_gender,
                        chest_imagenome_gender_criterion=chest_imagenome_gender_criterion,
                        # chexpert auxiliary task
                        classify_chexpert=classify_chexpert,
                        chexpert_criterion=chexpert_criterion,
                        # chest imagenome dataset
                        classify_chest_imagenome=classify_chest_imagenome,
                        chest_imagenome_multilabel_criterion=chest_imagenome_multilabel_criterion,
                        predict_bboxes_chest_imagenome=predict_bboxes_chest_imagenome,
                        chest_imagenome_bbox_coords_criterion=chest_imagenome_bbox_coords_criterion,
                        chest_imagenome_bbox_presence_criterion=chest_imagenome_bbox_presence_criterion,
                        chest_imagenome_bbox_loss_weight=chest_imagenome_bbox_loss_weight,
                        valid_chest_imagenome_label_indices=valid_chest_imagenome_label_indices,
                        # yolov8
                        using_yolov8=using_yolov8,
                        yolov8_criterion=yolov8_criterion,
                        # batchwise learning rate updates
                        update_lr_batchwise=update_lr_batchwise,
                        lr_scheduler=lr_scheduler,
                        )
    engine = Engine(step_fn)
    return engine