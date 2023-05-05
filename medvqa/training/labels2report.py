import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.losses import get_binary_multilabel_loss
from medvqa.losses.optimizers import GradientAccumulator
from medvqa.utils.constants import (
    MIMICCXR_DATASET_ID,
    MetricNames,
)

def get_step_fn(model, optimizer, tokenizer, training, device,
        shift_tokens_for_transformer=True,
        include_report=True,
        max_report_length=None,
        iters_to_accumulate=1, # for gradient accumulation
        nlg_criterion=None, # for report generation
        # automatic mixed precision
        use_amp=False,
        # chexpert aux task
        classify_chexpert=False,
        chexpert_criterion=None,
        chexpert_range=None,
        # chest imagenome dataset
        classify_chest_imagenome=False,
        chest_imagenome_criterion=None,
        chest_imagenome_range=None,
        # batchwise learning rate updates
        update_lr_batchwise=False,
        lr_scheduler=None,
    ):

    scaler = GradScaler(enabled=use_amp)
    
    assert tokenizer is not None, 'tokenizer is required'

    if training:
        gradient_accumulator = GradientAccumulator(optimizer, scaler, iters_to_accumulate)
        assert include_report, 'Training requires report'
        assert nlg_criterion is not None, 'Training requires report loss'
    
    if classify_chexpert:
        if training:
            assert chexpert_criterion is not None, 'Training requires chexpert loss'
        assert chexpert_range is not None, 'chexpert range is required'

    if classify_chest_imagenome:
        if training:
            assert chest_imagenome_criterion is not None, 'Training requires chest imagenome loss'
        assert chest_imagenome_range is not None, 'chest imagenome range is required'
    
    def step_fn__mimiccxr(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        predicted_binary_scores = batch['predicted_binary_scores'].to(device)
        if include_report:
            reports = batch['report'].to(device)
            if shift_tokens_for_transformer:
                reports_start = reports[:, :-1]
                reports_end = reports[:, 1:]
        if classify_chexpert:
            chexpert = batch['chexpert'].to(device)
        if classify_chest_imagenome:
            chest_imagenome = batch['chest_imagenome'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {                
                'device': device,
                'mode': 'train' if training else 'eval',
                'predicted_binary_scores': predicted_binary_scores,
            }
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
                if classify_chexpert or classify_chest_imagenome:
                    pred_label_logits = model_output['pred_label_logits']
                    pred_label_probs = model_output['pred_label_probs']
                if classify_chexpert:
                    pred_chexpert_logits = pred_label_logits[:, chexpert_range[0]:chexpert_range[1]]
                    pred_chexpert_probs = pred_label_probs[:, chexpert_range[0]:chexpert_range[1]]
                if classify_chest_imagenome:
                    pred_chest_imagenome_logits = pred_label_logits[:, chest_imagenome_range[0]:chest_imagenome_range[1]]
                    pred_chest_imagenome_probs = pred_label_probs[:, chest_imagenome_range[0]:chest_imagenome_range[1]]
                if training:
                    pred_report_logits = model_output['pred_reports']
                    pred_reports = pred_report_logits.argmax(dim=-1)
                else:
                    pred_reports = model_output['pred_reports']

                if training:
                    # Compute losses
                    losses = []
                    if classify_chexpert:
                        chexpert_loss = chexpert_criterion(pred_chexpert_logits, chexpert.float())
                        losses.append(chexpert_loss)
                    if classify_chest_imagenome:
                        chest_imagenome_loss = chest_imagenome_criterion(pred_chest_imagenome_logits, chest_imagenome.float())
                        losses.append(chest_imagenome_loss)
                    if include_report:
                        if shift_tokens_for_transformer:
                            report_loss = nlg_criterion(pred_report_logits.reshape(-1, pred_report_logits.shape[-1]), reports_end.reshape(-1))
                        else:
                            report_loss = nlg_criterion(pred_report_logits.reshape(-1, pred_report_logits.shape[-1]), reports.view(-1))
                    losses.append(report_loss)

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

def get_engine(model, tokenizer, classify_chexpert, classify_chest_imagenome, device,
                chexpert_range=None,
                chest_imagenome_range=None,
                include_report=True,
                shift_tokens_for_transformer=True,
                max_report_length=None,
                iters_to_accumulate=1,
                binary_loss_name='bce',
                focal_loss_weight=None,
                bce_loss_weight=None,
                wbce_loss_weight=None,
                use_amp=False,
                training=False,
                optimizer=None,
                update_lr_batchwise=False, lr_scheduler=None,
            ):
    
    # Criterion
    if include_report:
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
    if training and classify_chexpert:
        chexpert_criterion = get_binary_multilabel_loss(binary_loss_name, **binary_loss_kwargs)
    else:
        chexpert_criterion = None

    if training and classify_chest_imagenome:
        chest_imagenome_criterion = get_binary_multilabel_loss(binary_loss_name, **binary_loss_kwargs)
    else:
        chest_imagenome_criterion = None

    # Create engine
    step_fn = get_step_fn(model=model, optimizer=optimizer, tokenizer=tokenizer, training=training, device=device,
                            shift_tokens_for_transformer=shift_tokens_for_transformer,
                            include_report=include_report,
                            max_report_length=max_report_length,
                            iters_to_accumulate=iters_to_accumulate,
                            nlg_criterion=nlg_criterion,
                            use_amp=use_amp,
                            # chexpert auxiliary task
                            classify_chexpert=classify_chexpert,
                            chexpert_criterion=chexpert_criterion,
                            chexpert_range=chexpert_range,
                            # chest imagenome dataset
                            classify_chest_imagenome=classify_chest_imagenome,
                            chest_imagenome_criterion=chest_imagenome_criterion,
                            chest_imagenome_range=chest_imagenome_range,
                            # batchwise learning rate updates
                            update_lr_batchwise=update_lr_batchwise,
                            lr_scheduler=lr_scheduler,
                        )
    engine = Engine(step_fn)
    return engine