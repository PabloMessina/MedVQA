import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.losses import get_binary_multilabel_loss
from medvqa.utils.constants import (
    CXR14_DATASET_ID,
    IUXRAY_DATASET_ID,
    MIMICCXR_DATASET_ID,
    CHEXPERT_DATASET_ID,
    VINBIG_DATASET_ID,
)

def get_step_fn(model, optimizer, nlg_criterion, tokenizer, training, device,
        iters_to_accumulate=1, # for gradient accumulation
        # automatic mixed precision
        use_amp=False,
        # orientation aux task
        classify_orientation=False,
        iuxray_orientation_criterion=None,
        mimiccxr_orientation_criterion=None,
        # chexpert aux task
        classify_chexpert=False,
        chexpert_criterion=None,
        # question auxiliary task
        classify_questions=False,
        question_criterion=None,
        # chexpert dataset
        chexpert_aux_criterion=None,
        # cxr14 dataset
        cxr14_criterion=None,
        # vinbig dataset
        vinbig_criterion=None,
        # batchwise learning rate updatse
        update_lr_batchwise=False,
        lr_scheduler=None,
    ):

    scaler = GradScaler(enabled=use_amp)

    if update_lr_batchwise:
        assert lr_scheduler is not None

    if training:
        iters_count = 0
        def backward_and_optimizer_step(batch_loss):
            nonlocal iters_count
            assert batch_loss is not None
            batch_loss = batch_loss / iters_to_accumulate
            scaler.scale(batch_loss).backward()
            if (iters_count + 1) % iters_to_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                # batch_loss.backward()
                # optimizer.step()
                optimizer.zero_grad()                
            iters_count += 1
    
    def step_fn__mimiccxr_iuxray(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        images = batch['i'].to(device)
        texts = batch['t'].to(device)
        text_lengths = batch['tl']

        is_mimiccxr = (dataset_id == MIMICCXR_DATASET_ID)

        if classify_orientation:
            orientation = batch['orientation'].to(device)
        if classify_chexpert:
            chexpert = batch['chexpert'].to(device)
        if classify_questions:
            question_labels = batch['qlabels'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                'texts': texts,
                'text_lengths': text_lengths,
                'mimiccxr_forward': is_mimiccxr,
                'iuxray_forward': not is_mimiccxr,
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                
                model_output = model(**model_kwargs)
                pred_text_logits = model_output['pred_texts']
                if classify_orientation:
                    if is_mimiccxr:
                        pred_orientation_logits = model_output['mimiccxr_pred_orientation']
                    else:
                        pred_orientation_logits = model_output['iuxray_pred_orientation']
                if classify_chexpert:
                    pred_chexpert_logits = model_output[f'pred_chexpert']
                    pred_chexpert_probs = model_output[f'pred_chexpert_probs']
                if classify_questions:
                    pred_qlabels_logits = model_output['pred_qlabels']
                    pred_qlabels_probs = model_output[f'pred_qlabels_probs']

                if training:                    
                    # Compute losses
                    losses = []
                    text_loss = nlg_criterion(pred_text_logits.view(-1, pred_text_logits.shape[-1]), texts.view(-1))
                    losses.append(text_loss)                    
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
                    if len(losses) > 0:
                        batch_loss = sum(losses)
                    else:
                        batch_loss = None
                    # Backward pass + optimizer step if training
                    backward_and_optimizer_step(batch_loss)

        # Recover text
        pred_texts = pred_text_logits.argmax(-1)
        output = {
            'idxs': idxs,
            'backgrounds': tokenizer.clean_batch(texts.detach()),
            'pred_backgrounds': tokenizer.clean_batch(pred_texts.detach()),
            'dataset_id': dataset_id,
        }
        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
            output['background_loss'] = text_loss.detach()
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
            output['pred_qlabels_probs'] = pred_qlabels_probs.detach().cpu()
            if training:
                output['qlabels_loss'] = qlabels_loss.detach()

        return output
    
    def step_fn__chexpert_cxr14(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        images = batch['i'].to(device)
        labels = batch['l'].to(device)
        genders = batch['g'].to(device)
        orientations = batch['o'].to(device)
        dataset_name = 'chexpert' if dataset_id == CHEXPERT_DATASET_ID else 'cxr14'
        findings_criterion = chexpert_criterion if dataset_id == CHEXPERT_DATASET_ID else cxr14_criterion
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'raw_images': images,
                f'{dataset_name}_forward': True,
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision

                model_output = model(**model_kwargs)
                
                pred_labels_logits = model_output[f'pred_{dataset_name}']
                pred_labels_probs = model_output[f'pred_{dataset_name}_probs']
                pred_orientation_logits = model_output['pred_orientation']
                pred_gender_logits = model_output['pred_gender']

                if training:                    
                    # Compute losses                    
                    labels_loss = findings_criterion(pred_labels_logits, labels.float())
                    orientation_loss = chexpert_aux_criterion(pred_orientation_logits, orientations)
                    gender_loss = chexpert_aux_criterion(pred_gender_logits, genders)                    
                    batch_loss = labels_loss + orientation_loss + gender_loss
                    # Backward pass + optimizer step if training
                    backward_and_optimizer_step(batch_loss)
        
        output = {
            'idxs': idxs,
            'dataset_id': dataset_id,
        }
        if training:
            output['loss'] = batch_loss.detach()
            
        # dataset-specific labels
        output[dataset_name] = labels.detach().cpu()
        output[f'pred_{dataset_name}'] = (pred_labels_logits.detach() > 0).cpu()
        output[f'pred_{dataset_name}_probs'] = pred_labels_probs.detach().cpu()
        if training:
            output[f'{dataset_name}_loss'] = labels_loss.detach()

        # orientation
        output['orientation'] = orientations.detach()
        output['pred_orientation'] = pred_orientation_logits.argmax(-1).detach()        
        if training:
            output['orientation_loss'] = orientation_loss.detach()

        # gender
        output['gender'] = genders.detach()
        output['pred_gender'] = pred_gender_logits.argmax(-1).detach()        
        if training:
            output['gender_loss'] = gender_loss.detach()

        return output

    def step_fn__vinbig(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        vinbig_labels = batch['l'].to(device)
        images = batch['i'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'vinbig_forward': True,
                'raw_images': images,
            }
            
            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)                
                pred_vinbig_logits = model_output[f'pred_vinbig']
                pred_vinbig_probs = model_output[f'pred_vinbig_probs']
                if training:
                    # Compute losses
                    vinbig_loss = vinbig_criterion(pred_vinbig_logits, vinbig_labels.float())                    
                    batch_loss = vinbig_loss
                    # Backward pass + optimizer step if training
                    backward_and_optimizer_step(batch_loss)
        
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
            output[f'vinbig_loss'] = vinbig_loss.detach()

        return output

    _mim_iu_datasets = [MIMICCXR_DATASET_ID, IUXRAY_DATASET_ID]
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
        else: assert False, f'Unknown dataset_id {dataset_id}'
        # update learning rate batchwise
        if update_lr_batchwise:
            lr_scheduler.step()
        return output
    
    return step_fn

def get_engine(model, tokenizer, classify_orientation, classify_chexpert, classify_questions, device,
               iters_to_accumulate=1,
               binary_loss_name='bce',
               use_amp=False,
               training=False,
               use_chexpert=False,
               use_cxr14=False,
               use_vinbig=False,
               optimizer=None,
               update_lr_batchwise=False, lr_scheduler=None):
    
    # Criterion
    nlg_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding in loss
    
    # Auxiliary tasks

    if training and classify_orientation:
        iuxray_orientation_criterion = nn.CrossEntropyLoss()
        mimiccxr_orientation_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore unknown
    else:
        iuxray_orientation_criterion = None
        mimiccxr_orientation_criterion = None
    
    if training and classify_questions:
        question_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        question_criterion = None

    if training and use_chexpert or use_cxr14 or use_cxr14:
        chexpert_aux_criterion = nn.CrossEntropyLoss()
    else:
        chexpert_aux_criterion = None

    if training and classify_chexpert or use_chexpert:
        chexpert_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        chexpert_criterion = None

    if training and use_cxr14:
        cxr14_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        cxr14_criterion = None
    
    if training and use_vinbig:
        vinbig_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        vinbig_criterion = None

    # Create engine
    step_fn = get_step_fn(model, optimizer, nlg_criterion, tokenizer,
                          training=training,
                          device=device, use_amp=use_amp,
                          iters_to_accumulate=iters_to_accumulate,
                          # orientation auxiliary task
                          classify_orientation=classify_orientation,
                          iuxray_orientation_criterion=iuxray_orientation_criterion,
                          mimiccxr_orientation_criterion=mimiccxr_orientation_criterion,
                          # chexpert auxiliary task
                          classify_chexpert=classify_chexpert,
                          chexpert_criterion=chexpert_criterion,
                          # question auxiliary task
                          classify_questions=classify_questions,
                          question_criterion=question_criterion,
                          # chexpert dataset
                          chexpert_aux_criterion=chexpert_aux_criterion,
                          # cxr14 dataset
                          cxr14_criterion=cxr14_criterion,
                          # vinbig dataset
                          vinbig_criterion=vinbig_criterion,
                          # batchwise learning rate updates
                          update_lr_batchwise=update_lr_batchwise,
                          lr_scheduler=lr_scheduler,
                          )
    engine = Engine(step_fn)
    return engine