import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.utils.constants import MIMICCXR_DATASET_ID

def get_step_fn(model, optimizer, training, device,
            # automatic mixed precision
            use_amp=False,
            # tags aux task
            classify_tags=False,
            tags_criterion=None,
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
    ):

    scaler = GradScaler(enabled=use_amp)
    
    def step_fn(unused_engine, batch):

        # Extract elements from batch
        idxs = batch['idx']
        images = batch['i'].to(device)
        
        if classify_tags:
            tags = batch['tags'].to(device)
        if classify_orientation:
            dataset_id = batch['dataset_id']
            orientation = batch['orientation'].to(device)
        if classify_chexpert:
            chexpert = batch['chexpert'].to(device)
        if classify_questions:
            question_labels = batch['qlabels'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'images': images,
            }
            if classify_orientation:
                if dataset_id == MIMICCXR_DATASET_ID:
                    model_kwargs['mimiccxr_foward'] = True
                else:
                    model_kwargs['iuxray_foward'] = True

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision

                model_output = model(**model_kwargs)

                if classify_tags:
                    pred_tags_logits = model_output['pred_tags']            
                if classify_orientation:
                    if dataset_id == MIMICCXR_DATASET_ID:
                        pred_orientation_logits = model_output['mimiccxr_pred_orientation']
                    else:
                        pred_orientation_logits = model_output['iuxray_pred_orientation']
                if classify_chexpert:
                    pred_chexpert_logits = model_output['pred_chexpert']
                    pred_chexpert_probs = model_output['pred_chexpert_probs']
                if classify_questions:
                    pred_qlabels_logits = model_output['pred_qlabels']

                if training:                    
                    # Compute losses
                    losses = []
                    if classify_tags:
                        tags_loss = tags_criterion(pred_tags_logits, tags.float())
                        losses.append(tags_loss)
                    if classify_orientation:
                        if dataset_id == MIMICCXR_DATASET_ID:
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
                    assert batch_loss is not None
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # batch_loss.backward()
                    # optimizer.step()
                    optimizer.zero_grad()
        
        output = {
            'idxs': idxs,
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
            output['dataset_id'] = dataset_id
            if training:
                output['orientation_loss'] = orientation_loss.detach()
        if classify_chexpert:
            output['chexpert'] = chexpert.detach().cpu()
            output['pred_chexpert'] = (pred_chexpert_logits.detach() > 0).cpu()
            output['pred_chexpert_probs'] = pred_chexpert_probs.detach().cpu()
            if training:
                output['chexpert_loss'] = chexpert_loss.detach()
        if classify_questions:
            output['qlabels'] = question_labels.detach().cpu()
            output['pred_qlabels'] = (pred_qlabels_logits.detach() > 0).cpu()
            if training:
                output['qlabels_loss'] = qlabels_loss.detach()

        return output
    
    return step_fn

def get_engine(model, classify_tags, classify_orientation, classify_chexpert, classify_questions,
               device, use_amp=False, training=False, optimizer=None):    
    
    # Auxiliary tasks
    if classify_tags:
        tags_criterion = nn.BCEWithLogitsLoss()
    else:
        tags_criterion = None
    
    if classify_orientation:
        iuxray_orientation_criterion = nn.CrossEntropyLoss()
        mimiccxr_orientation_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore unknown
    else:
        iuxray_orientation_criterion = None
        mimiccxr_orientation_criterion = None

    if classify_chexpert:
        chexpert_criterion = nn.BCEWithLogitsLoss()
    else:
        chexpert_criterion = None
    
    if classify_questions:
        question_criterion = nn.BCEWithLogitsLoss()
    else:
        question_criterion = None

    # Create engine
    step_fn = get_step_fn(model, optimizer,
                          training=training,
                          device=device, use_amp=use_amp,
                          # tags auxiliary task
                          classify_tags=classify_tags,
                          tags_criterion=tags_criterion,
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
                          )
    engine = Engine(step_fn)
    return engine