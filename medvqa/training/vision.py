import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.losses import get_binary_multilabel_loss
from medvqa.utils.constants import IUXRAY_DATASET_ID, MIMICCXR_DATASET_ID, CHEXPERT_DATASET_ID, MetricNames

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
        # chexpert dataset
        chexpert_aux_criterion=None,
        # chest imagenome dataset
        classify_chest_imagenome=False,
        chest_imagenome_multilabel_criterion=None,        
    ):

    scaler = GradScaler(enabled=use_amp)
    
    def step_fn__mimiccxr_iuxray(batch):

        # Extract elements from batch
        idxs = batch['idx']
        images = batch['i'].to(device)
        dataset_id = batch['dataset_id']
        
        if classify_tags:
            tags = batch['tags'].to(device)
        if classify_orientation:            
            orientation = batch['orientation'].to(device)
        if classify_chexpert:
            chexpert = batch['chexpert'].to(device)
        if classify_questions:
            question_labels = batch['qlabels'].to(device)
        if classify_chest_imagenome:
            chest_imagenome = batch['chest_imagenome'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'images': images,
            }
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
                if classify_chest_imagenome:
                    pred_chest_imagenome_logits = model_output['pred_chest_imagenome']
                    pred_chest_imagenome_probs = model_output['pred_chest_imagenome_probs']

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
                    if classify_chest_imagenome:
                        chest_imagenome_loss = chest_imagenome_multilabel_criterion(pred_chest_imagenome_logits, chest_imagenome.float())
                        losses.append(chest_imagenome_loss)

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
            output['pred_chexpert'] = (pred_chexpert_logits.detach() > 0).cpu()
            output['pred_chexpert_probs'] = pred_chexpert_probs.detach().cpu()
            if training:
                output['chexpert_loss'] = chexpert_loss.detach()
        if classify_questions:
            output['qlabels'] = question_labels.detach().cpu()
            output['pred_qlabels'] = (pred_qlabels_logits.detach() > 0).cpu()
            if training:
                output['qlabels_loss'] = qlabels_loss.detach()
        if classify_chest_imagenome:
            output['chest_imagenome'] = chest_imagenome.detach().cpu()
            output[f'pred_chest_imagenome'] = (pred_chest_imagenome_logits.detach() > 0).cpu()
            output[f'pred_chest_imagenome_probs'] = pred_chest_imagenome_probs.detach().cpu()
            if training:
                output[MetricNames.CHEST_IMAGENOME_LABEL_LOSS] = chest_imagenome_loss.detach()

        return output

    def step_fn__chexpert(batch):

        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        images = batch['i'].to(device)
        orientations = batch['o'].to(device)
        genders = batch['g'].to(device)
        chexpert = batch['l'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {
                'images': images,
                'chexpert_forward': True,
                'dataset_id': dataset_id,
            }

            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision

                model_output = model(**model_kwargs)
                
                pred_chexpert_logits = model_output['pred_chexpert']
                pred_chexpert_probs = model_output['pred_chexpert_probs']
                pred_orientation_logits = model_output['pred_orientation']
                pred_gender_logits = model_output['pred_gender']

                if training:                    
                    # Compute losses
                    chexpert_loss = chexpert_criterion(pred_chexpert_logits, chexpert.float())
                    orientation_loss = chexpert_aux_criterion(pred_orientation_logits, orientations)
                    gender_loss = chexpert_aux_criterion(pred_gender_logits, genders)                    
                    batch_loss = chexpert_loss + orientation_loss + gender_loss

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
        if training:
            output['loss'] = batch_loss.detach()
            
        output['chexpert'] = chexpert.detach().cpu()
        output['pred_chexpert'] = (pred_chexpert_logits.detach() > 0).cpu()
        output['pred_chexpert_probs'] = pred_chexpert_probs.detach().cpu()
        if training:
            output['chexpert_loss'] = chexpert_loss.detach()

        output['orientation'] = orientations.detach()
        output['pred_orientation'] = pred_orientation_logits.argmax(-1).detach()        
        if training:
            output['orientation_loss'] = orientation_loss.detach()

        output['gender'] = genders.detach()
        output['pred_gender'] = pred_gender_logits.argmax(-1).detach()        
        if training:
            output['gender_loss'] = gender_loss.detach()

        return output
    
    def step_fn(unused_engine, batch):
        dataset_id = batch['dataset_id']
        # print(f"step_fn(dataset_id={dataset_id})")
        if dataset_id == MIMICCXR_DATASET_ID or dataset_id == IUXRAY_DATASET_ID:
            return step_fn__mimiccxr_iuxray(batch)
        if dataset_id == CHEXPERT_DATASET_ID:
            return step_fn__chexpert(batch)
        assert False, f'Unknown dataset_id {dataset_id}'

    return step_fn

def get_engine(model, classify_tags, classify_orientation, classify_chexpert, classify_questions,
               classify_chest_imagenome, device,
               binary_loss_name='bce',
               use_amp=False,
               training=False,
               train_with_chexpert_dataset=False,
               optimizer=None):    
    
    # Auxiliary tasks
    if classify_tags:
        tags_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        tags_criterion = None
    
    if classify_orientation:
        iuxray_orientation_criterion = nn.CrossEntropyLoss()
        mimiccxr_orientation_criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore unknown
    else:
        iuxray_orientation_criterion = None
        mimiccxr_orientation_criterion = None

    if classify_chexpert or train_with_chexpert_dataset:
        chexpert_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        chexpert_criterion = None
    
    if classify_questions:
        question_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        question_criterion = None

    if classify_chest_imagenome:
        chest_imagenome_multilabel_criterion = get_binary_multilabel_loss(binary_loss_name)
    else:
        chest_imagenome_multilabel_criterion = None

    if train_with_chexpert_dataset:
        chexpert_aux_criterion = nn.CrossEntropyLoss()
    else:
        chexpert_aux_criterion = None

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
                          # chexpert dataset
                          chexpert_aux_criterion=chexpert_aux_criterion,
                          # chest imagenome dataset
                          classify_chest_imagenome=classify_chest_imagenome,
                          chest_imagenome_multilabel_criterion=chest_imagenome_multilabel_criterion,                          
                        )
    engine = Engine(step_fn)
    return engine