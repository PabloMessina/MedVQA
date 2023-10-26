import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.losses import Focal_BCE_WBCE_Loss
from medvqa.losses.optimizers import GradientAccumulator
from medvqa.losses.spert_loss import SpERTLoss
from medvqa.losses.wce import WeigthedByClassCrossEntropyLoss
from medvqa.training.utils import batch_to_device

def get_step_fn(model, optimizer, training, validating, testing, device,
        iters_to_accumulate=1, # for gradient accumulation
        triplet_loss_criterion=None, # for triplet ranking
        metadata_classifier_loss_criterion=None, # for metadata classification
        chest_imagenome_classifier_loss_criterion=None, # for chest imagenome classification
        nli_loss_criterion=None, # for NLI
        entcon_loss_criterion=None, # for entailment contradiction
        spert_loss_criterion=None, # for SpERT
        decoder_criterion=None, # for text decoder
        max_grad_norm=None,
        # automatic mixed precision
        use_amp=False,
        # batchwise learning rate updates
        update_lr_batchwise=False,
        lr_scheduler=None,
        # loss weights
        triplet_loss_weight=1.0,
        category_classif_loss_weight=1.0,
        health_status_classif_loss_weight=1.0,
        comparison_status_classif_loss_weight=1.0,
        chest_imagenome_obs_classif_loss_weight=1.0,
        chest_imagenome_anatloc_classif_loss_weight=1.0,
        nli_loss_weight=1.0,
        entcon_loss_weight=1.0,
        sentence_autoencoder_loss_weight=1.0,
    ):

    scaler = GradScaler(enabled=use_amp)
    
    assert (training + validating + testing) == 1, 'Only one of training, validating, testing must be True'

    if training:
        gradient_accumulator = GradientAccumulator(optimizer, scaler, iters_to_accumulate, max_grad_norm)
    
    def step_fn__triplet_ranking(batch):

        # Extract elements from batch
        a_input_ids = batch['a_input_ids'].to(device) # anchor
        a_attention_mask = batch['a_attention_mask'].to(device)
        p_input_ids = batch['p_input_ids'].to(device) # positive
        p_attention_mask = batch['p_attention_mask'].to(device)
        n_input_ids = batch['n_input_ids'].to(device) # negative
        n_attention_mask = batch['n_attention_mask'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)
            
            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                a_embeddings = model(input_ids=a_input_ids, attention_mask=a_attention_mask)['text_embeddings']
                p_embeddings = model(input_ids=p_input_ids, attention_mask=p_attention_mask)['text_embeddings']
                n_embeddings = model(input_ids=n_input_ids, attention_mask=n_attention_mask)['text_embeddings']

                # dot product of embeddings
                ap_sim = torch.sum(a_embeddings * p_embeddings, dim=1)
                an_sim = torch.sum(a_embeddings * n_embeddings, dim=1)
                diff = ap_sim - an_sim

                if training:
                    losses = []
                    triplet_loss = triplet_loss_criterion(diff, torch.ones_like(diff))
                    triplet_loss = triplet_loss * triplet_loss_weight
                    losses.append(triplet_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

        # Prepare output
        output = {
            'triplet_logits': diff.detach(),
        }
        if training:
            output['loss'] = batch_loss.detach()
            output['triplet_loss'] = triplet_loss.detach()
        if 'rule_id' in batch:  # used for rule-wise evaluation
            output['rule_id'] = batch['rule_id']

        return output
    
    def step_fn__metadata_classification(batch):

        # Extract elements from batch
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        c_labels = batch['c'].to(device) # category
        hs_labels = batch['hs'].to(device) # health status
        cs_labels = batch['cs'].to(device) # comparison status
        
        with torch.set_grad_enabled(training):

            model.train(training)
            
            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                output = model(input_ids=input_ids, attention_mask=attention_mask, run_metadata_auxiliary_tasks=True)
                c_logits = output['category_logits']
                hs_logits = output['health_status_logits']
                cs_logits = output['comparison_status_logits']

                if training:
                    losses = []
                    c_loss = metadata_classifier_loss_criterion(c_logits, c_labels)
                    c_loss = c_loss * category_classif_loss_weight
                    losses.append(c_loss)
                    hs_loss = metadata_classifier_loss_criterion(hs_logits, hs_labels)
                    hs_loss = hs_loss * health_status_classif_loss_weight
                    losses.append(hs_loss)
                    cs_loss = metadata_classifier_loss_criterion(cs_logits, cs_labels)
                    cs_loss = cs_loss * comparison_status_classif_loss_weight
                    losses.append(cs_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

        # Prepare output
        output = {
            'pred_category': torch.argmax(c_logits, dim=1).detach(),
            'gt_category': c_labels.detach(),
            'pred_health_status': torch.argmax(hs_logits, dim=1).detach(),
            'gt_health_status': hs_labels.detach(),
            'pred_comparison_status': torch.argmax(cs_logits, dim=1).detach(),
            'gt_comparison_status': cs_labels.detach(),
        }
        if training:
            output['loss'] = batch_loss.detach()
            output['c_loss'] = c_loss.detach()
            output['hs_loss'] = hs_loss.detach()
            output['cs_loss'] = cs_loss.detach()

        return output
    
    def step_fn__chest_imagenome_observation_classification(batch):

        # Extract elements from batch
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['l'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)
            
            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                output = model(input_ids=input_ids, attention_mask=attention_mask, run_chest_imagenome_obs_task=True)
                logits = output['chest_imagenome_obs_logits']

                if training:
                    losses = []
                    chstimgn_loss = chest_imagenome_classifier_loss_criterion(logits, labels.float())
                    chstimgn_loss = chstimgn_loss * chest_imagenome_obs_classif_loss_weight
                    losses.append(chstimgn_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

        # Prepare output
        output = {
            'pred_labels': (logits.detach() > 0).cpu(),
            'gt_labels': labels.detach(),
        }
        if training:
            output['loss'] = batch_loss.detach()
            output['chstimgn_loss'] = chstimgn_loss.detach()

        return output
    
    def step_fn__chest_imagenome_anatomical_location_classification(batch):

        # Extract elements from batch
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['l'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)
            
            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                output = model(input_ids=input_ids, attention_mask=attention_mask, run_chest_imagenome_anatloc_task=True)
                logits = output['chest_imagenome_anatloc_logits']

                if training:
                    losses = []
                    chstimgn_loss = chest_imagenome_classifier_loss_criterion(logits, labels.float())
                    chstimgn_loss = chstimgn_loss * chest_imagenome_anatloc_classif_loss_weight
                    losses.append(chstimgn_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

        # Prepare output
        output = {
            'pred_labels': (logits.detach() > 0).cpu(),
            'gt_labels': labels.detach(),
        }
        if training:
            output['loss'] = batch_loss.detach()
            output['chstimgn_loss'] = chstimgn_loss.detach()

        return output
    
    def step_fn__nli(batch):

        # Extract elements from batch
        tokenized_premises = batch['tokenized_premises']
        p_input_ids = tokenized_premises['input_ids'].to(device)
        p_attention_mask = tokenized_premises['attention_mask'].to(device)
        tokenized_hypotheses = batch['tokenized_hypotheses']
        h_input_ids = tokenized_hypotheses['input_ids'].to(device)
        h_attention_mask = tokenized_hypotheses['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)
            
            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                logits = model.nli_forward(p_input_ids, p_attention_mask, h_input_ids, h_attention_mask)
                
                if training:
                    losses = []
                    nli_loss = nli_loss_criterion(logits, labels)
                    nli_loss = nli_loss * nli_loss_weight
                    losses.append(nli_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

        # Prepare output
        output = {
            'pred_labels': torch.argmax(logits.detach(), dim=1),
            'gt_labels': labels.detach(),
        }
        if training:
            output['loss'] = batch_loss.detach()
            output['nli_loss'] = nli_loss.detach()

        return output
    
    def step_fn__entcon(batch):

        # Extract elements from batch
        tokenized_ent_p = batch['tokenized_ent_p']
        tokenized_ent_h = batch['tokenized_ent_h']
        tokenized_con_p = batch['tokenized_con_p']
        tokenized_con_h = batch['tokenized_con_h']
        ent_p_input_ids = tokenized_ent_p['input_ids'].to(device)
        ent_p_attention_mask = tokenized_ent_p['attention_mask'].to(device)
        ent_h_input_ids = tokenized_ent_h['input_ids'].to(device)
        ent_h_attention_mask = tokenized_ent_h['attention_mask'].to(device)
        con_p_input_ids = tokenized_con_p['input_ids'].to(device)
        con_p_attention_mask = tokenized_con_p['attention_mask'].to(device)
        con_h_input_ids = tokenized_con_h['input_ids'].to(device)
        con_h_attention_mask = tokenized_con_h['attention_mask'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)
            
            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                ent_p_embeddings = model(input_ids=ent_p_input_ids, attention_mask=ent_p_attention_mask)['text_embeddings']
                ent_h_embeddings = model(input_ids=ent_h_input_ids, attention_mask=ent_h_attention_mask)['text_embeddings']
                con_p_embeddings = model(input_ids=con_p_input_ids, attention_mask=con_p_attention_mask)['text_embeddings']
                con_h_embeddings = model(input_ids=con_h_input_ids, attention_mask=con_h_attention_mask)['text_embeddings']
                # dot(ent_p, ent_h) - dot(con_p, con_h) > 0
                ent_sim = torch.sum(ent_p_embeddings * ent_h_embeddings, dim=1)
                con_sim = torch.sum(con_p_embeddings * con_h_embeddings, dim=1)
                diff = ent_sim - con_sim

                if training:
                    losses = []
                    entcon_loss = entcon_loss_criterion(diff, torch.ones_like(diff))
                    entcon_loss = entcon_loss * entcon_loss_weight
                    losses.append(entcon_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

        # Prepare output
        output = {
            'logits': diff.detach(),
        }
        if training:
            output['loss'] = batch_loss.detach()
            output['entcon_loss'] = entcon_loss.detach()

        return output
    
    def step_fn__spert(batch):

        assert training, 'Only training is supported for SpERT'
        
        batch = batch_to_device(batch, device)
        
        with torch.set_grad_enabled(training):

            model.train(training)
            
            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision

                entity_logits, rel_logits = model.spert_forward_train(
                    encodings=batch['encodings'], context_masks=batch['context_masks'],
                    entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                    relations=batch['rels'], rel_masks=batch['rel_masks'],
                )

                if training:
                    losses = []
                    spert_loss = spert_loss_criterion.compute(
                        entity_logits=entity_logits, rel_logits=rel_logits,
                        entity_types=batch['entity_types'], rel_types=batch['rel_types'],
                        entity_sample_masks=batch['entity_sample_masks'], rel_sample_masks=batch['rel_sample_masks'])
                    losses.append(spert_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

        # Prepare output
        output = {}
        if training:
            output['loss'] = batch_loss.detach()
            output['spert_loss'] = spert_loss.detach()

        return output
    
    def step_fn__sentence_autoencoder(batch):

        assert training or validating
        
        tokenized_sentences = batch['tokenized_sentences']
        input_ids = tokenized_sentences['input_ids'].to(device)
        attention_mask = tokenized_sentences['attention_mask'].to(device)
        decoder_ids = batch['decoder_ids'].to(device)
        # shift decoder_ids by one position for teacher forcing
        decoder_ids_start = decoder_ids[:, :-1] # ignore last token
        decoder_ids_end = decoder_ids[:, 1:] # ignore first token
        
        with torch.set_grad_enabled(training):
            model.train(training)
            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                decoder_logits = model.fact_decoder_forward_teacher_forcing(
                    input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_ids_start,
                )
                decoder_loss = decoder_criterion(decoder_logits.reshape(-1, decoder_logits.shape[-1]), decoder_ids_end.reshape(-1))
                if training:
                    decoder_loss = decoder_loss * sentence_autoencoder_loss_weight # scale loss by weight
                    losses = []
                    losses.append(decoder_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

        # Prepare output
        output = {}
        output['sae_loss'] = decoder_loss.detach()
        if training:
            output['loss'] = batch_loss.detach()

        return output
    
    def step_fn_wrapper(unused_engine, batch):
        flag = batch['flag']
        if flag == 't': # triplet ranking
            output = step_fn__triplet_ranking(batch)
        elif flag == 'mc': # metadata classification
            output = step_fn__metadata_classification(batch)
        elif flag == 'cioc': # chest imagenome observation classification
            output = step_fn__chest_imagenome_observation_classification(batch)
        elif flag == 'cialc': # chest imagenome anatomical location classification
            output = step_fn__chest_imagenome_anatomical_location_classification(batch)
        elif flag == 'nli': # NLI
            output = step_fn__nli(batch)
        elif flag == 'entcon': # entailment contradiction
            output = step_fn__entcon(batch)
        elif flag == 'spert': # SpERT
            output = step_fn__spert(batch)
        elif flag == 'sae': # sentence autoencoder
            output = step_fn__sentence_autoencoder(batch)
        else:
            raise ValueError(f'Invalid flag: {flag}')
        output['flag'] = flag # propagate flag
        # update learning rate batchwise
        if update_lr_batchwise:
            lr_scheduler.step()
        return output
    
    return step_fn_wrapper

def get_engine(model, device,
               iters_to_accumulate=1,
               use_amp=False,
               training=False,
               validating=False,
               testing=False,
               optimizer=None,
               update_lr_batchwise=False, lr_scheduler=None,
               max_grad_norm=None,
               triplet_loss_weight=1.0,
               category_classif_loss_weight=1.0,
               health_status_classif_loss_weight=1.0,
               comparison_status_classif_loss_weight=1.0,
               chest_imagenome_obs_classif_loss_weight=1.0,
               chest_imagenome_anatloc_classif_loss_weight=1.0,
               sentence_autoencoder_loss_weight=1.0,
               nli_loss_weight=1.0,
               entcon_loss_weight=1.0,
            ):
    
    # Triplet loss criterion as binary cross entropy
    triplet_loss_criterion = nn.BCEWithLogitsLoss()

    # Metadata classification loss criterion as weighted cross entropy
    metadata_classifier_loss_criterion = WeigthedByClassCrossEntropyLoss()

    # Chest imagenome classification loss criterion as weighted cross entropy
    chest_imagenome_classifier_loss_criterion = Focal_BCE_WBCE_Loss()

    # NLI loss
    nli_loss_criterion = nn.CrossEntropyLoss()

    # Entailment contradiction loss
    entcon_loss_criterion = nn.BCEWithLogitsLoss() # dot(ent_p, ent_h) - dot(con_p, con_h) > 0

    # SpERT loss
    rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    spert_loss_criterion = SpERTLoss(rel_criterion, entity_criterion)

    # Decoder loss
    decoder_criterion = nn.CrossEntropyLoss()
    
    # Create engine
    step_fn = get_step_fn(model=model, optimizer=optimizer, device=device,
                          training=training, validating=validating, testing=testing,
                          iters_to_accumulate=iters_to_accumulate,
                          triplet_loss_criterion=triplet_loss_criterion,
                          metadata_classifier_loss_criterion=metadata_classifier_loss_criterion,
                          chest_imagenome_classifier_loss_criterion=chest_imagenome_classifier_loss_criterion,
                          nli_loss_criterion=nli_loss_criterion,
                          entcon_loss_criterion=entcon_loss_criterion,
                          spert_loss_criterion=spert_loss_criterion,
                          decoder_criterion=decoder_criterion,
                          max_grad_norm=max_grad_norm,
                          use_amp=use_amp,
                          # batchwise learning rate updates
                          update_lr_batchwise=update_lr_batchwise,
                          lr_scheduler=lr_scheduler,
                          # loss weights
                          triplet_loss_weight=triplet_loss_weight,
                          category_classif_loss_weight=category_classif_loss_weight,
                          health_status_classif_loss_weight=health_status_classif_loss_weight,
                          comparison_status_classif_loss_weight=comparison_status_classif_loss_weight,
                          chest_imagenome_obs_classif_loss_weight=chest_imagenome_obs_classif_loss_weight,
                          chest_imagenome_anatloc_classif_loss_weight=chest_imagenome_anatloc_classif_loss_weight,
                          nli_loss_weight=nli_loss_weight,
                          entcon_loss_weight=entcon_loss_weight,
                          sentence_autoencoder_loss_weight=sentence_autoencoder_loss_weight,
                          )
    engine = Engine(step_fn)
    return engine