import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.losses.optimizers import GradientAccumulator
from medvqa.losses.wce import WeigthedByClassCrossEntropyLoss

def get_step_fn(model, optimizer, training, validating, testing, device,
        iters_to_accumulate=1, # for gradient accumulation
        triplet_loss_criterion=None, # for triplet ranking
        classifier_loss_criterion=None, # for classification
        # automatic mixed precision
        use_amp=False,
        # batchwise learning rate updates
        update_lr_batchwise=False,
        lr_scheduler=None,
    ):

    scaler = GradScaler(enabled=use_amp)
    
    assert (training + validating + testing) == 1, 'Only one of training, validating, testing must be True'

    if training:
        gradient_accumulator = GradientAccumulator(optimizer, scaler, iters_to_accumulate)
    
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
                    losses.append(triplet_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss)

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
    
    def step_fn__classification(batch):

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
                output = model(input_ids=input_ids, attention_mask=attention_mask, run_auxiliary_tasks=True)
                c_logits = output['category_logits']
                hs_logits = output['health_status_logits']
                cs_logits = output['comparison_status_logits']

                if training:
                    losses = []
                    c_loss = classifier_loss_criterion(c_logits, c_labels)
                    losses.append(c_loss)
                    hs_loss = classifier_loss_criterion(hs_logits, hs_labels)
                    losses.append(hs_loss)
                    cs_loss = classifier_loss_criterion(cs_logits, cs_labels)
                    losses.append(cs_loss)
                    batch_loss = sum(losses)
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss)

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
    
    def step_fn_wrapper(unused_engine, batch):
        flag = batch['flag']
        if flag == 't': # triplet ranking
            output = step_fn__triplet_ranking(batch)
        elif flag == 'c': # classification
            output = step_fn__classification(batch)
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
            ):
    
    # Triplet loss criterion as binary cross entropy
    triplet_loss_criterion = nn.BCEWithLogitsLoss()

    # Classification loss criterion as weighted cross entropy
    classifier_loss_criterion = WeigthedByClassCrossEntropyLoss()
    
    # Create engine
    step_fn = get_step_fn(model=model, optimizer=optimizer, device=device,
                          training=training, validating=validating, testing=testing,
                          iters_to_accumulate=iters_to_accumulate,
                          triplet_loss_criterion=triplet_loss_criterion,
                          classifier_loss_criterion=classifier_loss_criterion,
                          use_amp=use_amp,
                          # batchwise learning rate updates
                          update_lr_batchwise=update_lr_batchwise,
                          lr_scheduler=lr_scheduler,
                          )
    engine = Engine(step_fn)
    return engine