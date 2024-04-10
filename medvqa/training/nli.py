import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.losses.optimizers import GradientAccumulator

def get_bert_based_nli_step_fn(model, optimizer, training, validating, testing, device, nli_criterion,
                merged_input=False,
                iters_to_accumulate=1, # for gradient accumulation
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
    
    def step_fn(batch):

        # Extract elements from batch
        if merged_input:
            tokenized_texts = batch['tokenized_texts']
            tokenized_texts = {k: v.to(device) for k, v in tokenized_texts.items()}
        else:
            tokenized_premises = batch['tokenized_premises']
            tokenized_hypotheses = batch['tokenized_hypotheses']
            tokenized_premises = {k: v.to(device) for k, v in tokenized_premises.items()}
            tokenized_hypotheses = {k: v.to(device) for k, v in tokenized_hypotheses.items()}
        labels = batch['labels'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)
            
            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                if merged_input:
                    logits = model(tokenized_texts)
                else:
                    logits = model(tokenized_premises, tokenized_hypotheses)
                if training:
                    losses = []
                    nli_loss = nli_criterion(logits, labels)
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
    
    def step_fn_wrapper(unused_engine, batch):
        output = step_fn(batch)
        # update learning rate batchwise
        if update_lr_batchwise:
            lr_scheduler.step()
        return output
    
    return step_fn_wrapper

def get_embedding_based_nli_step_fn(model, optimizer, training, validating, testing, device, nli_criterion,
                iters_to_accumulate=1, # for gradient accumulation
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
    
    def step_fn(batch):

        # Extract elements from batch
        h_embs = batch['h_embs'].to(device)
        p_most_sim_embs = batch['p_most_sim_embs'].to(device)
        p_least_sim_embs = batch['p_least_sim_embs'].to(device)
        p_max_embs = batch['p_max_embs'].to(device)
        p_avg_embs = batch['p_avg_embs'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)
            
            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                logits = model(h_embs, p_most_sim_embs, p_least_sim_embs, p_max_embs, p_avg_embs)
                if training:
                    losses = []
                    # print('logits:', logits)
                    # print('logits.shape:', logits.shape)
                    nli_loss = nli_criterion(logits, labels)
                    # print('nli_loss:', nli_loss)
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
    
    def step_fn_wrapper(unused_engine, batch):
        output = step_fn(batch)
        # update learning rate batchwise
        if update_lr_batchwise:
            lr_scheduler.step()
        return output
    
    return step_fn_wrapper

def get_bert_based_nli_engine(model, device,
               iters_to_accumulate=1,
               use_amp=False,
               training=False,
               validating=False,
               testing=False,
               optimizer=None,
               update_lr_batchwise=False, lr_scheduler=None,
               merged_input=False,
            ):
    nli_criterion = nn.CrossEntropyLoss()
    
    # Create engine
    step_fn = get_bert_based_nli_step_fn(model=model, optimizer=optimizer, device=device,
                          training=training, validating=validating, testing=testing,
                          iters_to_accumulate=iters_to_accumulate,
                          use_amp=use_amp,
                          nli_criterion=nli_criterion,
                          # batchwise learning rate updates
                          update_lr_batchwise=update_lr_batchwise,
                          lr_scheduler=lr_scheduler,
                          merged_input=merged_input,
                          )
    engine = Engine(step_fn)
    return engine

def get_embedding_based_nli_engine(model, device,
               iters_to_accumulate=1,
               use_amp=False,
               training=False,
               validating=False,
               testing=False,
               optimizer=None,
               update_lr_batchwise=False, lr_scheduler=None,
            ):
    nli_criterion = nn.CrossEntropyLoss()
    
    # Create engine
    step_fn = get_embedding_based_nli_step_fn(model=model, optimizer=optimizer, device=device,
                          training=training, validating=validating, testing=testing,
                          iters_to_accumulate=iters_to_accumulate,
                          use_amp=use_amp,
                          nli_criterion=nli_criterion,
                          # batchwise learning rate updates
                          update_lr_batchwise=update_lr_batchwise,
                          lr_scheduler=lr_scheduler,
                          )
    engine = Engine(step_fn)
    return engine