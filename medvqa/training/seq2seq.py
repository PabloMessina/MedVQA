import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.losses.optimizers import GradientAccumulator

def get_step_fn(model, optimizer, training, validating, testing, device, max_len=None,
        iters_to_accumulate=1, # for gradient accumulation
        nlg_criterion=None, # for report generation
        # automatic mixed precision
        use_amp=False,
        # batchwise learning rate updates
        update_lr_batchwise=False,
        lr_scheduler=None,
        # other args
        use_t5=False,
        use_flan_t5=False,
        use_bart=False,
        num_beams=1,
    ):

    scaler = GradScaler(enabled=use_amp)
    
    assert sum([training, validating, testing]) == 1, 'Only one of training, validating, testing must be True'
    assert use_t5 or use_flan_t5 or use_bart # TODO: support more models eventually

    if training:
        gradient_accumulator = GradientAccumulator(optimizer, scaler, iters_to_accumulate)
        assert nlg_criterion is not None, 'Training requires report loss'
    
    def step_fn(batch):

        # Extract elements from batch
        output_ids = batch['output_ids'].to(device)
        output_text = batch['output_text']
        if use_t5 or use_flan_t5 or use_bart:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
        
        with torch.set_grad_enabled(training):

            model.train(training)

            # Prepare args for model forward
            model_kwargs = {}
            if training:
                model_kwargs['mode'] = 'train'
            elif validating:
                model_kwargs['mode'] = 'val'
            elif testing:
                model_kwargs['mode'] = 'test'

            if use_t5 or use_flan_t5 or use_bart:
                model_kwargs['input_ids'] = input_ids
                model_kwargs['attention_mask'] = attention_mask
                model_kwargs['labels'] = output_ids
                if testing:
                    model_kwargs['max_len'] = max_len
                    model_kwargs['num_beams'] = num_beams
            
            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                model_output = model(**model_kwargs)
                if training:
                    losses = []
                if use_t5 or use_flan_t5 or use_bart:
                    if training or validating:
                        pred_output_logits = model_output.logits.detach()
                        pred_output_ids = pred_output_logits.argmax(dim=-1)
                        seq2seq_loss = model_output.loss
                        if training:
                            losses.append(seq2seq_loss)
                    else:
                        pred_output_ids = model_output.detach()

                if training:
                    if len(losses) > 0:
                        batch_loss = sum(losses)
                    else:
                        batch_loss = None
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(batch_loss, model)

        # Prepare output
        output = {}
        output['gt_text'] = output_text
        if training and batch_loss is not None:
            output['loss'] = batch_loss.detach()
        if use_t5 or use_flan_t5 or use_bart:
            output['pred_output_ids'] = pred_output_ids.detach()
            if training or validating:
                output['seq2seq_loss'] = seq2seq_loss.detach()

        return output
    
    def step_fn_wrapper(unused_engine, batch):
        output = step_fn(batch)
        # update learning rate batchwise
        if update_lr_batchwise:
            lr_scheduler.step()
        return output
    
    return step_fn_wrapper

def get_engine(model, device,
                max_len=None,
                iters_to_accumulate=1,
                use_amp=False,
                training=False,
                validating=False,
                testing=False,
                optimizer=None,
                update_lr_batchwise=False, lr_scheduler=None,
                use_t5=False,
                use_flan_t5=False,
                use_bart=False,
                num_beams=1,
            ):
    
    # Criterion
    if use_t5 or use_flan_t5 or use_bart:
        nlg_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        nlg_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Create engine
    step_fn = get_step_fn(model=model, optimizer=optimizer, device=device,
                          training=training, validating=validating, testing=testing,                          
                          max_len=max_len,
                          iters_to_accumulate=iters_to_accumulate,
                          nlg_criterion=nlg_criterion,
                          use_amp=use_amp,
                          # batchwise learning rate updates
                          update_lr_batchwise=update_lr_batchwise,
                          lr_scheduler=lr_scheduler,
                          # other kwargs
                          use_t5=use_t5,
                          use_flan_t5=use_flan_t5,
                          use_bart=use_bart,
                          num_beams=num_beams,
                          )
    engine = Engine(step_fn)
    return engine