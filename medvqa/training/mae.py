import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from ignite.engine import Engine
from medvqa.losses.optimizers import GradientAccumulator

def get_step_fn(model, optimizer, training, device,        
        # gradient accumulation
        num_accumulation_steps=1,
        # automatic mixed precision
        use_amp=False,
        # batchwise learning rate updates
        update_lr_batchwise=False,
        lr_scheduler=None,
    ):

    scaler = GradScaler(enabled=use_amp)

    if update_lr_batchwise:
        assert lr_scheduler is not None

    if training:
        gradient_accumulator = GradientAccumulator(optimizer, scaler, num_accumulation_steps)
    
    def step_fn(unused_engine, batch):
            
        # Extract elements from batch
        idxs = batch['idx']
        dataset_id = batch['dataset_id']
        images = batch['i'].to(device)

        with torch.set_grad_enabled(training):
            model.train(training)
            # Forward pass
            with autocast(enabled=use_amp): # automatic mixed precision
                outputs = model(images)                
                if training:
                    # Backward pass + optimizer step if training
                    gradient_accumulator.step(outputs.loss)

        # Update learning rate batchwise
        if update_lr_batchwise:
            lr_scheduler.step()

        # Prepare output
        output = {
            'idxs': idxs,
            'dataset_id': dataset_id,
            'loss': outputs.loss.detach(),
        }
        return output
    
    return step_fn

def get_engine(model, device,
               num_accumulation_steps=1,
               use_amp=False,
               training=False,
               optimizer=None,
               update_lr_batchwise=False, lr_scheduler=None,
    ):
    step_fn = get_step_fn(model, optimizer,
                          training=training,
                          device=device, use_amp=use_amp,
                          num_accumulation_steps=num_accumulation_steps,                          
                          # batchwise learning rate updates
                          update_lr_batchwise=update_lr_batchwise,
                          lr_scheduler=lr_scheduler,
                          )
    engine = Engine(step_fn)
    return engine