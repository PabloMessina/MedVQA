from torch.optim import Adam, AdamW, SGD
from torch.nn.utils import clip_grad_norm_

def create_optimizer(name, params, lr):
    print(f'create_optimizer(): name = {name}')
    if name == 'adam':
        optimizer = Adam(params, lr=lr)
    elif name == 'adamw':
        optimizer = AdamW(params, lr=lr)
    elif name == 'sgd':
        optimizer = SGD(params, lr=lr)
    else:
        assert False, f'Unknown optimizer {name}'
    return optimizer

class GradientAccumulator:
    def __init__(self, optimizer, scaler, num_accumulation_steps, max_grad_norm=None):
        print(f'GradientAccumulator.__init__(): num_accumulation_steps = {num_accumulation_steps}, max_grad_norm = {max_grad_norm}')
        self.optimizer = optimizer
        self.scaler = scaler
        self.num_accumulation_steps = num_accumulation_steps
        self.step_count = 0
        self._max_grad_norm = max_grad_norm

    def step(self, batch_loss, model):
        # print(f'GradientAccumulator.step(): batch_loss = {batch_loss}')
        assert batch_loss is not None
        batch_loss = batch_loss / self.num_accumulation_steps
        self.scaler.scale(batch_loss).backward()
        if self._max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), self._max_grad_norm)
        self.step_count += 1
        if self.step_count % self.num_accumulation_steps == 0:
            print(f'*** GradientAccumulator.step(): step_count = {self.step_count}, num_accumulation_steps = {self.num_accumulation_steps}')
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # batch_loss.backward()
            # self.optimizer.step()
            self.optimizer.zero_grad()