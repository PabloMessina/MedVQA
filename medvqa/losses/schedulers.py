from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    ExponentialLR,
    _LRScheduler,
)
from bisect import bisect_right
import math
import warnings
import logging

logger = logging.getLogger(__name__)

class LRSchedulerNames:
    ReduceLROnPlateau = 'reduce-lr-on-plateau'
    ExponentialWarmUpPlusDecay = 'exp-warmup+decay'
    ExponentialWarmUpPlusCyclicDecay = 'exp-warmup+cyclicdecay'
    ExponentialWarmUpPlusCyclicDecayBatchwise = 'exp-warmup+cyclicdecay-batchwise'
    ExponentialWarmUpPlusCosineAnnealing = 'exp-warmup+cosine'
    ExponentialWarmUpPlusDecayPlusCyclicDecay = 'exp-warmup+decay+cyclicdecay'

class SequentialLR(_LRScheduler):
    def __init__(self, schedulers, milestones, last_epoch=-1):
        for scheduler_idx in range(1, len(schedulers)):
            if (schedulers[scheduler_idx].optimizer != schedulers[0].optimizer):
                raise ValueError(
                    "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                    "got schedulers at index {} and {} to be different".format(0, scheduler_idx)
                )
        if (len(milestones) != len(schedulers) - 1):
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                "than the number of milestone points, but got number of schedulers {} and the "
                "number of milestones to be equal to {}".format(len(schedulers), len(milestones))
            )
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        self._schedulers[idx].step()

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', '_schedulers')}
        state_dict['_schedulers'] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            state_dict['_schedulers'][idx] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        _schedulers = state_dict.pop('_schedulers')
        self.__dict__.update(state_dict)
        state_dict['_schedulers'] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)

class CosineAnnealingWarmRestarts(_LRScheduler):

    def __init__(self, optimizer, eta_0, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_0 = eta_0

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)

        self.T_cur = self.last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [self.eta_min + (self.eta_0 - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for _ in self.base_lrs]

    def step(self, epoch=None):

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class CyclicExponentialLR(_LRScheduler):

    def __init__(self, optimizer, gamma, steps_to_restart, initial_lr, last_epoch=-1):
        self.gamma = gamma
        self.steps_to_restart = steps_to_restart
        self.steps = -1
        self.initial_lr = initial_lr
        logger.info(f'self.steps_to_restart = {self.steps_to_restart}')
        logger.info(f'self.steps = {self.steps}')
        logger.info(f'self.initial_lr = {self.initial_lr}')
        super(CyclicExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.steps == 0:
            next_lrs = [self.initial_lr for _ in self.optimizer.param_groups]
        else:
            next_lrs = [group['lr'] * self.gamma for group in self.optimizer.param_groups]
        
        self.steps += 1
        if self.steps >= self.steps_to_restart:
            self.steps = 0
        return next_lrs


def calc_gamma(lr_a, lr_b, steps):
    return (lr_b / lr_a) ** (1/steps)

def parse_warmup_and_decay_args_string(string):
    args = string.split(',')
    assert len(args) == 5
    return float(args[0]), int(args[1]), float(args[2]), int(args[3]), float(args[4])

def parse_warmup_and_cosine_args_string(string):
    args = string.split(',')
    assert len(args) == 5
    return float(args[0]), int(args[1]), float(args[2]), int(args[3]), float(args[4])

def parse_warmup_decay_and_cyclic_decay_args_string(string):
    args = string.split(',')
    assert len(args) == 8
    lr0 = float(args[0])
    steps01 = int(args[1])
    lr1 = float(args[2])
    steps12 = int(args[3])
    lr2 = float(args[4])
    lr3 = float(args[5])
    restart_steps = int(args[6])
    lr4 = float(args[7])
    assert lr0 < lr1
    assert steps01 > 1
    assert lr1 > lr2
    assert steps12 > 1
    assert lr2 < lr3
    assert restart_steps > 1
    assert lr3 > lr4
    return lr0, steps01, lr1, steps12, lr2, lr3, restart_steps, lr4

def create_lr_scheduler(name, optimizer, factor=None, patience=None, warmup_and_decay_args=None,
                        warmup_decay_and_cyclic_decay_args=None, warmup_and_cosine_args=None,
                        n_batches_per_epoch=None):
    if name == LRSchedulerNames.ReduceLROnPlateau:
        assert factor is not None
        assert patience is not None
        scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True,
                                        factor=factor, patience=patience)
        update_batchwise = False
    elif name == LRSchedulerNames.ExponentialWarmUpPlusDecay:
        assert warmup_and_decay_args is not None
        logger.info(f'Using {name} scheduler: {warmup_and_decay_args}')
        lr0, steps01, lr1, steps12, lr2 = parse_warmup_and_decay_args_string(warmup_and_decay_args)
        logger.info(f'lr0 = {lr0}, steps01 = {steps01}, lr1 = {lr1}, steps12 = {steps12}, lr2 = {lr2}')
        scheduler1 = ExponentialLR(optimizer, gamma=calc_gamma(lr0, lr1, steps01), verbose=True)
        scheduler2 = ExponentialLR(optimizer, gamma=calc_gamma(lr1, lr2, steps12), verbose=True)
        scheduler = SequentialLR(schedulers=[scheduler1, scheduler2], milestones=[steps01+1])
        update_batchwise = False
    elif name == LRSchedulerNames.ExponentialWarmUpPlusCyclicDecayBatchwise:
        assert warmup_and_decay_args is not None
        assert n_batches_per_epoch is not None
        logger.info(f'Using {name} scheduler: {warmup_and_decay_args}')
        lr0, warmup_epochs, lr1, restart_epochs, lr2 = parse_warmup_and_decay_args_string(warmup_and_decay_args)
        logger.info(f'lr0 = {lr0}, warmup_epochs = {warmup_epochs}, lr1 = {lr1}, restart_epochs = {restart_epochs}, lr2 = {lr2}')
        warmup_steps = warmup_epochs * n_batches_per_epoch
        restart_steps = restart_epochs * n_batches_per_epoch
        scheduler1 = ExponentialLR(optimizer, gamma=calc_gamma(lr0, lr1, warmup_steps))
        scheduler2 = CyclicExponentialLR(optimizer, gamma=calc_gamma(lr1, lr2, restart_steps),
                                         steps_to_restart=restart_steps, initial_lr=lr1)
        scheduler = SequentialLR(schedulers=[scheduler1, scheduler2], milestones=[warmup_steps+1])
        for param_group in optimizer.param_groups: param_group['lr'] = lr0
        update_batchwise = True
    elif name == LRSchedulerNames.ExponentialWarmUpPlusCyclicDecay:
        assert warmup_and_decay_args is not None
        logger.info(f'Using {name} scheduler: {warmup_and_decay_args}')
        lr0, warmup_epochs, lr1, restart_epochs, lr2 = parse_warmup_and_decay_args_string(warmup_and_decay_args)
        logger.info(f'lr0 = {lr0}, warmup_epochs = {warmup_epochs}, lr1 = {lr1}, restart_epochs = {restart_epochs}, lr2 = {lr2}')
        scheduler1 = ExponentialLR(optimizer, gamma=calc_gamma(lr0, lr1, warmup_epochs))
        scheduler2 = CyclicExponentialLR(optimizer, gamma=calc_gamma(lr1, lr2, restart_epochs),
                                         steps_to_restart=restart_epochs, initial_lr=lr1)
        scheduler = SequentialLR(schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs+1])
        for param_group in optimizer.param_groups: param_group['lr'] = lr0
        update_batchwise = False
    elif name == LRSchedulerNames.ExponentialWarmUpPlusCosineAnnealing:
        assert warmup_and_cosine_args is not None
        assert n_batches_per_epoch is not None
        logger.info(f'Using {name} scheduler: {warmup_and_cosine_args}')
        lr0, warmup_epochs, lr1, restart_epochs, lr2 = parse_warmup_and_cosine_args_string(warmup_and_cosine_args)
        logger.info(f'lr0 = {lr0}, warmup_epochs = {warmup_epochs}, lr1 = {lr1}, restart_epochs = {restart_epochs}, lr2 = {lr2}')
        warmup_steps = warmup_epochs * n_batches_per_epoch
        restart_steps = warmup_epochs * n_batches_per_epoch
        scheduler1 = ExponentialLR(optimizer, gamma=calc_gamma(lr0, lr1, warmup_steps))
        scheduler2 = CosineAnnealingWarmRestarts(optimizer, lr1, restart_steps, eta_min=lr2)
        scheduler = SequentialLR(schedulers=[scheduler1, scheduler2], milestones=[warmup_steps+1])
        for param_group in optimizer.param_groups: param_group['lr'] = lr0
        update_batchwise = True
    elif name == LRSchedulerNames.ExponentialWarmUpPlusDecayPlusCyclicDecay:
        assert warmup_decay_and_cyclic_decay_args is not None
        logger.info(f'Using {name} scheduler: {warmup_decay_and_cyclic_decay_args}')
        lr0, epochs01, lr1, epochs12, lr2, lr3, restart_epochs, lr4 = \
            parse_warmup_decay_and_cyclic_decay_args_string(warmup_decay_and_cyclic_decay_args)
        logger.info(f'lr0 = {lr0}, epochs01 = {epochs01}, lr1 = {lr1}, epochs12 = {epochs12}, lr2 = {lr2}, '
                    f'lr3 = {lr3}, restart_epochs = {restart_epochs}, lr4 = {lr4}')
        scheduler1 = ExponentialLR(optimizer, gamma=calc_gamma(lr0, lr1, epochs01))
        scheduler2 = ExponentialLR(optimizer, gamma=calc_gamma(lr1, lr2, epochs12))
        scheduler3 = CyclicExponentialLR(optimizer, gamma=calc_gamma(lr3, lr4, restart_epochs-1),
                                            steps_to_restart=restart_epochs, initial_lr=lr3)
        scheduler = SequentialLR(schedulers=[scheduler1, scheduler2, scheduler3],
                                    milestones=[epochs01+1, epochs01+epochs12+1])
        for param_group in optimizer.param_groups: param_group['lr'] = lr0
        update_batchwise = False
    else:
        assert False, f'Unknown LR scheduler name: {name}'
    return scheduler, update_batchwise
