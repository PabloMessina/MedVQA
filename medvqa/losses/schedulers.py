from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, _LRScheduler
from bisect import bisect_right

class LRSchedulerNames:
    ReduceLROnPlateau = 'reduce-lr-on-plateau'
    WarmUpPlusDecay = 'warmup+decay'

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

def calc_gamma(lr_a, lr_b, steps):
    return (lr_b / lr_a) ** (1/steps)

def parse_warmup_and_decay_args_string(string):
    args = string.split(',')
    assert len(args) == 5
    return float(args[0]), int(args[1]), float(args[2]), int(args[3]), float(args[4])

def create_lr_scheduler(name, optimizer, factor=None, patience=None, warmup_and_decay_args=None):
    if name == LRSchedulerNames.ReduceLROnPlateau:
        assert factor is not None
        assert patience is not None
        scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True,
                                        factor=factor, patience=patience)
    elif name == LRSchedulerNames.WarmUpPlusDecay:
        assert warmup_and_decay_args is not None
        print(f'Using {name} scheduler: {warmup_and_decay_args}')
        lr0, steps01, lr1, steps12, lr2 = parse_warmup_and_decay_args_string(warmup_and_decay_args)
        print(lr0, steps01, lr1, steps12, lr2)
        scheduler1 = ExponentialLR(optimizer, gamma=calc_gamma(lr0, lr1, steps01), verbose=True)
        scheduler2 = ExponentialLR(optimizer, gamma=calc_gamma(lr1, lr2, steps12), verbose=True)
        scheduler = SequentialLR(schedulers=[scheduler1, scheduler2], milestones=[steps01+1])
    else:
        assert False, f'Unknown schduler name {name}'
    return scheduler
