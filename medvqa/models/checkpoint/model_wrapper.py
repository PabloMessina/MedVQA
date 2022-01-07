import torch
from ignite.handlers import Checkpoint

class EpochWrapper:
    def __init__(self, epoch):
        self.data = dict(epoch = epoch)
    
    def state_dict(self):
        return self.data

    def load_state_dict(self, data):
        self.data = data


class ModelWrapper:
    def __init__(self, model, optimizer=None, lr_scheduler=None, epoch=0):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch_wrapper = EpochWrapper(epoch)

    def get_epoch(self):
        return self.epoch_wrapper.data['epoch']

    def set_epoch(self, epoch):
        self.epoch_wrapper.data['epoch'] = epoch
    
    def to_save(self):
        return dict(
            model = self.model,
            optimizer = self.optimizer,
            lr_scheduler = self.lr_scheduler,
            epoch = self.epoch_wrapper,
        )

    def to_load(self, model_only=False):
        if model_only:
            print('Loading model and epoch only')
            return dict(
                model = self.model,
                epoch = self.epoch_wrapper,
            )
        return dict(
            model = self.model,
            optimizer = self.optimizer,
            lr_scheduler = self.lr_scheduler,
            epoch = self.epoch_wrapper,
        )

    def load_checkpoint(self, checkpoint_path, device, model_only=False):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        Checkpoint.load_objects(self.to_load(model_only), checkpoint)
        print('Checkpoint successfully loaded!')