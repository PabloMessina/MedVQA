import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss, TaskAlignedAssigner

class YOLOv11CustomLoss(v8DetectionLoss):

    def __init__(self, detect_module, hyp, device, tal_topk=10):  # model must be de-paralleled
        """This is a modified version of YOLOv8's original constructor in order to explicitly
        receive the detection module, which is useful if one wants to have multiple detection modules
        for multiple tasks."""

        print('From YOLOv11CustomLoss:')
        print(f'device: {device}')
        
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = hyp # hyperparameters
        self.stride = detect_module.stride  # model strides
        self.nc = detect_module.nc  # number of classes
        self.no = detect_module.nc + detect_module.reg_max * 4
        self.reg_max = detect_module.reg_max
        self.device = device

        self.use_dfl = detect_module.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(detect_module.reg_max).to(device)
        self.proj = torch.arange(detect_module.reg_max, dtype=torch.float, device=device)