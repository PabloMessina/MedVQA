import torch
from torch import nn
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.loss import BboxLoss
from ultralytics.yolo.utils.ops import xywh2xyxy

class YOLOV8MultiDetectionLayersLoss:
    """
    YOLOV8 loss function for multiple detection layers.
    This is a modified version of YOLOv8 original loss function.
    """

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.n_dect_layers = len(model.detection_layers)  # number of detection layers
        self.stride = [m.stride for m in model.detection_layers] # model strides
        self.nc = [m.nc for m in model.detection_layers] # number of classes
        self.no = [m.no for m in model.detection_layers] # number of outputs
        self.reg_max = [m.reg_max for m in model.detection_layers] # max number of regression targets
        self.device = device

        self.use_dfl = [m.reg_max > 1 for m in model.detection_layers] # use distance-based focal loss

        self.assigner = [
            TaskAlignedAssigner(topk=10, num_classes=self.nc[i], alpha=0.5, beta=6.0) \
            for i in range(self.n_dect_layers)
        ]
        self.bbox_loss = [
            BboxLoss(self.reg_max[i] - 1, use_dfl=self.use_dfl[i]).to(device) \
            for i in range(self.n_dect_layers)
        ]
        self.proj = [
            torch.arange(self.reg_max[i], dtype=torch.float, device=device) \
            for i in range(self.n_dect_layers)
        ]

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist, dect_layer_idx):
        if self.use_dfl[dect_layer_idx]:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj[dect_layer_idx].type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, dect_layer_idx):

        # print(f'From YOLOV8MultiDetectionLayersLoss: dect_layer_idx = {dect_layer_idx}')
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no[dect_layer_idx], -1) for xi in feats], 2).split(
            (self.reg_max[dect_layer_idx] * 4, self.nc[dect_layer_idx]), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[dect_layer_idx][0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride[dect_layer_idx], 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, dect_layer_idx)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner[dect_layer_idx](
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss[dect_layer_idx](pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)