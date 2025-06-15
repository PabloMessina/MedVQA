from multiprocessing import Pool
from medvqa.metrics.bbox.utils import compute_bbox_union_iou, compute_iou
from medvqa.metrics.condition_aware_metric import ConditionAwareMetric
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric

class DatasetAwareBboxIOU(DatasetAwareMetric):

    def __init__(self, output_transform, allowed_dataset_ids, use_detectron2=False, use_yolov8=False, for_vinbig=False):
        self._acc_score = 0
        self._count = 0
        self._use_detectron2 = use_detectron2
        self._use_yolov8 = use_yolov8
        self._for_vinbig = for_vinbig
        if self._for_vinbig:
            assert self._use_yolov8
        super().__init__(output_transform, allowed_dataset_ids)
    
    def reset(self):
        self._acc_score = 0
        self._count = 0

    def update(self, output):
        if self._for_vinbig:
            yolov8_predictions, gt_coords, gt_classes = output
            n = len(gt_classes)
            for i in range(n):
                for j in range(len(yolov8_predictions[i])):
                    cls = yolov8_predictions[i][j, 5].int().item()
                    max_iou = 0
                    for k in range(len(gt_coords[i])):
                        if gt_classes[i][k] == cls:
                            iou = compute_iou(yolov8_predictions[i][j, :4], gt_coords[i][k])
                            if iou > max_iou:
                                max_iou = iou
                    self._acc_score += max_iou
                    self._count += 1
        else:
            if self._use_detectron2:
                pred_boxes, pred_classes, scores, gt_coords, gt_presence = output
                n = len(gt_presence)
                for i in range(n):
                    for j in range(len(pred_boxes[i])):
                        if scores[i][j] < 0.5:
                            continue
                        cls = pred_classes[i][j].item()
                        if gt_presence[i][cls] == 1:
                            iou = compute_iou(pred_boxes[i][j], gt_coords[i][cls])
                            self._acc_score += iou
                            self._count += 1
            elif self._use_yolov8:
                yolov8_predictions, gt_coords, gt_presence = output
                n = len(gt_presence)
                for i in range(n):
                    for j in range(len(yolov8_predictions[i])):
                        cls = yolov8_predictions[i][j, 5].int().item()
                        if cls < len(gt_presence[i]) and gt_presence[i][cls] == 1:
                            iou = compute_iou(yolov8_predictions[i][j, :4], gt_coords[i][cls])
                            self._acc_score += iou
                            self._count += 1
            else:
                pred_coords, gt_coords, gt_presence = output
                n, m = gt_presence.shape
                assert len(pred_coords.shape) == 3
                assert len(gt_coords.shape) == 3
                for i in range(n):
                    for j in range(0, m):
                        if gt_presence[i, j] == 1:
                            pred = pred_coords[i, j]
                            gt = gt_coords[i, j]
                            self._acc_score += compute_iou(pred, gt)
                            self._count += 1

    def compute(self):
        if self._count == 0:
            print('WARNING: Bbox IOU defaulting to 0 since self._count is 0')
            return 0
        return self._acc_score / self._count

class ConditionAwareBboxIOU(ConditionAwareMetric):

    def __init__(self, nc, output_transform, condition_function=lambda _: True, use_yolov8=False, for_vinbig=False, class_mask=None):
        super().__init__(output_transform, condition_function)
        self._acc_score = 0
        self._count = 0
        self.nc = nc
        self._use_yolov8 = use_yolov8
        self._for_vinbig = for_vinbig
        self._class_mask = class_mask

    def reset(self):
        self._acc_score = 0
        self._count = 0

    def update(self, output):
        if self._for_vinbig:
            if self._use_yolov8:
                yolov8_predictions, gt_coords, gt_classes = output
                bs = len(gt_classes) # Batch size
                for i in range(bs): # For each image in the batch
                    # Group ground truth coordinates by class
                    gt_coords_per_class = [[] for _ in range(self.nc)]
                    for cls, box in zip(gt_classes[i], gt_coords[i]):
                        gt_coords_per_class[cls].append(box)
                    # Group predictions by class
                    pred_coords_per_class = [[] for _ in range(self.nc)]
                    for j in range(len(yolov8_predictions[i])):
                        cls = yolov8_predictions[i][j, 5].int().item()
                        box = yolov8_predictions[i][j, :4]
                        pred_coords_per_class[cls].append(box)
                    # Compute IOU for each class present in the image
                    for cls in range(self.nc):
                        if len(gt_coords_per_class[cls]) > 0:
                            iou = compute_bbox_union_iou(pred_coords_per_class[cls], gt_coords_per_class[cls])
                            self._acc_score += iou
                            self._count += 1
            else:
                predicted_bboxes, gt_coords, gt_classes = output
                bs = len(gt_classes) # Batch size
                for i in range(bs): # For each image in the batch
                    # Group ground truth coordinates by class
                    gt_coords_per_class = [[] for _ in range(self.nc)]
                    for cls, box in zip(gt_classes[i], gt_coords[i]):
                        gt_coords_per_class[cls].append(box)
                    # Group predictions by class
                    pred_coords_per_class = [[] for _ in range(self.nc)]
                    coords = predicted_bboxes[i][0].detach()
                    classes = predicted_bboxes[i][2].detach()
                    for j in range(len(coords)):
                        cls = classes[j].item()
                        box = coords[j]
                        pred_coords_per_class[cls].append(box)
                    # Compute IOU for each class present in the image
                    for cls in range(self.nc):
                        if len(gt_coords_per_class[cls]) > 0:
                            iou = compute_bbox_union_iou(pred_coords_per_class[cls], gt_coords_per_class[cls])
                            self._acc_score += iou
                            self._count += 1
        else:
            if self._use_yolov8:
                yolov8_predictions, gt_coords, gt_presence = output
                n = len(gt_presence)
                for i in range(n):
                    for j in range(len(yolov8_predictions[i])):
                        cls = yolov8_predictions[i][j, 5].int().item()
                        if self._class_mask is not None and self._class_mask[cls] == 0:
                            continue # Skip if class is masked
                        if gt_presence[i][cls] == 1:
                            iou = compute_iou(yolov8_predictions[i][j, :4], gt_coords[i, cls])
                        else:
                            iou = 0
                        self._acc_score += iou
                        self._count += 1
            else:
                predicted_bboxes, gt_coords, gt_presence = output
                bs = len(gt_presence)
                n = len(gt_presence[0])
                assert len(predicted_bboxes) == bs
                assert len(gt_coords) == bs
                # print(f'DEBUG: ConditionAwareBboxIOU.update(): bs={bs}, n={n}')
                for i in range(bs): # For each image in the batch
                    for j in range(n): # For each class
                        if self._class_mask is not None and self._class_mask[j] == 0:
                            continue # Skip if class is masked
                        if gt_presence[i][j] == 1:
                            if predicted_bboxes[i][j] is None:
                                iou = 0
                            else:
                                # print(f'len(predicted_bboxes[i][j][0])={len(predicted_bboxes[i][j][0])}, len(gt_coords[i][j])={len(gt_coords[i][j])}')
                                iou = compute_bbox_union_iou(predicted_bboxes[i][j][0], gt_coords[i][j])
                            self._acc_score += iou
                            self._count += 1
                        elif predicted_bboxes[i][j] is not None:
                            self._acc_score += 0
                            self._count += 1

    def compute(self):
        if self._count == 0:
            print('WARNING: Bbox IOU defaulting to 0 since self._count is 0')
            return 0
        return self._acc_score / self._count
    
class ConditionAwareBboxIOUClassAgnostic(ConditionAwareMetric):

    def __init__(self, output_transform, bbox_format, condition_function=lambda _: True):
        super().__init__(output_transform, condition_function)
        self._acc_score = 0
        self._count = 0
        self._bbox_format = bbox_format

    def reset(self):
        self._acc_score = 0
        self._count = 0

    def update(self, output):
        predicted_bboxes, gt_bboxes = output
        bs = len(gt_bboxes) # Batch size
        for i in range(bs): # For each image in the batch
            if len(gt_bboxes[i]) > 0:
                iou = compute_bbox_union_iou(predicted_bboxes[i][0], gt_bboxes[i], bbox_format=self._bbox_format)
                self._acc_score += iou
                self._count += 1

    def compute(self):
        if self._count == 0:
            print('WARNING: Bbox IOU defaulting to 0 since self._count is 0')
            return 0
        return self._acc_score / self._count
    

class ConditionAwareBboxIOUperClass(ConditionAwareMetric):

    def __init__(self, nc, output_transform, condition_function=lambda _: True,
                 use_yolov8=False, use_fact_conditioned_yolo=False, for_vinbig=False, class_mask=None):
        super().__init__(output_transform, condition_function)
        self._acc_score = [0] * nc
        self._count = [0] * nc
        self.nc = nc
        self._use_yolov8 = use_yolov8
        self._use_fact_conditioned_yolo = use_fact_conditioned_yolo
        self._for_vinbig = for_vinbig
        self._class_mask = class_mask

    def reset(self):
        self._acc_score = [0] * self.nc
        self._count = [0] * self.nc

    def update(self, output):
        if self._for_vinbig:
            if self._use_yolov8:
                yolov8_predictions, gt_coords, gt_classes = output
                bs = len(gt_classes) # Batch size
                for i in range(bs): # For each image in the batch
                    # Group ground truth coordinates by class
                    gt_coords_per_class = [[] for _ in range(self.nc)]
                    for cls, box in zip(gt_classes[i], gt_coords[i]):
                        try:
                            gt_coords_per_class[cls].append(box)
                        except IndexError:
                            print(f'IndexError: cls={cls}, gt_classes[i]={gt_classes[i]}, gt_coords[i]={gt_coords[i]}, len(gt_coords_per_class)={len(gt_coords_per_class)}')
                            raise
                    # Group predictions by class
                    pred_coords_per_class = [[] for _ in range(self.nc)]
                    for j in range(len(yolov8_predictions[i])):
                        cls = yolov8_predictions[i][j, 5].int().item()
                        box = yolov8_predictions[i][j, :4]
                        pred_coords_per_class[cls].append(box)
                    # Compute IOU for each class present in the image
                    for cls in range(self.nc):
                        if len(gt_coords_per_class[cls]) > 0:
                            iou = compute_bbox_union_iou(pred_coords_per_class[cls], gt_coords_per_class[cls])
                            self._acc_score[cls] += iou
                            self._count[cls] += 1
            elif self._use_fact_conditioned_yolo:
                yolo_predictions, gt_coords, gt_classes = output
                assert isinstance(yolo_predictions, list)
                bs = len(gt_classes) # Batch size
                for i in range(bs): # For each image in the batch
                    assert isinstance(yolo_predictions[i], list)
                    assert len(yolo_predictions[i]) == self.nc, f'len(yolo_predictions[i])={len(yolo_predictions[i])}, self.nc={self.nc}' # One prediction per class
                    # Group ground truth coordinates by class
                    gt_coords_per_class = [[] for _ in range(self.nc)]
                    for cls, box in zip(gt_classes[i], gt_coords[i]):
                        gt_coords_per_class[cls].append(box)
                    # Compute IOU for each class present in the image
                    for cls in range(self.nc):
                        if len(gt_coords_per_class[cls]) > 0:
                            pred_coords = yolo_predictions[i][cls][:, :4] # Extract coordinates
                            iou = compute_bbox_union_iou(pred_coords, gt_coords_per_class[cls])
                            self._acc_score[cls] += iou
                            self._count[cls] += 1
            else:
                predicted_bboxes, gt_coords, gt_classes = output
                bs = len(gt_classes) # Batch size
                for i in range(bs): # For each image in the batch
                    # Group ground truth coordinates by class
                    gt_coords_per_class = [[] for _ in range(self.nc)]
                    for cls, box in zip(gt_classes[i], gt_coords[i]):
                        gt_coords_per_class[cls].append(box)
                    # Group predictions by class
                    pred_coords_per_class = [[] for _ in range(self.nc)]
                    coords = predicted_bboxes[i][0].detach()
                    classes = predicted_bboxes[i][2].detach()
                    for j in range(len(coords)):
                        cls = classes[j].item()
                        box = coords[j]
                        pred_coords_per_class[cls].append(box)
                    # Compute IOU for each class present in the image
                    for cls in range(self.nc):
                        if len(gt_coords_per_class[cls]) > 0:
                            iou = compute_bbox_union_iou(pred_coords_per_class[cls], gt_coords_per_class[cls])
                            self._acc_score[cls] += iou
                            self._count[cls] += 1
        else:
            if self._use_yolov8:
                yolov8_predictions, gt_coords, gt_presence = output
                
                n = len(gt_presence)
                for i in range(n): # For each image in the batch
                    # Group ground truth coordinates by class
                    gt_coords_per_class = [[] for _ in range(self.nc)]
                    for cls in range(self.nc):
                        if gt_presence[i, cls] == 1:
                            gt_coords_per_class[cls].append(gt_coords[i, cls])
                    # Group predictions by class
                    pred_coords_per_class = [[] for _ in range(self.nc)]
                    for j in range(len(yolov8_predictions[i])):
                        cls = yolov8_predictions[i][j, 5].int().item()
                        box = yolov8_predictions[i][j, :4]
                        pred_coords_per_class[cls].append(box)
                    # Compute IOU for each class present in the image
                    for cls in range(self.nc):
                        if self._class_mask is not None and self._class_mask[cls] == 0:
                            continue # Skip if class is masked
                        if len(gt_coords_per_class[cls]) > 0:
                            iou = compute_bbox_union_iou(pred_coords_per_class[cls], gt_coords_per_class[cls])
                            self._acc_score[cls] += iou
                            self._count[cls] += 1
            else:
                raise NotImplementedError('TODO: Implement self._use_yolov8=False')
                predicted_bboxes, gt_coords, gt_presence = output
                bs = len(gt_presence)
                n = len(gt_presence[0])
                assert len(predicted_bboxes) == bs
                assert len(gt_coords) == bs
                for i in range(bs): # For each image in the batch
                    for cls in range(n): # For each class
                        if self._class_mask is not None and self._class_mask[cls] == 0:
                            continue # Skip if class is masked
                        if gt_presence[i][cls] == 1:
                            if predicted_bboxes[i][cls] is None:
                                iou = 0
                            else:
                                iou = compute_bbox_union_iou(predicted_bboxes[i][cls][0], gt_coords[i][cls])
                            self._acc_score[cls] += iou
                            self._count[cls] += 1
                        elif predicted_bboxes[i][cls] is not None:
                            self._acc_score[cls] += 0
                            self._count[cls] += 1

    def compute(self):
        return [self._acc_score[i] / self._count[i] if self._count[i] > 0 else 0 for i in range(self.nc)]
    

def _compute_iou_for_sample(args):
    """
    Computes the sum of IoUs and the count of classes for a single sample.
    This function is designed to be called by a multiprocessing Pool.
    """
    predicted_bboxes, gt_coords, gt_classes = args
    acc_score = 0
    count = 0

    # Obtain number of classes for this sample
    if not gt_classes:
        return 0, 0
    nc = max(gt_classes) + 1

    # Group ground truth coordinates by class
    gt_coords_per_class = [[] for _ in range(nc)]
    for cls, box in zip(gt_classes, gt_coords):
        gt_coords_per_class[cls].append(box)

    # Group predictions by class
    pred_coords_per_class = [[] for _ in range(nc)]
    coords = predicted_bboxes[0]
    classes = predicted_bboxes[1]
    for j in range(len(coords)):
        cls = classes[j]
        if cls < nc:  # Skip classes not present in the ground truth
            box = coords[j]
            pred_coords_per_class[cls].append(box)

    # Compute IOU for each class present in the image
    for cls in range(nc):
        if len(gt_coords_per_class[cls]) > 0:
            iou = compute_bbox_union_iou(
                pred_coords_per_class[cls], gt_coords_per_class[cls]
            )
            acc_score += iou
            count += 1

    return acc_score, count

class ConditionAwareBboxIOUOpenClass(ConditionAwareMetric):
    def __init__(self, output_transform, num_processes=8, condition_function=lambda _: True):
        super().__init__(output_transform, condition_function)
        self._predictions = []
        self._ground_truths = []
        self.num_processes = num_processes

    def reset(self):
        self._predictions = []
        self._ground_truths = []

    def update(self, output):
        """
        Accumulates predictions and ground truths. This is a very fast operation.
        """
        predicted_bboxes, gt_coords, gt_classes = output
        # predicted_bboxes: List of tuples (coordinates, scores, classes)
        # gt_coords: List of lists of ground truth coordinates
        # gt_classes: List of lists of ground truth classes

        # Move data to CPU and convert to basic types to free GPU and prepare
        # for multiprocessing (which requires picklable objects).
        for i in range(len(gt_classes)):
            # Detach tensors, move to CPU, and convert to lists
            coords = predicted_bboxes[i][0].detach().cpu().tolist()
            classes = predicted_bboxes[i][2].detach().cpu().tolist()

            self._predictions.append((coords, classes))
            self._ground_truths.append((gt_coords[i], gt_classes[i]))

    def compute(self):
        """
        Computes the metric using multiprocessing to parallelize the work.
        """
        if not self._predictions:
            print(
                "WARNING: Bbox IOU defaulting to 0 since no data was provided."
            )
            return 0

        # Prepare arguments for each process
        num_samples = len(self._predictions)
        args_list = [
            (self._predictions[i], self._ground_truths[i][0], self._ground_truths[i][1])
            for i in range(num_samples)
        ]

        total_acc_score = 0
        total_count = 0

        # Use a multiprocessing pool to compute the IoU for all samples in parallel
        # The `with` statement ensures the pool is properly closed.
        with Pool(processes=self.num_processes) as pool:
            results = pool.map(_compute_iou_for_sample, args_list)

        # Aggregate results from all processes
        for acc_score, count in results:
            total_acc_score += acc_score
            total_count += count

        if total_count == 0:
            print(
                "WARNING: Bbox IOU defaulting to 0 since total_count is 0."
            )
            return 0

        return total_acc_score / total_count