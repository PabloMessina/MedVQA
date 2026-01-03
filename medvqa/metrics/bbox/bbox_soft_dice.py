from typing import Callable, Tuple
from medvqa.metrics.condition_aware_metric import ConditionAwareMetric
from medvqa.utils.bbox_utils import convert_bboxes_into_presence_map, cxcywh_to_xyxy
from medvqa.utils.metrics_utils import calculate_soft_dice


class ConditionAwareBboxSoftDice(ConditionAwareMetric):
    """
    Calculates the average soft Dice score between predicted probability maps
    and ground truth bounding boxes, conditioned on a given function.

    This metric converts ground truth bounding boxes into a binary mask and
    then computes the soft Dice score against the model's predicted probability map.
    """

    def __init__(
        self,
        output_transform: Callable,
        output_len: int,
        H: int,
        W: int,
        bbox_format: str,
        mask_resolution: Tuple[int, int] = (100, 100),
        condition_function: Callable = lambda _: True,
    ):
        """
        Initializes the ConditionAwareBboxSoftDice metric.

        Args:
            output_transform: A callable that transforms the output of the model.
            output_len: Number of outputs to unpack from the engine output.
            H: Height of the probability maps output by the model.
            W: Width of the probability maps output by the model.
            bbox_format: Format of the bounding boxes ('xyxy' or 'cxcywh').
            mask_resolution: Resolution of the mask to compute the Dice score.
            condition_function: A callable that checks if the metric should be computed.
        """
        assert output_len in [2, 3], "output_len must be 2 or 3."
        assert bbox_format in [
            "xyxy",
            "cxcywh",
        ], "bbox_format must be 'xyxy' or 'cxcywh'."

        super().__init__(output_transform, condition_function)
        self._predictions = []
        self._ground_truths = []
        self.H = H
        self.W = W
        self.output_len = output_len
        self.bbox_format = bbox_format
        self.mask_resolution = mask_resolution

    def reset(self):
        self._predictions = []
        self._ground_truths = []

    def update(self, output):
        """
        Accumulates predictions and ground truths from the engine's output.
        This logic is identical to ConditionAwareBboxCNR.
        """
        assert (
            len(output) == self.output_len
        ), f"Expected {self.output_len} outputs, but got {len(output)}."

        if self.output_len == 2:
            prob_maps, gt_coords = output
            assert prob_maps.ndim == 2  # (B, H*W)
            B = prob_maps.shape[0]
            prob_maps = prob_maps.view(B, self.H, self.W)
            prob_maps = prob_maps.detach().cpu().numpy()
            for i in range(B):
                self._predictions.append(prob_maps[i])
                self._ground_truths.append(gt_coords[i])
        elif self.output_len == 3:
            prob_maps, gt_coords, gt_classes = output
            assert prob_maps.ndim == 3  # (B, C, H*W)
            B, C = prob_maps.shape[:2]
            prob_maps = prob_maps.view(B, C, self.H, self.W)
            prob_maps = prob_maps.detach().cpu().numpy()
            for i in range(B):
                class_bboxes = [[] for _ in range(C)]
                for bbox, cls in zip(gt_coords[i], gt_classes[i]):
                    class_bboxes[cls].append(bbox)
                for j in range(C):
                    if len(class_bboxes[j]) > 0:
                        self._predictions.append(prob_maps[i, j])
                        self._ground_truths.append(class_bboxes[j])
        else:
            raise ValueError("output_len must be 2 or 3.")

    def compute(self):
        """
        Computes the average soft Dice score over all accumulated samples.
        """
        if len(self._predictions) == 0:
            raise ValueError(
                "No predictions or ground truths to compute Soft Dice."
            )

        total_soft_dice = 0.0
        for prob_map, gt_bboxes in zip(self._predictions, self._ground_truths):
            # Convert bboxes to 'xyxy' if needed, then create the binary mask
            if self.bbox_format == "cxcywh":
                gt_bboxes_xyxy = [cxcywh_to_xyxy(bbox) for bbox in gt_bboxes]
            else:
                gt_bboxes_xyxy = gt_bboxes
            mask = convert_bboxes_into_presence_map(
                gt_bboxes_xyxy, self.mask_resolution
            )

            # Calculate the soft Dice score for the current sample
            soft_dice = calculate_soft_dice(mask, prob_map)
            total_soft_dice += soft_dice

        average_soft_dice = total_soft_dice / len(self._predictions)
        return average_soft_dice