import torch

def compute_balanced_segmentation_loss(pred_mask, gt_mask):
        gt_nonzero = gt_mask > 0
        nonzero_count = torch.sum(gt_nonzero).item()
        gt_zeros = ~gt_nonzero
        zero_count = torch.sum(gt_zeros).item()

        error = torch.abs(pred_mask - gt_mask)
        if nonzero_count > 0:
            nonzero_error = torch.sum(error[gt_nonzero]) / nonzero_count
        else:
            nonzero_error = 0
        if zero_count > 0:
            zero_error = torch.sum(error[gt_zeros]) / zero_count
        else:
            zero_error = 0
        
        # print("nonzero_count: ", nonzero_count)
        # print("zero_count: ", zero_count)
        # print("nonzero_error: ", nonzero_error)
        # print("zero_error: ", zero_error)

        return nonzero_error + zero_error

