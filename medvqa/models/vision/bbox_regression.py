import math
import torch
import torch.nn as nn
from torchvision.ops import roi_align
import torchvision.ops as ops
from medvqa.metrics.bbox.utils import cxcywh_to_xyxy_tensor, get_grid_centers
import logging

logger = logging.getLogger(__name__)

class BBoxRegressorVersion:
    V1 = 'v1'
    V2 = 'v2'
    V3 = 'v3'
    V4 = 'v4'
    V4_1 = 'v4_1'
    V5 = 'v5'
    V6 = 'v6'

class BoundingBoxRegressor_v1(nn.Module):

    def __init__(self, local_feat_dim, global_feat_dim, hidden_dim, num_classes, train_average_bbox_coords):
        super().__init__()
        logger.info('BoundingBoxRegressor_v1:')
        logger.info(f'  local_feat_dim: {local_feat_dim}')
        logger.info(f'  global_feat_dim: {global_feat_dim}')
        logger.info(f'  hidden_dim: {hidden_dim}')
        logger.info(f'  num_classes: {num_classes}')
        self.local_feat_dim = local_feat_dim
        self.global_feat_dim = global_feat_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.loc_proj = nn.Linear(local_feat_dim, num_classes * hidden_dim)
        self.glob_proj = nn.Linear(global_feat_dim, num_classes * hidden_dim)
        self.bbox_hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.bbox_regressor = nn.Linear(hidden_dim, 4)
        self.bbox_binary_classifier = nn.Linear(hidden_dim, 1)
        # create parameters for the average coordinates of the training set
        self.train_average_bbox_coords = nn.parameter.Parameter(torch.zeros(4 * num_classes), requires_grad=False)
        if type(train_average_bbox_coords) == list:
            train_average_bbox_coords = torch.tensor(train_average_bbox_coords)
        self.train_average_bbox_coords.data.copy_(train_average_bbox_coords)

    def forward(self, local_features, global_average_pool):
        # local_features: (batch_size, num_regions, local_feat_dim)
        # global_average_pool: (batch_size, global_feat_dim)
        # return: bbox_coords (batch_size, num_classes * 4)
        #         bbox_presence (batch_size, num_classes)

        batch_size = local_features.size(0)
        
        # 1) Project the global feature to a different vector per class
        xg = self.glob_proj(global_average_pool) # (batch_size, num_classes * hidden_dim)
        xg = xg.reshape(xg.size(0), -1, self.hidden_dim) # (batch_size, num_classes, hidden_dim)
        xg = xg.unsqueeze(2) # (batch_size, num_classes, 1, hidden_dim)
        # assert xg.shape == (batch_size, self.num_classes, 1, self.hidden_dim)

        # 2) Project the local features to a different tensor for each class
        xl = self.loc_proj(local_features) # (batch_size, num_regions, num_classes * hidden_dim)
        xl = xl.reshape(xl.size(0), xl.size(1), self.num_classes, self.hidden_dim) # (batch_size, num_regions, num_classes, hidden_dim)
        xl = xl.permute(0, 2, 1, 3) # (batch_size, num_classes, num_regions, hidden_dim)
        # assert xl.shape == (batch_size, self.num_classes, local_features.size(1), self.hidden_dim)
        
        # 3) Compute the attention weights using dot product between projected global feature and local features        
        dot_prod = torch.sum(xg * xl, dim=-1) # (batch_size, num_classes, num_regions)        
        attn_weights = torch.softmax(dot_prod, dim=-1) # (batch_size, num_classes, num_regions)
        attn_weights = attn_weights.unsqueeze(-1) # (batch_size, num_classes, num_regions, 1)

        # 4) Compute the weighted sum of projected local features for each class
        weighted_sum = torch.sum(attn_weights * xl, dim=2) # (batch_size, num_classes, hidden_dim)

        # 5) Compute the bounding box coordinates and presence for each class
        bbox = self.bbox_hidden_layer(weighted_sum) # (batch_size, num_classes, hidden_dim)
        bbox = torch.relu(bbox)
        bbox_coords = self.bbox_regressor(bbox) # (batch_size, num_classes, 4)
        bbox_coords = bbox_coords + self.train_average_bbox_coords.view(1, -1, 4) # (batch_size, num_classes, 4)
        bbox_presence = self.bbox_binary_classifier(bbox) # (batch_size, num_classes, 1)
        # assert bbox_coords.shape == (batch_size, self.num_classes, 4)
        # assert bbox_presence.shape == (batch_size, self.num_classes, 1)
        
        # 6) Reshape the bounding box coordinates and presence and return
        bbox_coords = bbox_coords.view(batch_size, -1) # (batch_size, num_classes * 4)
        bbox_presence = bbox_presence.squeeze(-1) # (batch_size, num_classes)
        return bbox_coords, bbox_presence

class BoundingBoxRegressor_v2(nn.Module):

    def __init__(self, local_feat_dim, global_feat_dim, hidden_dim, num_classes, num_regions,
                train_average_bbox_coords):
        super().__init__()
        logger.info('BoundingBoxRegressor_v2:')
        logger.info(f'  local_feat_dim: {local_feat_dim}')
        logger.info(f'  global_feat_dim: {global_feat_dim}')
        logger.info(f'  hidden_dim: {hidden_dim}')
        logger.info(f'  num_classes: {num_classes}')
        logger.info(f'  num_regions: {num_regions}')
        self.local_feat_dim = local_feat_dim
        self.global_feat_dim = global_feat_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.loc_projs = nn.ModuleList([nn.Linear(local_feat_dim, hidden_dim) for _ in range(num_classes)])
        self.glob_projs = nn.ModuleList([nn.Linear(global_feat_dim, hidden_dim) for _ in range(num_classes)])
        self.bbox_coords_fc = nn.ModuleList([nn.Linear((num_regions + 1) * hidden_dim, 4) for _ in range(num_classes)]) # +1 for global feature
        self.bbox_presence_fc = nn.ModuleList([nn.Linear((num_regions + 1) * hidden_dim, 1) for _ in range(num_classes)]) # +1 for global feature
        # create parameters for the average coordinates of the training set
        self.train_average_bbox_coords = nn.parameter.Parameter(torch.zeros(4 * num_classes), requires_grad=False)
        if type(train_average_bbox_coords) == list:
            train_average_bbox_coords = torch.tensor(train_average_bbox_coords)
        self.train_average_bbox_coords.data.copy_(train_average_bbox_coords)

    def forward(self, local_features, global_features):
        # local_features: (batch_size, num_regions, local_feat_dim)
        # global_features: (batch_size, global_feat_dim)
        # return: bbox_coords (batch_size, num_classes * 4)
        #         bbox_presence (batch_size, num_classes)

        batch_size = local_features.size(0)
        pred_bbox_coords = []
        pred_bbox_presence = []
        
        for i in range(self.num_classes):
            # 1) Project global features
            xg = self.glob_projs[i](global_features) # (batch_size, hidden_dim)
            xg = xg.unsqueeze(1) # (batch_size, 1, hidden_dim)
            # 2) Project local features
            xl = self.loc_projs[i](local_features) # (batch_size, num_regions, hidden_dim)
            # 3) Concatenate global and local features
            x = torch.cat([xg, xl], dim=1) # (batch_size, num_regions + 1, hidden_dim)
            x = x.view(batch_size, -1) # (batch_size, (num_regions + 1) * hidden_dim)
            x = torch.relu(x) # (batch_size, (num_regions + 1) * hidden_dim) 
            # 4) Compute the bounding box coordinates and presence
            bbox_coords = self.bbox_coords_fc[i](x) # (batch_size, 4)
            bbox_presence = self.bbox_presence_fc[i](x) # (batch_size, 1)
            pred_bbox_coords.append(bbox_coords)
            pred_bbox_presence.append(bbox_presence)

        # 5) Concatenate the bounding box coordinates and presence for all classes
        pred_bbox_coords = torch.stack(pred_bbox_coords, dim=1) # (batch_size, num_classes, 4)
        pred_bbox_coords = pred_bbox_coords.view(batch_size, -1) # (batch_size, num_classes * 4)
        pred_bbox_coords = pred_bbox_coords + self.train_average_bbox_coords.view(1, -1) # (batch_size, num_classes * 4)
        pred_bbox_presence = torch.stack(pred_bbox_presence, dim=1) # (batch_size, num_classes, 1)
        pred_bbox_presence = pred_bbox_presence.squeeze(-1) # (batch_size, num_classes)
        return pred_bbox_coords, pred_bbox_presence

class BoundingBoxRegressor_v3(nn.Module):

    def __init__(self, local_feat_dim, global_feat_dim, hidden_dim, num_classes, num_regions,
                train_average_bbox_coords):
        super().__init__()
        logger.info('BoundingBoxRegressor_v3:')
        logger.info(f'  local_feat_dim: {local_feat_dim}')
        logger.info(f'  global_feat_dim: {global_feat_dim}')
        logger.info(f'  hidden_dim: {hidden_dim}')
        logger.info(f'  num_classes: {num_classes}')
        logger.info(f'  num_regions: {num_regions}')
        self.local_feat_dim = local_feat_dim
        self.global_feat_dim = global_feat_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.loc_proj = nn.Linear(local_feat_dim, hidden_dim)
        self.glob_proj = nn.Linear(global_feat_dim, hidden_dim)
        self.midde_layer = nn.Linear((num_regions + 1) * hidden_dim, hidden_dim)
        self.bbox_coords_fc = nn.Linear(hidden_dim, 4 * num_classes)
        self.bbox_presence_fc = nn.Linear(hidden_dim, num_classes)
        # create parameters for the average coordinates of the training set
        self.train_average_bbox_coords = nn.parameter.Parameter(torch.zeros(4 * num_classes), requires_grad=False)
        if type(train_average_bbox_coords) == list:
            train_average_bbox_coords = torch.tensor(train_average_bbox_coords)
        self.train_average_bbox_coords.data.copy_(train_average_bbox_coords)

    def forward(self, local_features, global_average_pool):
        # local_features: (batch_size, num_regions, local_feat_dim)
        # global_average_pool: (batch_size, global_feat_dim)
        # return: bbox_coords (batch_size, num_classes * 4)
        #         bbox_presence (batch_size, num_classes)

        batch_size = local_features.size(0)
        # 1) Project global features
        xg = self.glob_proj(global_average_pool) # (batch_size, hidden_dim)
        xg = xg.unsqueeze(1) # (batch_size, 1, hidden_dim)
        # 2) Project local features
        xl = self.loc_proj(local_features) # (batch_size, num_regions, hidden_dim)
        # 3) Concatenate global and local features
        x = torch.cat([xg, xl], dim=1) # (batch_size, num_regions + 1, hidden_dim)
        x = x.view(batch_size, -1) # (batch_size, (num_regions + 1) * hidden_dim)
        x = torch.relu(x) # (batch_size, (num_regions + 1) * hidden_dim)
        x = self.midde_layer(x) # (batch_size, hidden_dim)
        x = torch.relu(x) # (batch_size, hidden_dim)
        # 4) Compute the bounding box coordinates and presence
        bbox_coords = self.bbox_coords_fc(x) # (batch_size, 4 * num_classes)
        bbox_presence = self.bbox_presence_fc(x) # (batch_size, num_classes)
        bbox_coords = bbox_coords + self.train_average_bbox_coords.view(1, -1) # (batch_size, 4 * num_classes)
        return bbox_coords, bbox_presence

class BoundingBoxRegressorAndMultiLabelClassifier_v4(nn.Module):

    def __init__(self, local_feat_dim, global_feat_dim, hidden_dim, num_bboxes, num_bboxes_to_supervise,
                 num_regions, train_average_bbox_coords, bbox_to_labels, bbox_group_to_labels):
        super().__init__()
        logger.info('BoundingBoxRegressorAndMultiLabelClassifier_v4:')
        logger.info(f'  local_feat_dim: {local_feat_dim}')
        logger.info(f'  global_feat_dim: {global_feat_dim}')
        logger.info(f'  hidden_dim: {hidden_dim}')
        logger.info(f'  num_bboxes: {num_bboxes}')
        logger.info(f'  num_regions: {num_regions}')
        self.local_feat_dim = local_feat_dim
        self.global_feat_dim = global_feat_dim
        self.hidden_dim = hidden_dim
        self.num_bboxes = num_bboxes
        self.num_bboxs_to_supervise = num_bboxes_to_supervise # because not all bboxes are supervised due to lack of annotations
        self.bbox_to_labels = bbox_to_labels
        self.bbox_group_to_labels = bbox_group_to_labels
        assert len(bbox_to_labels) <= num_bboxes
        assert type(bbox_to_labels) == list # list of [idx, labels]
        assert type(bbox_to_labels[0]) == list 
        assert type(bbox_group_to_labels) == list # list of [idxs, labels]
        assert type(bbox_group_to_labels[0]) == list
        assert num_bboxes_to_supervise <= num_bboxes
        # create projection layers for bbox prediction
        self.loc_projs = nn.ModuleList([nn.Linear(local_feat_dim, hidden_dim) for _ in range(num_bboxes)])
        self.glob_projs = nn.ModuleList([nn.Linear(global_feat_dim, hidden_dim) for _ in range(num_bboxes)])
        self.bbox_mid_layer = nn.ModuleList([nn.Linear((num_regions + 1) * hidden_dim, hidden_dim) for _ in range(num_bboxes)]) # +1 for global feature
        self.bbox_coords_fc = nn.ModuleList([nn.Linear(hidden_dim, 4) for _ in range(num_bboxes)]) 
        self.bbox_presence_fc = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_bboxes)])
        # create parameters for the average coordinates of the training set
        self.train_average_bbox_coords = nn.parameter.Parameter(torch.zeros(4 * num_bboxes_to_supervise), requires_grad=False)
        if type(train_average_bbox_coords) == list:
            train_average_bbox_coords = torch.tensor(train_average_bbox_coords)
        self.train_average_bbox_coords.data.copy_(train_average_bbox_coords)
        # create layers for the multi-label classification
        self.loc_mlc_fc = nn.ModuleList([nn.Linear(hidden_dim, len(labels)) for _, labels in bbox_to_labels])
        self.glob_mlc_fc = nn.ModuleList([nn.Linear(hidden_dim * len(bbox_group), len(labels)) \
                                            for bbox_group, labels in bbox_group_to_labels])

    def forward(self, local_features, global_average_pool):
        # local_features: (batch_size, num_regions, local_feat_dim)
        # global_average_pool: (batch_size, global_feat_dim)
        # return: bbox_coords (batch_size, num_bboxes * 4)
        #         bbox_presence (batch_size, num_bboxes)
        #         mlc_scores (batch_size, num_classes)

        batch_size = local_features.size(0)       
        
        # 1) Bbox feature extraction and prediction
        bbox_features = []
        pred_bbox_coords = []
        pred_bbox_presence = []
        for i in range(self.num_bboxes):
            # 1.1) Project global features
            xg = self.glob_projs[i](global_average_pool) # (batch_size, hidden_dim)
            xg = xg.unsqueeze(1) # (batch_size, 1, hidden_dim)
            # 1.2) Project local features
            xl = self.loc_projs[i](local_features) # (batch_size, num_regions, hidden_dim)
            # 1.3) Concatenate global and local features
            x = torch.cat([xg, xl], dim=1) # (batch_size, num_regions + 1, hidden_dim)
            x = x.view(batch_size, -1) # (batch_size, (num_regions + 1) * hidden_dim)
            x = torch.relu(x) # (batch_size, (num_regions + 1) * hidden_dim)
            # 1.4) Apply mid layer
            x = self.bbox_mid_layer[i](x) # (batch_size, hidden_dim)
            x = torch.relu(x) # (batch_size, hidden_dim)
            bbox_features.append(x)
            # 1.5) Compute the bounding box coordinates and presence
            # but only for the first num_bboxs_to_supervise bboxes
            if i < self.num_bboxs_to_supervise:
                bbox_coords = self.bbox_coords_fc[i](x) # (batch_size, 4)
                bbox_presence = self.bbox_presence_fc[i](x) # (batch_size, 1)
                pred_bbox_coords.append(bbox_coords)
                pred_bbox_presence.append(bbox_presence)
        # 1.6) Concatenate the bounding box coordinates and presence for all classes
        pred_bbox_coords = torch.stack(pred_bbox_coords, dim=1) # (batch_size, num_bboxs_to_supervise, 4)
        pred_bbox_coords = pred_bbox_coords.view(batch_size, -1) # (batch_size, num_bboxs_to_supervise * 4)
        pred_bbox_coords = pred_bbox_coords + self.train_average_bbox_coords.view(1, -1) # (batch_size, num_bboxs_to_supervise * 4)
        pred_bbox_presence = torch.stack(pred_bbox_presence, dim=1) # (batch_size, num_bboxs_to_supervise, 1)
        pred_bbox_presence = pred_bbox_presence.squeeze(-1) # (batch_size, num_bboxs_to_supervise)

        # 2) Multi-label classification
        mlc_scores = []
        # 2.1) Compute the multi-label classification for each bounding box
        for i, (idx, _) in enumerate(self.bbox_to_labels):
            loc_label_logits = self.loc_mlc_fc[i](bbox_features[idx]) # (batch_size, num_labels)
            mlc_scores.append(loc_label_logits)
        # 2.2) Compute the multi-label classification for each bounding box group
        for i, (bbox_group, _) in enumerate(self.bbox_group_to_labels):
            # 2.2.1) Get the features for the bounding boxes in the group
            group_features = [bbox_features[i] for i in bbox_group]
            group_features = torch.stack(group_features, dim=1) # (batch_size, num_bboxes, hidden_dim)
            group_features = group_features.view(batch_size, -1) # (batch_size, num_bboxes * hidden_dim)
            # 2.2.2) Apply the global multi-label classification layer
            glob_label_logits = self.glob_mlc_fc[i](group_features) # (batch_size, num_labels)
            mlc_scores.append(glob_label_logits)
        # 2.3) Concatenate the local and global logits
        mlc_scores = torch.concat(mlc_scores, dim=1) # (batch_size, (num_bboxes + num_bbox_groups) * num_labels)

        # 3) Return the predicted bounding box coordinates and presence, and the multi-label classification scores
        return pred_bbox_coords, pred_bbox_presence, mlc_scores
    
class BoundingBoxRegressorAndMultiLabelClassifier_v4_1(nn.Module):

    def __init__(self, local_feat_dim, hidden_dim, num_bboxes,  num_regions, bbox_to_labels, bbox_group_to_labels):
        super().__init__()
        logger.info('BoundingBoxRegressorAndMultiLabelClassifier_v4_1')
        logger.info(f'  local_feat_dim: {local_feat_dim}')
        logger.info(f'  hidden_dim: {hidden_dim}')
        logger.info(f'  num_bboxes: {num_bboxes}')
        logger.info(f'  num_regions: {num_regions}')
        self.local_feat_dim = local_feat_dim
        self.hidden_dim = hidden_dim
        self.num_regions = num_regions
        self.num_bboxes = num_bboxes
        self.bbox_to_labels = bbox_to_labels
        self.bbox_group_to_labels = bbox_group_to_labels
        assert len(bbox_to_labels) <= num_bboxes
        assert type(bbox_to_labels) == list # list of [idx, labels]
        assert type(bbox_to_labels[0]) == list 
        assert type(bbox_group_to_labels) == list # list of [idxs, labels]
        assert type(bbox_group_to_labels[0]) == list
        # create bbox projection layers
        self.loc_projs = nn.ModuleList([nn.Linear(local_feat_dim, hidden_dim) for _ in range(num_bboxes)])
        self.bbox_mid_layer = nn.ModuleList([nn.Linear(num_regions * hidden_dim, hidden_dim) for _ in range(num_bboxes)])
        # create layers for the multi-label classification
        self.loc_mlc_fc = nn.ModuleList([nn.Linear(hidden_dim, len(labels)) for _, labels in bbox_to_labels])
        self.glob_mlc_fc = nn.ModuleList([nn.Linear(hidden_dim * len(bbox_group), len(labels)) \
                                            for bbox_group, labels in bbox_group_to_labels])

    def forward(self, local_features):
        # local_features: (batch_size, num_regions, local_feat_dim)
        # return: mlc_scores (batch_size, num_classes)

        batch_size = local_features.size(0)       
        
        # 1) Bbox feature extraction
        bbox_features = []
        for i in range(self.num_bboxes):
            # Project local features
            x = self.loc_projs[i](local_features) # (batch_size, num_regions, hidden_dim)
            x = x.view(batch_size, -1)
            x = torch.relu(x) # (batch_size, num_regions * hidden_dim)
            assert x.shape == (batch_size, self.num_regions * self.hidden_dim)
            # Apply mid layer
            x = self.bbox_mid_layer[i](x) # (batch_size, hidden_dim)
            x = torch.relu(x) # (batch_size, hidden_dim)
            bbox_features.append(x)

        # 2) Multi-label classification
        mlc_scores = []
        # 2.1) Compute the multi-label classification for each bounding box
        for i, (idx, _) in enumerate(self.bbox_to_labels):
            loc_label_logits = self.loc_mlc_fc[i](bbox_features[idx]) # (batch_size, num_labels)
            mlc_scores.append(loc_label_logits)
        # 2.2) Compute the multi-label classification for each bounding box group
        for i, (bbox_group, _) in enumerate(self.bbox_group_to_labels):
            # 2.2.1) Get the features for the bounding boxes in the group
            group_features = [bbox_features[i] for i in bbox_group]
            group_features = torch.stack(group_features, dim=1) # (batch_size, num_bboxes, hidden_dim)
            group_features = group_features.view(batch_size, -1) # (batch_size, num_bboxes * hidden_dim)
            # 2.2.2) Apply the global multi-label classification layer
            glob_label_logits = self.glob_mlc_fc[i](group_features) # (batch_size, num_labels)
            mlc_scores.append(glob_label_logits)
        # 2.3) Concatenate the local and global logits
        mlc_scores = torch.concat(mlc_scores, dim=1) # (batch_size, (num_bboxes + num_bbox_groups) * num_labels)

        # 3) Return multi-label classification scores
        return mlc_scores

class BoundingBoxRegressorAndMultiLabelClassifier_v5(nn.Module):
    """
    Bounding box regressor and multi-label classifier (v5).
    We will receive:
        - local_features: (batch_size, local_feat_dim, width, height)
        - global_features: (batch_size, global_feat_dim)
        - pred_bbox_coords: (batch_size, num_bboxes * 4)
            which we assume to have been predicted by a previous model
        - pred_bbox_classes: (batch_size, num_bboxes)
            which we assume to have been predicted by a previous model
    What we will do:
        - We will use the predicted bboxes to perform ROI Align pooling on the local features
        - If certain bboxes are not present, we will replace them with linear projections of the global and local features
        - We will use each individual ROI pooled features to:
            - predict refined bounding box coordinates
            - predict the bounding box presence
        - We will merge the ROI pooled features for each bbox group to:
            - predict the multi-label classification scores
    What we will return:
        - pred_bbox_coords: (batch_size, num_bboxes * 4)
        - pred_bbox_presence: (batch_size, num_bboxes)
        - mlc_scores: (batch_size, num_classes)
    """

    def __init__(self, local_feat_dim, input_size, roi_align_output_size,
                roi_align_spatial_scale, hidden_dim, num_boxes, num_boxes_to_supervise,
                bbox_to_labels, bbox_group_to_labels):
        super().__init__()
        logger.info('BoundingBoxRegressorAndMultiLabelClassifier_v5:')
        logger.info(f'  local_feat_dim: {local_feat_dim}')
        logger.info(f'  input_size: {input_size}')
        logger.info(f'  roi_align_output_size: {roi_align_output_size}')
        logger.info(f'  roi_align_spatial_scale: {roi_align_spatial_scale}')
        logger.info(f'  hidden_dim: {hidden_dim}')
        logger.info(f'  num_boxes: {num_boxes}')
        logger.info(f'  num_boxes_to_supervise: {num_boxes_to_supervise}')        
        
        assert len(bbox_to_labels) <= num_boxes # some bboxes may not have labels
        assert type(bbox_to_labels) == list # list of [idx, labels]
        assert type(bbox_to_labels[0]) == list 
        assert type(bbox_group_to_labels) == list # list of [idxs, labels]
        assert type(bbox_group_to_labels[0]) == list
        assert num_boxes_to_supervise <= num_boxes

        self.local_feat_dim = local_feat_dim
        self.input_size = input_size
        self.roi_align_output_size = roi_align_output_size
        self.roi_align_spatial_scale = roi_align_spatial_scale
        self.hidden_dim = hidden_dim
        self.num_boxes = num_boxes
        self.num_boxes_to_supervise = num_boxes_to_supervise
        self.bbox_to_labels = bbox_to_labels
        self.bbox_group_to_labels = bbox_group_to_labels
        self.num_labels = sum(len(labels) for _, labels in bbox_to_labels) + \
                            sum(len(labels) for _, labels in bbox_group_to_labels)
        logger.info(f'  self.num_labels: {self.num_labels}')

        # create layers for the multi-label classification
        self.loc_mlc_fc = nn.ModuleList([nn.Linear(hidden_dim, len(labels)) for _, labels in bbox_to_labels])
        self.glob_mlc_fc = nn.ModuleList([nn.Linear(hidden_dim * len(bbox_group), len(labels)) \
                                            for bbox_group, labels in bbox_group_to_labels])

        # create layers for the bounding box regression
        self.roi_align_fc = nn.Linear(roi_align_output_size ** 2 * local_feat_dim, hidden_dim)
        self.bbox_coords_fc = nn.ModuleList([nn.Linear(hidden_dim, 4) for _ in range(num_boxes_to_supervise)])
        self.bbox_presence_fc = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_boxes_to_supervise)])

    def forward(self, local_features, pred_bbox_coords, refine_bbox_coords=True):
        batch_size = local_features.size(0)
        assert local_features.shape == (batch_size, self.local_feat_dim, self.input_size, self.input_size)
        assert pred_bbox_coords.shape == (batch_size, self.num_boxes_to_supervise, 4)        
        assert len(pred_bbox_coords) == batch_size
        assert 0 <= pred_bbox_coords.min() and pred_bbox_coords.max() <= 1

        # 1) ROI Align pooling
        # convert the predicted bounding box coordinates to the format expected by ROI Align
        n_boxes = batch_size * self.num_boxes
        roi_align_boxes = torch.zeros((n_boxes, 5), dtype=torch.float32, device=local_features.device)
        # set all coordinates to [0, 0, 1, 1] by default so that missing boxes will be replaced by the whole image
        roi_align_boxes[:, 3:] = 1 # (n_boxes, 5)
        for i in range(batch_size):
            batch_start = i * self.num_boxes
            batch_end = (i + 1) * self.num_boxes
            roi_align_boxes[batch_start:batch_end, 0] = i # batch index
            roi_align_boxes[batch_start:batch_start+self.num_boxes_to_supervise, 1:] = pred_bbox_coords[i] # (x1, y1, x2, y2)
        # perform ROI Align pooling
        roi_align_output = roi_align(
            input=local_features,
            boxes=roi_align_boxes,
            output_size=self.roi_align_output_size,
            spatial_scale=self.roi_align_spatial_scale,
        )
        assert roi_align_output.shape == (n_boxes, self.local_feat_dim, self.roi_align_output_size, self.roi_align_output_size)
        # flatten the ROI Align output and project it to the hidden dimension
        roi_align_output = roi_align_output.view(n_boxes, -1) # (n_boxes, local_feat_dim * roi_align_output_size ** 2)
        bbox_features = self.roi_align_fc(roi_align_output) # (n_boxes, hidden_dim)
        bbox_features = torch.relu(bbox_features)
        bbox_features = bbox_features.view(batch_size, self.num_boxes, -1) # (batch_size, num_boxes, hidden_dim)        

        # 2) Predict bounding box coordinates and presence
        if refine_bbox_coords:
            bbox_features_to_supervise = bbox_features[:, :self.num_boxes_to_supervise] # (batch_size, num_boxes_to_supervise, hidden_dim)
            assert bbox_features_to_supervise.shape == (batch_size, self.num_boxes_to_supervise, self.hidden_dim)
            pred_bbox_deltas = self.bbox_coords_fc(bbox_features_to_supervise) # (batch_size, num_boxes_to_supervise, 4)
            # the deltas are scale invariant, so we need to scale them back to the original image size
            # x1_adjusted = x1_pred + x1_delta * (x2_pred - x1_pred)
            pred_bbox_widths = pred_bbox_coords[:, :, 2] - pred_bbox_coords[:, :, 0] # (batch_size, num_boxes_to_supervise)
            pred_bbox_heights = pred_bbox_coords[:, :, 3] - pred_bbox_coords[:, :, 1] # (batch_size, num_boxes_to_supervise)
            pred_refined_bbox_coords = torch.zeros_like(pred_bbox_coords) # (batch_size, num_boxes_to_supervise, 4)
            pred_refined_bbox_coords[:, :, 0] = pred_bbox_coords[:, :, 0] + pred_bbox_deltas[:, :, 0] * pred_bbox_widths
            pred_refined_bbox_coords[:, :, 1] = pred_bbox_coords[:, :, 1] + pred_bbox_deltas[:, :, 1] * pred_bbox_heights
            pred_refined_bbox_coords[:, :, 2] = pred_bbox_coords[:, :, 2] + pred_bbox_deltas[:, :, 2] * pred_bbox_widths
            pred_refined_bbox_coords[:, :, 3] = pred_bbox_coords[:, :, 3] + pred_bbox_deltas[:, :, 3] * pred_bbox_heights
            pred_bbox_presence = self.bbox_presence_fc(bbox_features_to_supervise) # (batch_size, num_boxes_to_supervise, 1)
            pred_bbox_presence = pred_bbox_presence.squeeze(-1) # (batch_size, num_boxes_to_supervise)

        # 3) Predict multi-label classification scores
        mlc_scores = []
        # 3.1) Compute the multi-label classification for each bounding box
        for i, (idx, _) in enumerate(self.bbox_to_labels):
            loc_label_logits = self.loc_mlc_fc[i](bbox_features[:, idx]) # (batch_size, num_labels)
            mlc_scores.append(loc_label_logits)
        # 3.2) Compute the multi-label classification for each bounding box group
        for i, (bbox_group, _) in enumerate(self.bbox_group_to_labels):
            # 3.2.1) Get the features for the bounding boxes in the group
            group_features = [bbox_features[:, idx] for idx in bbox_group]
            group_features = torch.cat(group_features, dim=1) # (batch_size, hidden_dim * len(bbox_group))
            # 3.2.2) Apply the global multi-label classification layer
            glob_label_logits = self.glob_mlc_fc[i](group_features) # (batch_size, num_labels)
            mlc_scores.append(glob_label_logits)
        # 3.3) Concatenate the local and global logits
        mlc_scores = torch.concat(mlc_scores, dim=1) # (batch_size, num_labels)
        assert mlc_scores.shape == (batch_size, self.num_labels)

        # 4) Return the predicted refined bounding box coordinates, presence and multi-label classification scores
        if refine_bbox_coords:
            return pred_refined_bbox_coords, pred_bbox_presence, mlc_scores
        else:
            return mlc_scores

class BoundingBoxRegressorAndMultiLabelClassifier_v6(nn.Module):
    def __init__(self, global_feat_dim, local_feat_dim, input_size, roi_align_output_size,
                roi_align_spatial_scale, hidden_dim, num_boxes, num_boxes_to_supervise,
                bbox_to_labels, bbox_group_to_labels, train_average_bbox_coords):
        super().__init__()
        logger.info('BoundingBoxRegressorAndMultiLabelClassifier_v6:')
        logger.info(f'  global_feat_dim: {global_feat_dim}')
        logger.info(f'  local_feat_dim: {local_feat_dim}')
        logger.info(f'  input_size: {input_size}')
        logger.info(f'  roi_align_output_size: {roi_align_output_size}')
        logger.info(f'  roi_align_spatial_scale: {roi_align_spatial_scale}')
        logger.info(f'  hidden_dim: {hidden_dim}')
        logger.info(f'  num_boxes: {num_boxes}')
        logger.info(f'  num_boxes_to_supervise: {num_boxes_to_supervise}')
        logger.info(f'  len(train_average_bbox_coords): {len(train_average_bbox_coords)}')
        
        assert len(bbox_to_labels) <= num_boxes # some bboxes may not have labels
        assert type(bbox_to_labels) == list # list of [idx, labels]
        assert type(bbox_to_labels[0]) == list 
        assert type(bbox_group_to_labels) == list # list of [idxs, labels]
        assert type(bbox_group_to_labels[0]) == list
        assert num_boxes_to_supervise <= num_boxes

        self.local_feat_dim = local_feat_dim
        self.input_size = input_size
        self.roi_align_output_size = roi_align_output_size
        self.roi_align_spatial_scale = roi_align_spatial_scale
        self.hidden_dim = hidden_dim
        self.num_boxes = num_boxes
        self.num_boxes_to_supervise = num_boxes_to_supervise
        self.bbox_to_labels = bbox_to_labels
        self.bbox_group_to_labels = bbox_group_to_labels
        self.num_labels = sum(len(labels) for _, labels in bbox_to_labels) + \
                            sum(len(labels) for _, labels in bbox_group_to_labels)
        logger.info(f'  self.num_labels: {self.num_labels}')

        # create parameters for the average coordinates of the training set
        self.train_average_bbox_coords = nn.parameter.Parameter(torch.zeros(len(train_average_bbox_coords)), requires_grad=False)
        if type(train_average_bbox_coords) == list:
            train_average_bbox_coords = torch.tensor(train_average_bbox_coords)
        self.train_average_bbox_coords.data.copy_(train_average_bbox_coords)

        # create layers for the bounding box regression
        self.loc_projs = nn.ModuleList([nn.Linear(local_feat_dim, hidden_dim) for _ in range(num_boxes)])
        self.glob_projs = nn.ModuleList([nn.Linear(global_feat_dim, hidden_dim) for _ in range(num_boxes)])
        self.bbox_mid_layer = nn.ModuleList([nn.Linear((roi_align_output_size**2 + 1) * hidden_dim, hidden_dim) for _ in range(num_boxes)]) # +1 for global feature
        self.bbox_coords_fc = nn.ModuleList([nn.Linear(hidden_dim, 4) for _ in range(num_boxes)]) 
        self.bbox_presence_fc = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_boxes)])
        self.roi_align_fc = nn.Linear(roi_align_output_size ** 2 * local_feat_dim, hidden_dim)

        # create layers for the multi-label classification
        self.loc_mlc_fc = nn.ModuleList([nn.Linear(hidden_dim, len(labels)) for _, labels in bbox_to_labels])
        self.glob_mlc_fc = nn.ModuleList([nn.Linear(hidden_dim * len(bbox_group), len(labels)) \
                                            for bbox_group, labels in bbox_group_to_labels])


    def forward(self, global_features, local_features, pred_bbox_coords):
        batch_size = local_features.size(0)
        assert local_features.shape == (batch_size, self.local_feat_dim, self.input_size, self.input_size)
        assert pred_bbox_coords.shape == (batch_size, self.num_boxes_to_supervise, 4)        
        assert len(pred_bbox_coords) == batch_size
        assert 0 <= pred_bbox_coords.min() and pred_bbox_coords.max() <= 1

        # 1) ROI Align pooling
        # convert the predicted bounding box coordinates to the format expected by ROI Align
        n_boxes = batch_size * self.num_boxes
        roi_align_boxes = torch.zeros((n_boxes, 5), dtype=torch.float32, device=local_features.device)
        # set all coordinates to [0, 0, 1, 1] by default so that missing boxes will be replaced by the whole image
        roi_align_boxes[:, 3:] = 1 # (n_boxes, 5)
        for i in range(batch_size):
            batch_start = i * self.num_boxes
            batch_end = (i + 1) * self.num_boxes
            roi_align_boxes[batch_start:batch_end, 0] = i # batch index
            roi_align_boxes[batch_start:batch_start+self.num_boxes_to_supervise, 1:] = pred_bbox_coords[i] # (x1, y1, x2, y2)
        # perform ROI Align pooling
        roi_align_output = roi_align(
            input=local_features,
            boxes=roi_align_boxes,
            output_size=self.roi_align_output_size,
            spatial_scale=self.roi_align_spatial_scale,
        )
        assert roi_align_output.shape == (n_boxes, self.local_feat_dim, self.roi_align_output_size, self.roi_align_output_size)
        roi_align_output = roi_align_output.reshape(batch_size, self.num_boxes, self.local_feat_dim, -1)
        roi_align_output = roi_align_output.permute(0, 1, 3, 2)
        
        # 2) Bbox feature extraction and prediction
        bbox_features = []
        pred_bbox_coords = []
        pred_bbox_presence = []
        for i in range(self.num_boxes):
            # Project global features
            xg = self.glob_projs[i](global_features) # (batch_size, hidden_dim)
            xg = xg.unsqueeze(1) # (batch_size, 1, hidden_dim)
            # Project ROI Align features
            xl = self.loc_projs[i](roi_align_output[:, i]) # (batch_size, num_regions, hidden_dim)
            # Concatenate global and local features
            x = torch.cat([xg, xl], dim=1) # (batch_size, num_regions + 1, hidden_dim)
            x = x.view(batch_size, -1) # (batch_size, (num_regions + 1) * hidden_dim)
            x = torch.relu(x) # (batch_size, (num_regions + 1) * hidden_dim)
            # Apply mid layer
            x = self.bbox_mid_layer[i](x) # (batch_size, hidden_dim)
            bbox_features.append(x)
            # Compute the bounding box coordinates and presence
            # but only for the first num_boxes_to_supervise bounding boxes
            if i < self.num_boxes_to_supervise:
                bbox_coords = self.bbox_coords_fc[i](x) # (batch_size, 4)
                bbox_presence = self.bbox_presence_fc[i](x) # (batch_size, 1)
                pred_bbox_coords.append(bbox_coords)
                pred_bbox_presence.append(bbox_presence)
        # Concatenate the bounding box coordinates and presence for all classes
        pred_bbox_coords = torch.stack(pred_bbox_coords, dim=1) # (batch_size, num_bboxs_to_supervise, 4)
        pred_bbox_coords = pred_bbox_coords.view(batch_size, -1) # (batch_size, num_bboxs_to_supervise * 4)
        pred_bbox_coords = pred_bbox_coords + self.train_average_bbox_coords.view(1, -1) # (batch_size, num_bboxs_to_supervise * 4)
        pred_bbox_presence = torch.stack(pred_bbox_presence, dim=1) # (batch_size, num_bboxs_to_supervise, 1)
        pred_bbox_presence = pred_bbox_presence.squeeze(-1) # (batch_size, num_bboxs_to_supervise)

        # 3) Predict multi-label classification scores
        mlc_scores = []
        # 3.1) Compute the multi-label classification for each bounding box
        for i, (idx, _) in enumerate(self.bbox_to_labels):
            loc_label_logits = self.loc_mlc_fc[i](bbox_features[idx]) # (batch_size, num_labels)
            mlc_scores.append(loc_label_logits)
        # 3.2) Compute the multi-label classification for each bounding box group
        for i, (bbox_group, _) in enumerate(self.bbox_group_to_labels):
            # 3.2.1) Get the features for the bounding boxes in the group
            group_features = [bbox_features[idx] for idx in bbox_group]
            group_features = torch.cat(group_features, dim=1) # (batch_size, hidden_dim * len(bbox_group))
            # 3.2.2) Apply the global multi-label classification layer
            glob_label_logits = self.glob_mlc_fc[i](group_features) # (batch_size, num_labels)
            mlc_scores.append(glob_label_logits)
        # 3.3) Concatenate the local and global logits
        mlc_scores = torch.concat(mlc_scores, dim=1) # (batch_size, num_labels)
        assert mlc_scores.shape == (batch_size, self.num_labels)

        # 4) Return the predicted bounding box coordinates, presence and multi-label classification scores
        return pred_bbox_coords, pred_bbox_presence, mlc_scores
    
class MultiClassBoundingBoxRegressor(nn.Module):
    """
    A bounding box regressor for multiple object classes. 
    This model predicts bounding box coordinates and presence probabilities
    for an arbitrary number of classes and regions.
    
    Supports both 'xyxy' and 'cxcywh' bbox formats, and can predict coordinates
    relative to grid cell centers.
    """
    def __init__(self, local_feat_dim, bbox_format='xyxy', predict_relative=False):
        """
        Initializes the bounding box regressor.
        
        Args:
            local_feat_dim (int): The dimension of the input feature vector.
            bbox_format (str): The format of bounding box coordinates, either 'xyxy' or 'cxcywh'.
            predict_relative (bool): Whether to predict bbox coordinates relative to grid cell centers.
        """
        super().__init__()
        logger.info('MultiClassBoundingBoxRegressor')
        logger.info(f'  local_feat_dim: {local_feat_dim}')
        logger.info(f'  bbox_format: {bbox_format}')
        logger.info(f'  predict_relative: {predict_relative}')
        
        self.local_feat_dim = local_feat_dim
        self.bbox_format = bbox_format
        self.predict_relative = predict_relative
        
        if bbox_format not in ['xyxy', 'cxcywh']:
            raise ValueError("bbox_format must be either 'xyxy' or 'cxcywh'")
        
        # Fully connected layers for bounding box regression and presence prediction
        self.bbox_coords_fc = nn.Linear(local_feat_dim, 4)  # Outputs (cx, cy, w, h) or (x1, y1, x2, y2)
        self.bbox_presence_fc = nn.Linear(local_feat_dim, 1) # Outputs probability of presence
        
        # Cache for class ID tensor to avoid redundant computation
        self.class_ids_cache = dict()
        
        # Cache for grid cell centers
        self.grid_centers_cache = dict()
    
    def _get_class_ids(self, num_classes, num_regions):
        """
        Retrieves or computes a tensor of class indices for the given dimensions.
        
        Args:
            num_classes (int): Number of object classes.
            num_regions (int): Number of regions to predict.
        
        Returns:
            torch.Tensor: A tensor of shape (num_classes, num_regions) containing class IDs.
        """
        shape = (num_classes, num_regions)
        if shape not in self.class_ids_cache:
            self.class_ids_cache[shape] = torch.arange(
                num_classes, device=self.bbox_coords_fc.weight.device
            ).unsqueeze(-1).expand(-1, num_regions).contiguous()
        return self.class_ids_cache[shape]
    
    def _get_grid_centers(self, num_regions):
        """
        Retrieves or computes a tensor of grid centers for the given number of regions.
        
        Args:
            num_regions (int): Number of regions to predict.
        
        Returns:
            torch.Tensor: A tensor of shape (num_regions, 2) containing grid centers.
        """
        if num_regions not in self.grid_centers_cache:
            grid_height = grid_width = math.isqrt(num_regions)
            assert grid_height * grid_width == num_regions, "Number of regions must be a square number"
            device = self.bbox_coords_fc.weight.device
            centers = get_grid_centers(grid_height, grid_width, device) # (H, W, 2)
            centers = centers.view(-1, 2)  # (num_regions, 2)
            if self.bbox_format == 'xyxy': # We need to repeat the centers for each corner
                centers = centers.repeat(1, 2) # (num_regions, 4)
            self.grid_centers_cache[num_regions] = centers
            # print_blue(f'grid_centers_cache: {self.grid_centers_cache[num_regions].shape}')
        return self.grid_centers_cache[num_regions]


    def forward(self, local_features, predict_presence=True, predict_coords=True,
                apply_nms=False, iou_threshold=0.3, conf_threshold=0.5, max_det_per_class=20):
        """
        Performs bounding box regression and optionally applies Non-Maximum Suppression (NMS).
        
        Args:
            local_features (torch.Tensor): Input feature tensor of shape (batch_size, num_classes, num_regions, local_feat_dim).
            predict_presence (bool): Whether to predict the presence probability of bounding boxes.
            predict_coords (bool): Whether to predict bounding box coordinates.
            apply_nms (bool): Whether to apply Non-Maximum Suppression (NMS) to filter overlapping boxes.
            iou_threshold (float): IoU threshold for NMS.
            conf_threshold (float): Confidence threshold for filtering predictions.
            max_det_per_class (int): Maximum number of detections per class after thresholding.
        
        Returns:
            If apply_nms=True:
                list of tuples: [(bbox_coords, bbox_probs, class_ids)] for each image in batch.
            Otherwise:
                torch.Tensor: Bounding box coordinates (if predict_coords=True).
                torch.Tensor: Bounding box presence scores (if predict_presence=True).
        """
        
        if apply_nms:
            assert predict_coords and predict_presence, "NMS requires both coordinate and presence prediction"
            
            # Compute bounding box coordinates and presence probabilities
            bbox_coords = self.bbox_coords_fc(local_features)  # (batch_size, num_classes, num_regions, 4)
            bbox_presence = self.bbox_presence_fc(local_features)  # (batch_size, num_classes, num_regions, 1)
            bbox_presence_probs = torch.sigmoid(bbox_presence).squeeze(-1)  # (batch_size, num_classes, num_regions)

            # Ensure correct tensor shapes
            assert bbox_coords.ndim == 4 and bbox_coords.shape[-1] == 4, "Bounding box coordinates must have shape (batch, num_classes, num_regions, 4)"
            assert bbox_presence.ndim == 4 and bbox_presence_probs.ndim == 3, "Presence scores must have correct dimensions"

            num_classes = bbox_coords.size(1)
            num_regions = bbox_coords.size(2)
            class_ids = self._get_class_ids(num_classes, num_regions)  # (num_classes, num_regions)

            # Add grid cell centers to bbox_coords if predicting relative coordinates
            if self.predict_relative:
                if self.bbox_format == 'xyxy':
                    bbox_coords += self._get_grid_centers(num_regions).view(1, 1, num_regions, 4)
                else: # 'cxcywh' -> we only add centers to the first two coordinates
                    bbox_coords[:, :, :, :2] += self._get_grid_centers(num_regions).view(1, 1, num_regions, 2)
            
            output = [None] * bbox_coords.size(0)  # Store results for each image in batch
            
            for i in range(bbox_coords.size(0)):  # Iterate over batch
                bbox_coords_i = bbox_coords[i]  # (num_classes, num_regions, 4)
                bbox_probs_i = bbox_presence_probs[i]  # (num_classes, num_regions)
                class_ids_i = class_ids  # (num_classes, num_regions)
                
                # Apply max detections per class constraint
                if bbox_coords_i.size(1) > max_det_per_class:
                    bbox_probs_i, idxs = torch.topk(bbox_probs_i, max_det_per_class, dim=1) # (num_classes, max_det_per_class)
                    bbox_coords_i = torch.gather(bbox_coords_i, 1, idxs.unsqueeze(-1).expand(-1, -1, 4)) # (num_classes, max_det_per_class, 4)
                    class_ids_i  = torch.gather(class_ids_i, 1, idxs) # (num_classes, max_det_per_class)
                
                # Apply confidence threshold
                mask = bbox_probs_i > conf_threshold # (num_classes, num_regions) or (num_classes, max_det_per_class)
                bbox_coords_i = bbox_coords_i[mask]  # (num_detections, 4)
                bbox_probs_i = bbox_probs_i[mask]  # (num_detections)
                class_ids_i = class_ids_i[mask]  # (num_detections)
                
                # Apply NMS if there are remaining detections
                if bbox_coords_i.size(0) > 0:
                    if self.bbox_format == 'cxcywh':
                        bbox_coords_i_xyxy = cxcywh_to_xyxy_tensor(bbox_coords_i) # Convert to 'xyxy' format
                    else: # 'xyxy'
                        bbox_coords_i_xyxy = bbox_coords_i # Already in 'xyxy' format
                    shifted_bbox_coords = bbox_coords_i_xyxy + class_ids_i.unsqueeze(-1).float() * 10.0  # Offset for per-class separation
                    keep = ops.nms(shifted_bbox_coords, bbox_probs_i, iou_threshold) # NMS requires 'xyxy' format
                    bbox_coords_i = bbox_coords_i[keep]
                    bbox_probs_i = bbox_probs_i[keep]
                    class_ids_i = class_ids_i[keep]
                
                output[i] = (bbox_coords_i, bbox_probs_i, class_ids_i)
            
            return output, bbox_presence
        
        # Compute bounding box coordinates if requested
        if predict_coords:
            bbox_coords = self.bbox_coords_fc(local_features) # (batch_size, num_classes, num_regions, 4)
            if self.predict_relative:
                num_regions = bbox_coords.size(2)
                if self.bbox_format == 'xyxy':
                    bbox_coords += self._get_grid_centers(num_regions).view(1, 1, num_regions, 4)
                else: # 'cxcywh' -> we only add centers to the first two coordinates
                    bbox_coords[:, :, :, :2] += self._get_grid_centers(num_regions).view(1, 1, num_regions, 2)
        else:
            predict_coords = None
        # Compute bounding box presence probabilities if requested
        bbox_presence = self.bbox_presence_fc(local_features) if predict_presence else None
        
        # Return appropriate values based on requested predictions
        if predict_coords and predict_presence:
            return bbox_coords, bbox_presence
        elif predict_coords:
            return bbox_coords
        elif predict_presence:
            return bbox_presence
