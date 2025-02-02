import torch
import torch.nn as nn
from torchvision.ops import roi_align
import torch.nn.functional as F

class MLCVersion:
    DEFAULT = 'default' # global features -> fully connected layer -> softmax
    V1 = 'v1'
    V2 = 'v2'
    V3 = 'v3'
    
    @staticmethod
    def get_versions():
        return [MLCVersion.DEFAULT, MLCVersion.V1, MLCVersion.V2, MLCVersion.V3]

class MultilabelClassifier_v1(nn.Module):
    def __init__(self, local_feat_dim, global_feat_dim, hidden_dim, num_bboxes, num_regions,
                 bbox_to_labels, bbox_group_to_labels):
        super().__init__()
        print('MultilabelClassifier_v1:')
        print(f'  local_feat_dim: {local_feat_dim}')
        print(f'  global_feat_dim: {global_feat_dim}')
        print(f'  hidden_dim: {hidden_dim}')
        print(f'  num_bboxes: {num_bboxes}')
        print(f'  num_regions: {num_regions}')
        self.local_feat_dim = local_feat_dim
        self.global_feat_dim = global_feat_dim
        self.hidden_dim = hidden_dim
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
        self.glob_projs = nn.ModuleList([nn.Linear(global_feat_dim, hidden_dim) for _ in range(num_bboxes)])
        self.bbox_mid_layer = nn.ModuleList([nn.Linear((num_regions + 1) * hidden_dim, hidden_dim) for _ in range(num_bboxes)]) # +1 for global features
        # create layers for the multi-label classification
        self.loc_mlc_fc = nn.ModuleList([nn.Linear(hidden_dim, len(labels)) for _, labels in bbox_to_labels])
        self.glob_mlc_fc = nn.ModuleList([nn.Linear(hidden_dim * len(bbox_group), len(labels)) \
                                            for bbox_group, labels in bbox_group_to_labels])

    def forward(self, local_features, global_features):
        # local_features: (batch_size, num_regions, local_feat_dim)
        # global_features: (batch_size, global_feat_dim)
        # return: mlc_scores (batch_size, num_classes)

        batch_size = local_features.size(0)       
        
        # 1) Bbox feature extraction
        bbox_features = []
        for i in range(self.num_bboxes):
            # 1.1) Project global features
            xg = self.glob_projs[i](global_features) # (batch_size, hidden_dim)
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

        # 2) Multi-label classification
        mlc_scores = []
        # 2.1) Compute the multi-label classification for each bounding box
        for i, (idx, _) in enumerate(self.bbox_to_labels):
            loc_label_logits = self.loc_mlc_fc[i](bbox_features[idx]) # (batch_size, num_labels)
            mlc_scores.append(loc_label_logits)
        # 2.2) Compute the multi-label classification for each bounding box group
        for i, (bbox_group, _) in enumerate(self.bbox_group_to_labels):
            # 2.2.1) Get the features for the bounding boxes in the group
            group_features = [bbox_features[j] for j in bbox_group]
            group_features = torch.cat(group_features, dim=1) # (batch_size, hidden_dim * len(bbox_group))
            # 2.2.2) Apply the global multi-label classification layer
            glob_label_logits = self.glob_mlc_fc[i](group_features) # (batch_size, num_labels)
            mlc_scores.append(glob_label_logits)
        # 2.3) Concatenate the local and global logits
        mlc_scores = torch.cat(mlc_scores, dim=1) # (batch_size, (num_bboxes + num_bbox_groups) * num_labels)

        # 3) Return the predicted multi-label classification scores
        return mlc_scores

class MultilabelClassifier_v2(nn.Module):
    def __init__(self, global_feat_dim, local_feat_dim, input_size, roi_align_output_size,
                roi_align_spatial_scale, hidden_dim, num_boxes, num_annotated_boxes,
                bbox_to_labels, bbox_group_to_labels):
        super().__init__()
        print('MultilabelClassifier_v2:')
        print(f'  global_feat_dim: {global_feat_dim}')
        print(f'  local_feat_dim: {local_feat_dim}')
        print(f'  input_size: {input_size}')
        print(f'  roi_align_output_size: {roi_align_output_size}')
        print(f'  roi_align_spatial_scale: {roi_align_spatial_scale}')
        print(f'  hidden_dim: {hidden_dim}')
        print(f'  num_boxes: {num_boxes}')
        print(f'  num_annotated_boxes: {num_annotated_boxes}')
        
        assert len(bbox_to_labels) <= num_boxes # some bboxes may not have labels
        assert type(bbox_to_labels) == list # list of [idx, labels]
        assert type(bbox_to_labels[0]) == list 
        assert type(bbox_group_to_labels) == list # list of [idxs, labels]
        assert type(bbox_group_to_labels[0]) == list
        assert num_annotated_boxes <= num_boxes # some bboxes may not be annotated

        self.local_feat_dim = local_feat_dim
        self.input_size = input_size
        self.roi_align_output_size = roi_align_output_size
        self.roi_align_spatial_scale = roi_align_spatial_scale
        self.hidden_dim = hidden_dim
        self.num_boxes = num_boxes
        self.num_annotated_boxes = num_annotated_boxes
        self.bbox_to_labels = bbox_to_labels
        self.bbox_group_to_labels = bbox_group_to_labels
        self.num_labels = sum(len(labels) for _, labels in bbox_to_labels) + \
                            sum(len(labels) for _, labels in bbox_group_to_labels)
        print(f'  self.num_labels: {self.num_labels}')

        # create bbox projection layers
        self.loc_projs = nn.ModuleList([nn.Linear(local_feat_dim, hidden_dim) for _ in range(num_boxes)])
        self.glob_projs = nn.ModuleList([nn.Linear(global_feat_dim, hidden_dim) for _ in range(num_boxes)])
        self.bbox_mid_layer = nn.ModuleList([nn.Linear((roi_align_output_size**2 + 1) * hidden_dim, hidden_dim) for _ in range(num_boxes)]) # +1 for global feature
        self.roi_align_fc = nn.Linear(roi_align_output_size ** 2 * local_feat_dim, hidden_dim)

        # create layers for the multi-label classification
        self.loc_mlc_fc = nn.ModuleList([nn.Linear(hidden_dim, len(labels)) for _, labels in bbox_to_labels])
        self.glob_mlc_fc = nn.ModuleList([nn.Linear(hidden_dim * len(bbox_group), len(labels)) \
                                            for bbox_group, labels in bbox_group_to_labels])

    def forward(self, local_features, global_features, pred_bbox_coords):
        batch_size = local_features.size(0)
        assert local_features.shape == (batch_size, self.local_feat_dim, self.input_size, self.input_size)
        assert pred_bbox_coords.shape == (batch_size, self.num_annotated_boxes, 4)
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
            roi_align_boxes[batch_start:batch_start+self.num_annotated_boxes, 1:] = pred_bbox_coords[i] # (x1, y1, x2, y2)
        # perform ROI Align pooling
        roi_align_output = roi_align(
            input=local_features,
            boxes=roi_align_boxes,
            output_size=self.roi_align_output_size,
            spatial_scale=self.roi_align_spatial_scale,
        )
        assert roi_align_output.shape == (n_boxes, self.local_feat_dim, self.roi_align_output_size, self.roi_align_output_size)
        roi_align_output = roi_align_output.reshape(batch_size, self.num_boxes, self.local_feat_dim, -1)
        roi_align_output = roi_align_output.permute(0, 1, 3, 2) # (batch_size, num_boxes, num_regions, local_feat_dim)
        
        # 2) Bbox feature extraction
        bbox_features = []
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
        
        # 3) Predict multi-label classification scores
        mlc_scores = []
        # 3.1) Compute the multi-label classification for each bounding box
        for i, (idx, _) in enumerate(self.bbox_to_labels):
            loc_label_logits = self.loc_mlc_fc[i](bbox_features[idx]) # (batch_size, num_labels)
            mlc_scores.append(loc_label_logits)
        # 3.2) Compute the multi-label classification for each bounding box group
        for i, (bbox_group, _) in enumerate(self.bbox_group_to_labels):
            # 3.2.1) Get the features for the bounding boxes in the group
            group_features = [bbox_features[j] for j in bbox_group]
            group_features = torch.cat(group_features, dim=1) # (batch_size, hidden_dim * len(bbox_group))
            # 3.2.2) Apply the global multi-label classification layer
            glob_label_logits = self.glob_mlc_fc[i](group_features) # (batch_size, num_labels)
            mlc_scores.append(glob_label_logits)
        # 3.3) Concatenate the local and global logits
        mlc_scores = torch.cat(mlc_scores, dim=1) # (batch_size, num_labels)
        assert mlc_scores.shape == (batch_size, self.num_labels)

        # 4) Return the predicted bounding box coordinates, presence and multi-label classification scores
        return mlc_scores

class MultilabelClassifier_v3(nn.Module):
    def __init__(self, local_feat_dim, global_feat_dim, hidden_dim, num_regions, num_labels):
        super().__init__()
        print('MultilabelClassifier_v3:')
        print(f'  local_feat_dim: {local_feat_dim}')
        print(f'  global_feat_dim: {global_feat_dim}')
        print(f'  hidden_dim: {hidden_dim}')
        print(f'  num_regions: {num_regions}')
        print(f'  num_labels: {num_labels}')
        self.local_feat_dim = local_feat_dim
        self.global_feat_dim = global_feat_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        # create bbox projection layers
        self.loc_proj = nn.Linear(local_feat_dim, hidden_dim)
        self.glob_proj = nn.Linear(global_feat_dim, hidden_dim)
        self.fc = nn.Linear((num_regions + 1) * hidden_dim, num_labels)

    def forward(self, local_features, global_features):
        # local_features: (batch_size, num_regions, local_feat_dim)
        # global_features: (batch_size, global_feat_dim)
        # return: mlc_scores (batch_size, num_labels)
        batch_size = local_features.size(0)
        x_loc = self.loc_proj(local_features) # (batch_size, num_regions, hidden_dim)
        x_glob = self.glob_proj(global_features) # (batch_size, hidden_dim)
        x_concat = torch.cat([x_loc, x_glob.unsqueeze(1)], dim=1) # (batch_size, num_regions + 1, hidden_dim)
        x_concat = x_concat.view(batch_size, -1) # (batch_size, (num_regions + 1) * hidden_dim)
        x_concat = F.gelu(x_concat) # (batch_size, (num_regions + 1) * hidden_dim)
        mlc_scores = self.fc(x_concat) # (batch_size, num_labels)
        return mlc_scores