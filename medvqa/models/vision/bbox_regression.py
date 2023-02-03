import torch
import torch.nn as nn

class BoundingBoxRegressor(nn.Module):

    def __init__(self, local_feat_dim, global_feat_dim, hidden_dim, num_classes):
        super().__init__()
        self.local_feat_dim = local_feat_dim
        self.global_feat_dim = global_feat_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.loc_proj = nn.Linear(local_feat_dim, num_classes * hidden_dim)
        self.glob_proj = nn.Linear(global_feat_dim, num_classes * hidden_dim)
        self.bbox_hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.bbox_regressor = nn.Linear(hidden_dim, 4)
        self.bbox_binary_classifier = nn.Linear(hidden_dim, 1)
        print('BoundingBoxRegressor:')
        print(f'  local_feat_dim: {local_feat_dim}')
        print(f'  global_feat_dim: {global_feat_dim}')
        print(f'  hidden_dim: {hidden_dim}')
        print(f'  num_classes: {num_classes}')

    def forward(self, local_features, global_average_pool):
        # local_features: (batch_size, num_regions, local_feat_dim)
        # global_average_pool: (batch_size, local_feat_dim)
        # return: (batch_size, num_classes, 4)
        
        # 1) Project the global feature to a different vector for each class
        xg = self.glob_proj(global_average_pool) # (batch_size, num_classes * hidden_dim)
        xg = xg.reshape(xg.size(0), -1, self.hidden_dim) # (batch_size, num_classes, hidden_dim)
        xg = xg.unsqueeze(2) # (batch_size, num_classes, 1, hidden_dim)
        assert xg.shape == (global_average_pool.size(0), self.num_classes, 1, self.hidden_dim)

        # 2) Project the local features to a different tensor for each class
        xl = self.loc_proj(local_features) # (batch_size, num_regions, num_classes * hidden_dim)
        xl = xl.reshape(xl.size(0), xl.size(1), self.num_classes, self.hidden_dim) # (batch_size, num_regions, num_classes, hidden_dim)
        xl = xl.permute(0, 2, 1, 3) # (batch_size, num_classes, num_regions, hidden_dim)
        assert xl.shape == (local_features.size(0), self.num_classes, local_features.size(1), self.hidden_dim)
        
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
        bbox_presence = self.bbox_binary_classifier(bbox) # (batch_size, num_classes, 1)        
        assert bbox_coords.shape == (weighted_sum.size(0), self.num_classes, 4)
        assert bbox_presence.shape == (weighted_sum.size(0), self.num_classes, 1)
        
        # 6) Reshape the bounding box coordinates and presence and return
        bbox_coords = bbox_coords.view(bbox_coords.size(0), -1) # (batch_size, num_classes * 4)
        bbox_presence = bbox_presence.squeeze(-1) # (batch_size, num_classes)
        return bbox_coords, bbox_presence
                


        