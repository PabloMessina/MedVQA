import torch
import torch.nn as nn

class BBoxRegressorVersion:
    V1 = 'v1'
    V2 = 'v2'
    V3 = 'v3'

class BoundingBoxRegressor_v1(nn.Module):

    def __init__(self, local_feat_dim, global_feat_dim, hidden_dim, num_classes, train_average_bbox_coords):
        super().__init__()
        print('BoundingBoxRegressor_v1:')
        print(f'  local_feat_dim: {local_feat_dim}')
        print(f'  global_feat_dim: {global_feat_dim}')
        print(f'  hidden_dim: {hidden_dim}')
        print(f'  num_classes: {num_classes}')
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
        print('BoundingBoxRegressor_v2:')
        print(f'  local_feat_dim: {local_feat_dim}')
        print(f'  global_feat_dim: {global_feat_dim}')
        print(f'  hidden_dim: {hidden_dim}')
        print(f'  num_classes: {num_classes}')
        print(f'  num_regions: {num_regions}')
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

    def forward(self, local_features, global_average_pool):
        # local_features: (batch_size, num_regions, local_feat_dim)
        # global_average_pool: (batch_size, global_feat_dim)
        # return: bbox_coords (batch_size, num_classes * 4)
        #         bbox_presence (batch_size, num_classes)

        batch_size = local_features.size(0)
        pred_bbox_coords = []
        pred_bbox_presence = []
        
        for i in range(self.num_classes):
            # 1) Project global features
            xg = self.glob_projs[i](global_average_pool) # (batch_size, hidden_dim)
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
        print('BoundingBoxRegressor_v3:')
        print(f'  local_feat_dim: {local_feat_dim}')
        print(f'  global_feat_dim: {global_feat_dim}')
        print(f'  hidden_dim: {hidden_dim}')
        print(f'  num_classes: {num_classes}')
        print(f'  num_regions: {num_regions}')
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











        