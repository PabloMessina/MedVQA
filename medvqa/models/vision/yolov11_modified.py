from typing import List
from types import SimpleNamespace
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression
from ultralytics.nn.tasks import DetectionModel
from medvqa.models.FiLM_utils import LinearFiLM
from medvqa.models.common import set_inplace_flag
from medvqa.models.mlp import MLP
from medvqa.losses.yolov11_custom_loss import YOLOv11CustomLoss

_DEFAULT_BOX_GAIN = 7.5
_DEFAULT_CLS_GAIN = 0.5
_DEFAULT_DFL_GAIN = 1.5

class ClassificationTaskDescriptor:
    def __init__(self, task_name: str, label_names: List[str], class_names: List[str]):
        """
        Initializes a descriptor for a classification task.

        Args:
            task_name (str): Name of the classification task.
            label_names (List[str]): List of label names.
            num_labels (int): Number of labels.
        """
        self.task_name = task_name
        self.label_names = label_names
        self.num_labels = len(label_names)
        self.class_names = class_names
        self.num_classes = len(class_names)
        assert self.num_labels > 0, "Number of labels must be greater than 0."
        assert self.num_classes > 0, "Number of classes must be greater than 0."

class DetectionTaskDescriptor:
    def __init__(self, task_name: str, class_names: List[str]):
        """
        Initializes a descriptor for a detection task.

        Args:
            task_name (str): Name of the detection task.
            label_names (List[str]): List of label names.
            num_labels (int): Number of labels.
        """
        self.task_name = task_name
        self.class_names = class_names
        self.num_classes = len(class_names)
        assert self.num_classes > 0, "Number of classes must be greater than 0."

def get_feature_dimensions(model_path: str, feature_layers: List[str], image_size: int):
    """Helper function to get feature dimensions for initialization"""
    temp_model = YOLO(model_path).model
    dummy_input = torch.randn(1, 3, image_size, image_size)
    feature_dims = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            feature_dims[name] = output.shape
        return hook
    
    hooks = []
    for name, module in temp_model.named_modules():
        if name in feature_layers:
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    temp_model(dummy_input)
    
    for hook in hooks:
        hook.remove()

    feature_dims = [feature_dims[layer] for layer in feature_layers]
    return feature_dims

def _apply_film_and_pool_features(features: List[torch.Tensor], film_layers: nn.ModuleList,
                            local_attention_hidden_layers: nn.ModuleList, local_attention_final_layers: nn.ModuleList,
                            query_embed: torch.Tensor, return_attention_maps: bool = False,
                            return_features_after_film: bool = False):
    """
    Applies FiLM to features.

    Args:
        features (List[torch.Tensor]): List of input features.
        film_layers (nn.ModuleList): List of FiLM layers.
        local_attention_hidden_layers (nn.ModuleList): List of hidden layers for local attention.
        local_attention_final_layers (nn.ModuleList): List of final layers for local attention.
        query_embed (torch.Tensor): Query embedding tensor of shape (B, Q, E).
        return_attention_maps (bool): Whether to return attention maps.
        return_features_after_film (bool): Whether to return features after FiLM.

    Returns:
        List[torch.Tensor] | Dict[str, List[torch.Tensor]]: List of output tensors or dictionary of outputs.
    """
    Q = query_embed.shape[1] # Number of queries
    attention_pooled_features = []
    if return_attention_maps:
        attention_maps = []
    if return_features_after_film:
        features_after_film = []

    # Process each feature layer
    for i, feature in enumerate(features):
        
        # Before FiLM, feature.shape = (B, C, H, W)
        H = feature.shape[2]
        W = feature.shape[3]
        feature = feature.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        feature = feature.unsqueeze(1).expand(-1, Q, -1, -1)  # (B, Q, H*W, C)
        query_embed_ = query_embed.unsqueeze(2).expand(-1, -1, H*W, -1)  # (B, Q, H*W, E)

        # Apply FiLM
        feature_after_film = film_layers[i](feature, query_embed_)  # (B, Q, H*W, C)

        # Apply local attention
        hidden = F.gelu(local_attention_hidden_layers[i](feature_after_film)) # (B, Q, H*W, hidden_size)
        attention = torch.sigmoid(local_attention_final_layers[i](hidden)) # (B, Q, H*W, 1)
        attended = attention * feature_after_film # (B, Q, H*W, C)
        attended = attended.sum(dim=2) # (B, Q, C)
        attended /= (attention.sum(dim=2) + 1e-8) # Normalize

        # Collect pooled features
        attention_pooled_features.append(attended) # (B, Q, C)

        # Collect attention maps
        if return_attention_maps:
            attention_maps.append(attention.squeeze(-1)) # (B, Q, H*W)

        # Collect features after FiLM
        if return_features_after_film:
            features_after_film.append(feature_after_film) # (B, Q, H*W, C)
            
    output = {}
    output["attention_pooled_features"] = attention_pooled_features
    if return_attention_maps:
        output["attention_maps"] = attention_maps
    if return_features_after_film:
        output["features_after_film"] = features_after_film
    if len(output) == 1:
        output = output["attention_pooled_features"]
    return output

class ClassificationHead(nn.Module):
    def __init__(self, in_feat_dims: List[tuple],
                 num_labels: int, num_classes: int, query_embed_size: int,
                 mlp_hidden_dims: List[int]):
        """
        Initializes a classification head.

        Args:
            in_feat_dims (List[tuple]): List of input feature dimensions.
            num_labels (int): Number of labels.
            num_classes (int): Number of classes.
            query_embed_size (int): Size of the query embedding.
            mlp_hidden_dims (List[int]): List of hidden dimensions in the classification MLP.
        """
        super(ClassificationHead, self).__init__()

        self.num_classes = num_classes
        
        # Create a query embedding matrix
        self.query_embed = nn.Embedding(num_labels, query_embed_size)

        # Create classification head
        self.classify = MLP(
            in_dim=sum(x[1] for x in in_feat_dims), # Sum of all channel dimensions
            hidden_dims=mlp_hidden_dims,
            out_dim=num_classes,
            activation=nn.GELU,
        )        

    def forward(self, features: List[torch.Tensor],
                film_layers: nn.ModuleList, local_attention_hidden_layers: nn.ModuleList,
                local_attention_final_layers: nn.ModuleList,
                label_ids: torch.Tensor = None):
        """
        Forward pass of the classification head.

        Args:
            features (List[torch.Tensor]): List of input features.
            film_layers (nn.ModuleList): List of FiLM layers.
            local_attention_hidden_layers (nn.ModuleList): List of hidden layers for local attention.
            local_attention_final_layers (nn.ModuleList): List of final layers for local attention.            
            label_ids (torch.Tensor): Tensor of label IDs. If None, use all labels.

        Returns:
            List[torch.Tensor]: List of output logits.
        """
        # Get query embedding
        query_embed = self.query_embed(label_ids) if label_ids is not None else\
                      self.query_embed.weight # (num_labels, query_embed_size)

        # Apply FiLM to features
        attention_pooled_features = _apply_film_and_pool_features(
            features=features,
            film_layers=film_layers,
            local_attention_hidden_layers=local_attention_hidden_layers,
            local_attention_final_layers=local_attention_final_layers,
            query_embed=query_embed,
        )

        # Classify
        pooled_features = torch.cat(attention_pooled_features, dim=2) # (B, num_labels, sum(C))
        logits = self.classify(pooled_features) # (B, num_labels, num_classes)
        if self.num_classes == 1:
            logits = logits.squeeze(-1) # (B, num_labels) -> we are doing binary classification

        return logits
    
def _create_detection_head(original_cfg: dict, yolo_model: nn.Module, image_size: int, num_classes: int, verbose: bool = True):
    """
    Creates a new detection head.

    Args:
        original_cfg (dict): Original configuration of the YOLO model.
        yolo_model (nn.Module): YOLO model.
        image_size (int): Size of the input image.
        num_classes (int): Number of classes.
        verbose (bool): Whether to print verbose information.

    Returns:
        nn.Module: Detection head module.
    """
    cfg = copy.deepcopy(original_cfg)
    cfg['imgsz'] = image_size # Set image size
    cfg['inplace'] = False # Turn off inplace operations because we don't want to modify the input features
    tmp_model = DetectionModel(cfg, ch=3, nc=num_classes, verbose=verbose)
    tmp_model.load(yolo_model) # Load weights
    set_inplace_flag(tmp_model.model[-1], False) # Turn off inplace operations
    return tmp_model.model[-1] # Return the detection head

class YOLOv11MultiClassifierDetector(nn.Module):
    def __init__(self,
                 classification_tasks: List[ClassificationTaskDescriptor],
                 detection_tasks: List[DetectionTaskDescriptor],
                 query_embed_size: int,
                 mlp_hidden_dims: List[int],
                 local_attention_hidden_size: int,
                 image_size: int,
                 model_name_or_path: str,
                 alias: str,
                 device: torch.device):
        """
        Initializes a YOLOv11MultiClassifierDetector model.

        Args:
            classification_tasks (List[ClassificationTaskDescriptor]): List of classification tasks.
            detection_tasks (List[DetectionTaskDescriptor]): List of detection tasks.
            query_embed_size (int): Size of the query embedding.
            mlp_hidden_dims (List[int]): List of hidden dimensions in the classification MLP.
            local_attention_hidden_size (int): Size of the hidden layer in local attention.
            image_size (int): Size of the input image.
            model_name_or_path (str): Name or path of the model to load as a starting point.
            alias (str): Alias for the model.
            device (torch.device): Device to use for the model (required by YOLOv11CustomLoss).
        """
        super(YOLOv11MultiClassifierDetector, self).__init__()
        self.classification_tasks = classification_tasks
        self.detection_tasks = detection_tasks
        self.num_classification_tasks = len(classification_tasks)
        self.num_detection_tasks = len(detection_tasks)
        self.alias = alias
        assert self.num_classification_tasks > 0 or self.num_detection_tasks > 0, "At least one task must be specified."

        # Load model
        yolo = YOLO(model=model_name_or_path, task='detect')
        self.yolo_model = yolo.model
        self.original_detect = self.yolo_model.model[-1]  # Get original detection head
        self.original_args = self.yolo_model.args
        self.original_cfg = self.yolo_model.yaml
        if 'box' not in self.original_args:
            self.original_args['box'] = _DEFAULT_BOX_GAIN
        if 'cls' not in self.original_args:
            self.original_args['cls'] = _DEFAULT_CLS_GAIN
        if 'dfl' not in self.original_args:
            self.original_args['dfl'] = _DEFAULT_DFL_GAIN
        
        if self.num_detection_tasks > 0:
            # Create new detection heads
            self.detect = nn.ModuleDict()
            for task in detection_tasks:
                self.detect[task.task_name] = _create_detection_head(original_cfg=self.original_cfg,
                                                                    yolo_model=self.yolo_model,
                                                                    image_size=image_size,
                                                                    num_classes=task.num_classes,
                                                                    verbose=True)
                
            # Create YOLO loss instances, one for each detection task
            self.yolo_loss_dict = {}
            hyp = SimpleNamespace(**self.original_args) # HACK: Create a SimpleNamespace from the original args to make it work
            for task in detection_tasks:
                self.yolo_loss_dict[task.task_name] = YOLOv11CustomLoss(detect_module=self.detect[task.task_name],
                                                                        hyp=hyp, device=device)
                
        if self.num_classification_tasks > 0:
            # Get feature dimensions for initialization
            f_layers = [f'model.{x}' for x in self.original_detect.f] # Get feature layers that are used in the detection head
            feature_dims = get_feature_dimensions(model_name_or_path, f_layers, image_size)
            print('feature_dims:', feature_dims) # list of shapes (1, C, H, W)
            
            # Create FiLM, local attention, and classification layers
            self.film_layers = nn.ModuleList()
            self.local_attention_hidden_layers = nn.ModuleList()
            self.local_attention_final_layers = nn.ModuleList()
            for shape in feature_dims:
                in_dim = shape[1] # Get the channel dimension
                self.film_layers.append(LinearFiLM(in_dim=in_dim, condition_dim=query_embed_size))
                self.local_attention_hidden_layers.append(nn.Linear(in_dim, local_attention_hidden_size))
                self.local_attention_final_layers.append(nn.Linear(local_attention_hidden_size, 1))

            # Create new classification heads
            self.classify = nn.ModuleDict()
            for task in classification_tasks:
                self.classify[task.task_name] = ClassificationHead(
                    in_feat_dims=feature_dims,
                    num_labels=task.num_labels,
                    num_classes=task.num_classes if task.num_classes >= 3 else 1, # If num_classes < 3, do binary classification
                    query_embed_size=query_embed_size,
                    mlp_hidden_dims=mlp_hidden_dims,                
                )

        # Remove the original detection head
        self.yolo_model.model = self.yolo_model.model[:-1]
    
    def forward(self, x: torch.Tensor,
                detection_task_names: List[str]|str = None,
                classification_task_names: List[str]|str = None,
                label_ids: torch.Tensor = None,
                conf_thres: float = 0.1,
                iou_thres: float = 0.1,
                max_det: int = 40,
                batch: dict = None,
                apply_nms: bool = True):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.
            detection_task_names (List[str] or str): List of detection task names.
            classification_task_names (List[str] or str): List of classification task names.
            label_ids (torch.Tensor): Tensor of label IDs. If None, use all labels.
            conf_thres (float): Confidence threshold for non-maximum suppression.
            iou_thres (float): IoU threshold for non-maximum suppression.
            max_det (int): Maximum number of detections to keep.
            batch (dict): Batch dictionary. Necessary to compute loss.
            apply_nms (bool): Whether to apply non-maximum suppression.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of outputs for each task.
        """
        assert (detection_task_names is not None and len(detection_task_names) > 0) or\
               (classification_task_names is not None and len(classification_task_names) > 0),\
               "At least one task must be specified."
        
        if self.training and detection_task_names is not None:
            assert batch is not None, "Batch dictionary must be provided during training for detection tasks."

        B = x.shape[0] # Batch size
        H = x.shape[2] # Height
        W = x.shape[3] # Width

        # Run modified forward pass of YOLOv11
        # (See original forward pass in ultralytics.nn.tasks.py > BaseModel._predict_once)
        y = [] # outputs
        save = self.yolo_model.save
        for m in self.yolo_model.model:
            if m.f != -1: # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x) # run
            y.append(x if m.i in save else None) # save output

        # Get features used in detection head
        features = [y[j] for j in self.original_detect.f]

        outputs = {}

        # Get detection outputs
        if detection_task_names:
            if isinstance(detection_task_names, str):
                detection_task_names = [detection_task_names]
            detect_outputs = {}
            for task_name in detection_task_names:
                features_ = [f for f in features] # Shallow copy to avoid modifying the original feature list (needed to fix a bug)
                preds = self.detect[task_name](features_)
                if self.training:
                    # Compute loss
                    loss, loss_items = self.yolo_loss_dict[task_name](preds, batch)
                    loss /= B
                    detect_outputs[task_name] = {"preds": preds, "loss": loss, "loss_items": loss_items}
                else:
                    # Apply non-maximum suppression
                    assert isinstance(preds, tuple) and len(preds) == 2
                    if apply_nms:
                        preds = non_max_suppression(preds[0], conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
                        # normalize coordinates
                        for pred in preds:
                            pred[:, 0] /= W # x1
                            pred[:, 1] /= H # y1
                            pred[:, 2] /= W # x2
                            pred[:, 3] /= H # y2
                    else:
                        preds = preds[0] # Get raw predictions
                    detect_outputs[task_name] = preds

            outputs["detection"] = detect_outputs

        # Get classification outputs
        if classification_task_names:
            if isinstance(classification_task_names, str):
                classification_task_names = [classification_task_names]
            classify_outputs = {}
            for task_name in classification_task_names:
                classify_outputs[task_name] = self.classify[task_name](features,
                                                                    self.film_layers,
                                                                    self.local_attention_hidden_layers,
                                                                    self.local_attention_final_layers,
                                                                    label_ids)
            outputs["classification"] = classify_outputs

        return outputs
    
    def __str__(self) -> str:
        classif_task_names = [task.task_name for task in self.classification_tasks]
        detect_task_names = [task.task_name for task in self.detection_tasks]
        classif_task_str = ",".join(classif_task_names)
        detect_task_str = ",".join(detect_task_names)
        tasks_strs = []
        if len(classif_task_names) > 0:
            tasks_strs.append(f"c:{classif_task_str}")
        if len(detect_task_names) > 0:
            tasks_strs.append(f"d:{detect_task_str}")
        tasks_str = ";".join(tasks_strs)
        return f"{self.alias}({tasks_str})"


class YOLOv11FactConditionedClassifierDetector(nn.Module):
    def __init__(self,
                 fact_embed_size: int,
                 mlp_hidden_dims: List[int],
                 local_attention_hidden_size: int,
                 image_size: int,
                 model_name_or_path: str,
                 yolo_alias: str,
                 device: torch.device):
        """
        Initializes a YOLOv11FactConditionedClassifierDetector model.

        Args:
            fact_embed_size (int): Size of the fact embedding.
            mlp_hidden_dims (List[int]): List of hidden dimensions in the classification MLP.
            local_attention_hidden_size (int): Size of the hidden layer in local attention.
            image_size (int): Size of the input image.
            model_name_or_path (str): Name or path of the model to load as a starting point.
            yolo_alias (str): Alias for the model.
            device (torch.device): Device to use for the model (required by YOLOv11CustomLoss).
        """
        super(YOLOv11FactConditionedClassifierDetector, self).__init__()
        self.yolo_alias = yolo_alias
        self.fact_embed_size = fact_embed_size
        self.mlp_hidden_dims = mlp_hidden_dims
        self.local_attention_hidden_size = local_attention_hidden_size

        # Load model
        yolo = YOLO(model=model_name_or_path, task='detect')
        self.yolo_model = yolo.model
        self.original_detect = self.yolo_model.model[-1]  # Get original detection head
        self.original_args = self.yolo_model.args
        self.original_cfg = self.yolo_model.yaml
        if 'box' not in self.original_args:
            self.original_args['box'] = _DEFAULT_BOX_GAIN
        if 'cls' not in self.original_args:
            self.original_args['cls'] = _DEFAULT_CLS_GAIN
        if 'dfl' not in self.original_args:
            self.original_args['dfl'] = _DEFAULT_DFL_GAIN
                
        # Create new detection head
        self.detect = _create_detection_head(original_cfg=self.original_cfg,
                                            yolo_model=self.yolo_model,
                                            image_size=image_size,
                                            num_classes=1, # Each fact is a single class
                                            verbose=True)
        # Create YOLO loss instance
        hyp = SimpleNamespace(**self.original_args)
        self.yolo_loss = YOLOv11CustomLoss(detect_module=self.detect, hyp=hyp, device=device)
        
        # Get feature dimensions for initialization
        f_layers = [f'model.{x}' for x in self.original_detect.f] # Get feature layers that are used in the detection head
        feature_dims = get_feature_dimensions(model_name_or_path, f_layers, image_size)
        print('feature_dims:', feature_dims) # list of shapes (1, C, H, W)
        
        # Create FiLM, local attention, and classification layers
        self.film_layers = nn.ModuleList()
        self.local_attention_hidden_layers = nn.ModuleList()
        self.local_attention_final_layers = nn.ModuleList()
        for shape in feature_dims:
            in_dim = shape[1] # Get the channel dimension
            self.film_layers.append(LinearFiLM(in_dim=in_dim, condition_dim=fact_embed_size))
            self.local_attention_hidden_layers.append(nn.Linear(in_dim, local_attention_hidden_size))
            self.local_attention_final_layers.append(nn.Linear(local_attention_hidden_size, 1))

        # Create classification head
        self.mlp_classifier = MLP(
            in_dim=(sum(x[1] for x in feature_dims) # Sum of all channel dimensions
                    + feature_dims[-1][2] * feature_dims[-1][3]), # Add the last feature's spatial dimensions
            hidden_dims=mlp_hidden_dims,
            out_dim=1, # Binary classification
            activation=nn.GELU,
        )

        # Remove the original detection head
        self.yolo_model.model = self.yolo_model.model[:-1]
    
    def forward(self,
                images: torch.Tensor,
                fact_embeddings: torch.Tensor,
                apply_nms: bool = True,
                conf_thres: float = 0.1,
                iou_thres: float = 0.1,
                max_det: int = 10,
                batch: dict = None,
                use_first_n_facts_for_detection: int = None):
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): Input image tensor of shape (B, C, H, W).
            fact_embeddings (torch.Tensor): Tensor of fact embeddings of shape (B, F, E).
            apply_nms (bool): Whether to apply non-maximum suppression.
            conf_thres (float): Confidence threshold for non-maximum suppression.
            iou_thres (float): IoU threshold for non-maximum suppression.
            max_det (int): Maximum number of detections to keep.
            batch (dict): Batch dictionary. Necessary to compute loss during training.
            use_first_n_facts_for_detection (int): Number of facts to use for detection. If None, use all facts.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of outputs for each task.
        """
        
        if self.training:
            assert batch is not None, "Batch dictionary must be provided during training for detection."

        B = images.shape[0] # Batch size
        F = fact_embeddings.shape[1] # Number of facts
        if use_first_n_facts_for_detection is not None:
            assert 1 <= use_first_n_facts_for_detection <= F, "Invalid value for use_first_n_facts_for_detection."

        # Run modified forward pass of YOLOv11
        # (See original forward pass in ultralytics.nn.tasks.py > BaseModel._predict_once)
        y = [] # outputs
        save = self.yolo_model.save
        x = images # Input image tensor, shape (B, C, H, W)
        for m in self.yolo_model.model:
            if m.f != -1: # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x) # run
            y.append(x if m.i in save else None) # save output

        # Get features used in detection head
        features = [y[j] for j in self.original_detect.f] # list of shape (B, C, H, W)

        # Apply FiLM and local attention to features
        tmp = _apply_film_and_pool_features(
            features=features,
            film_layers=self.film_layers,
            local_attention_hidden_layers=self.local_attention_hidden_layers,
            local_attention_final_layers=self.local_attention_final_layers,
            query_embed=fact_embeddings,
            return_attention_maps=True,
            return_features_after_film=True,
        )
        attention_pooled_features = tmp["attention_pooled_features"] # list of shape (B, F, C)
        attention_maps = tmp["attention_maps"] # list of shape (B, F, H*W)
        features_after_film = tmp["features_after_film"] # list of shape (B, F, H*W, C)

        outputs = {}

        # ---- Get classification output ----
        # 1. Concatenate pooled features and attention maps
        concat_list = attention_pooled_features + [attention_maps[-1]] # all features + last attention map
        concat_features = torch.cat(concat_list, dim=2) # (B, F, sum(C) + H*W)
        # 2. Classify
        logits = self.mlp_classifier(concat_features) # (B, F, 1)
        logits = logits.squeeze(-1) # (B, F)
        assert logits.shape == (B, F)
        outputs["classification_logits"] = logits

        # ---- Get detection output ----
        # 0. Get the number of facts to use for detection
        if use_first_n_facts_for_detection is not None:
            F = use_first_n_facts_for_detection
            features_after_film = [feat[:, :F, :, :] for feat in features_after_film]
            
        # 1. Adapt features after FiLM for detection, from (B, F, H*W, C) to (B*F, C, H, W)
        features_after_film_for_det = []
        for i in range(len(features_after_film)):
            C = features[i].shape[1]
            H = features[i].shape[2]
            W = features[i].shape[3]
            assert features_after_film[i].shape == (B, F, H*W, C)
            feat = features_after_film[i]
            feat = feat.reshape(B*F, H, W, C).permute(0, 3, 1, 2) # (B*F, C, H, W)
            features_after_film_for_det.append(feat) # list of shape (B*F, C, H, W)
        
        # 2. Run detection head
        preds = self.detect(features_after_film_for_det)

        # 3. Process detection outputs
        if self.training:
            assert isinstance(preds, list) and len(preds) == 3
            assert preds[0].shape[0] == B*F
            # Compute loss
            loss, loss_items = self.yolo_loss(preds, batch)
            loss /= B*F # Normalize by "amplified" batch size
            detect_outputs = {"preds": preds, "loss": loss, "loss_items": loss_items}
        else:
            # Apply non-maximum suppression
            assert isinstance(preds, tuple) and len(preds) == 2
            if apply_nms:
                preds = non_max_suppression(preds[0], conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
                assert isinstance(preds, list) and len(preds) == B*F
                # normalize coordinates
                W = images.shape[3]
                H = images.shape[2]
                for pred in preds:
                    pred[:, 0] /= W # x1
                    pred[:, 1] /= H # y1
                    pred[:, 2] /= W # x2
                    pred[:, 3] /= H # y2
                # convert to lists (B) of lists (F) of tensors
                preds = [[preds[i*F + j] for j in range(F)] for i in range(B)]
            else:
                preds = preds[0] # Get raw predictions
            detect_outputs = preds

        outputs["detection"] = detect_outputs

        return outputs
    
    def __str__(self) -> str:
        dims_str = f'{self.fact_embed_size},{self.mlp_hidden_dims},{self.local_attention_hidden_size}'
        dims_str = dims_str.replace(' ', '')
        return f'{self.yolo_alias}(fact_cond,cls,det,{dims_str})'
    

class YOLOv11FeatureExtractor(nn.Module):
    def __init__(self, model_name_or_path: str, yolo_alias: str):
        """
        Initializes a YOLOv11FeatureExtractor model.

        Args:
            model_name_or_path (str): Name or path of the model to load as a starting point.
            yolo_alias (str): Alias for the model.
        """
        super(YOLOv11FeatureExtractor, self).__init__()
        self.yolo_alias = yolo_alias

        # Load model
        yolo = YOLO(model=model_name_or_path, task='detect')
        self.yolo_model = yolo.model
        self.original_detect = self.yolo_model.model[-1]  # Get original detection head
        self.yolo_model.model = self.yolo_model.model[:-1] # Remove the detection head
    
    def forward(self, images: torch.Tensor):
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            List[torch.Tensor]: List of output features.
        """

        # Run modified forward pass of YOLOv11
        # (See original forward pass in ultralytics.nn.tasks.py > BaseModel._predict_once)
        y = [] # outputs
        save = self.yolo_model.save
        x = images # Input image tensor, shape (B, C, H, W)
        for m in self.yolo_model.model:
            if m.f != -1: # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x) # run
            y.append(x if m.i in save else None) # save output

        # Get features used in original detection head
        features = [y[j] for j in self.original_detect.f] # list of shape (B, C, H, W)

        # Return features
        return features
    
    def __str__(self) -> str:
        return f'YOLOv11FeatureExtractor({self.yolo_alias})'