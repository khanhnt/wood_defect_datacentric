"""Model construction for the locked Faster R-CNN robustness check."""

from __future__ import annotations


def build_model(num_foreground_classes: int = 2, pretrained: bool = True):
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        min_size=1024,
        max_size=1024,
        trainable_backbone_layers=3,
        box_score_thresh=0.001,
        box_nms_thresh=0.7,
        box_detections_per_img=300,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(num_foreground_classes) + 1)
    return model
