# Faster R-CNN robustness artifacts

Source generation: `access_r1_g3_fasterrcnn`.

The robustness block uses a COCO-pretrained Faster R-CNN with a MobileNetV3-Large
FPN backbone on VNWoodKnot. Baseline, A1 crop, and A2 colour jitter are evaluated at
seeds 42, 43, and 44. Within this block, the detector protocol is fixed and A1/A2 are
the only training-data interventions.

Thresholds are selected on the 75 clean validation images and applied unchanged to
the 75 clean test images. The negative-aware files report epsilon values 0, 0.01,
0.02, and 0.05.

Training code commit: `97a52f1b292e7334942fc46dfef1894a35a589ac`.
Analysis code commit: `d043f249ee6a3c659cb3707a74aac9014838c587`.
