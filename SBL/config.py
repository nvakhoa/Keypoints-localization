# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN


def add_simplebaseline_config(cfg):
    """
    Add config for SBL.
    """
    cfg.MODEL.SBL = CN()

    # Anchor parameters
    cfg.MODEL.SBL.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]

    # Convolutions to use in the towers
    cfg.MODEL.SBL.NUM_CONVS = 4

    # Number of foreground classes.
    cfg.MODEL.SBL.NUM_CLASSES = 80
    # Channel size for the classification tower
    cfg.MODEL.SBL.CLS_CHANNELS = 256

    cfg.MODEL.SBL.SCORE_THRESH_TEST = 0.05
    # Only the top (1000 * #levels) candidate boxes across all levels are
    # considered jointly during test (to improve speed)
    cfg.MODEL.SBL.TOPK_CANDIDATES_TEST = 6000
    cfg.MODEL.SBL.NMS_THRESH_TEST = 0.5

    # Box parameters
    # Channel size for the box tower
    cfg.MODEL.SBL.BBOX_CHANNELS = 128
    # Weights on (dx, dy, dw, dh)
    cfg.MODEL.SBL.BBOX_REG_WEIGHTS = (1.5, 1.5, 0.75, 0.75)

    # Loss parameters
    cfg.MODEL.SBL.FOCAL_LOSS_GAMMA = 3.0
    cfg.MODEL.SBL.FOCAL_LOSS_ALPHA = 0.3

    # Mask parameters
    # Channel size for the mask tower
    cfg.MODEL.SBL.MASK_CHANNELS = 128
    # Mask loss weight
    cfg.MODEL.SBL.MASK_LOSS_WEIGHT = 2.0
    # weight on positive pixels within the mask
    cfg.MODEL.SBL.POSITIVE_WEIGHT = 1.5
    # Whether to predict in the aligned representation
    cfg.MODEL.SBL.ALIGNED_ON = False
    # Whether to use the bipyramid architecture
    cfg.MODEL.SBL.BIPYRAMID_ON = False