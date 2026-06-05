from detectron2.config import CfgNode as CN


def add_box_verd_config(cfg):
    """Config knobs for the BoxVerd meta-architecture (PoseSegModel-style
    self-supervised box-only mask loss on top of CondInst/BoxInst)."""
    cfg.MODEL.BOX_VERD = CN()

    # Number of slice-and-roll operations in the shuffler consistency
    # augmentation (each direction). 0 disables shuffling (single forward pass;
    # obj0 is then supervised on the unshuffled features).
    cfg.MODEL.BOX_VERD.SHUFFLE_NUM = 4

    # Linear warmup (iters) for the self-supervised seg losses. The obj0 ridge
    # target is bootstrapped from the model's own predictions, so it is unstable
    # early; ramp it in. 0 disables warmup.
    cfg.MODEL.BOX_VERD.WARMUP_ITERS = 3000

    # Ridge threshold: a pixel is a positive if its (foreground-masked) score
    # exceeds RIDGE_THRESH * min(row_max, col_max). Port of PoseSegModel's 0.95.
    cfg.MODEL.BOX_VERD.RIDGE_THRESH = 0.95

    # Weight applied to the 1px-dilated (boundary) positives in the obj0 BCE.
    cfg.MODEL.BOX_VERD.INFLATE_WEIGHT = 0.1

    # Overall scale on the segmentation losses (PoseSegModel's `args.seg`).
    cfg.MODEL.BOX_VERD.SEG_WEIGHT = 1.0

    # Which channel to emit as the predicted mask at inference: "obj0" or "obj1".
    cfg.MODEL.BOX_VERD.INFER_CHANNEL = "obj1"
