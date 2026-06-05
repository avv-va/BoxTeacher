#!/usr/bin/env python
"""
Run a trained BoxVerd model on select images and save mask visualizations.

BoxVerd registers a custom meta-architecture (`MODEL.META_ARCHITECTURE:
BoxVerd`) and custom config keys (`MODEL.BOX_VERD.*`), so the stock detectron2 /
AdelaiDet demo scripts can't load it. This script imports the `boxverd` package
(which registers both) before building the model. Inference itself is unchanged
from CondInst/BoxInst -- at test time BoxVerd just calls the parent forward and
emits per-instance masks (see projects/BoxVerd/boxverd/boxverd.py).

Example (run inside the container; paths are relative to the repo root, which
is the container working dir):

    docker/boxteacher.sh run python projects/BoxVerd/infer.py \
        --config-file output/boxverd_phenobench_r50_1x/config.yaml \
        --weights   output/boxverd_phenobench_r50_1x/model_final.pth \
        --input     datasets/phenobench/images/val/06-05_00223_P0038051.png \
        --output    output/boxverd_phenobench_r50_1x/infer_vis \
        --confidence-threshold 0.3

`--input` accepts one or more image paths, a glob, or a directory. Only files
reachable *inside* the container are visible -- the val images are already
bind-mounted; for arbitrary images, drop them under datasets/ or output/ on the
host (both are mounted) or extend docker/boxteacher.sh with another -v mount.
"""
import argparse
import glob
import os

import cv2
import torch
import tqdm

from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

from adet.config import get_cfg

# Importing the package registers the BoxVerd meta-arch in META_ARCH_REGISTRY
# and exposes add_box_verd_config for the custom MODEL.BOX_VERD.* keys.
from boxverd import add_box_verd_config

# phenobench classes, in label order (see train_net.py register_coco_instances).
THING_CLASSES = ["crop", "weed"]


def setup_cfg(args):
    cfg = get_cfg()
    add_box_verd_config(cfg)
    cfg.merge_from_file(args.config_file)
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    # Score threshold across the detector heads this repo might use.
    thr = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = thr
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = thr
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg


def get_parser():
    p = argparse.ArgumentParser(description="BoxVerd single-image inference")
    p.add_argument("--config-file", required=True,
                   help="trained model's config.yaml (in its OUTPUT_DIR)")
    p.add_argument("--weights", default=None,
                   help="checkpoint to load; defaults to MODEL.WEIGHTS in the config. "
                        "Pass model_final.pth from the OUTPUT_DIR.")
    p.add_argument("--input", nargs="+", required=True,
                   help="image path(s), a glob, or a directory")
    p.add_argument("--output", required=True,
                   help="directory to write visualizations into")
    p.add_argument("--confidence-threshold", type=float, default=0.3,
                   help="minimum score for an instance to be shown")
    return p


def expand_inputs(inputs):
    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        return sorted(os.path.join(inputs[0], f) for f in os.listdir(inputs[0]))
    if len(inputs) == 1:
        hits = glob.glob(os.path.expanduser(inputs[0]))
        assert hits, "no images matched {}".format(inputs[0])
        return sorted(hits)
    return inputs


def main():
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)

    # The demo isn't tied to a registered dataset, so attach class names directly
    # for nicely-labeled visualizations.
    metadata = MetadataCatalog.get("__boxverd_infer__")
    metadata.thing_classes = THING_CLASSES

    os.makedirs(args.output, exist_ok=True)
    paths = expand_inputs(args.input)

    for path in tqdm.tqdm(paths):
        img = read_image(path, format="BGR")            # consistent with eval
        predictions = predictor(img)
        instances = predictions["instances"].to("cpu")
        # Visualizer wants RGB.
        vis = Visualizer(img[:, :, ::-1], metadata=metadata,
                         instance_mode=ColorMode.IMAGE)
        out = vis.draw_instance_predictions(instances)
        out_path = os.path.join(args.output, os.path.basename(path))
        out.save(out_path)
        logger.info("{}: {} instances -> {}".format(
            path, len(instances), out_path))


if __name__ == "__main__":
    main()
