# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
BoxVerd training script for the PhenoBench dataset.

Modeled on AdelaiDet's tools/train_net.py: it uses DatasetMapperWithBasis and the
AdetCheckpointer, registers the phenobench_train/phenobench_val datasets (as in
projects/BoxTeacher/train_net.py), adds the BoxVerd config, and imports the
boxverd package so the BoxVerd meta-architecture registers itself. Unlike
BoxTeacher there is no EMA teacher, so no EMA hook is added.

Lives under the live-mounted projects/ tree, so editing it (and the boxverd
package / configs) needs no Docker rebuild.

Usage (inside the container, from the repo root):
    docker/boxteacher.sh run
    python projects/BoxVerd/train_net.py \
        --config-file projects/BoxVerd/configs/boxverd_phenobench_r50_1x.yaml --num-gpus 8
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer

# Importing the package registers the BoxVerd meta-architecture and exposes the
# config function. `projects/BoxVerd` is on sys.path[0] when this script is run.
from boxverd import add_box_verd_config


# --- custom datasets -------------------------------------------------------
# phenobench (box-supervised): boxes from phenobench-yolo, converted to COCO by
# projects/BoxTeacher/convert_phenobench_to_coco.py. Paths are relative to the
# working dir (the repo root). Images are bind-mounted by docker/boxteacher.sh.
_PHENOBENCH = {
    "phenobench_train": (
        "datasets/phenobench/images/train",
        "datasets/phenobench/annotations/train.json",
    ),
    "phenobench_val": (
        "datasets/phenobench/images/val",
        "datasets/phenobench/annotations/val.json",
    ),
}
for _name, (_img_root, _json) in _PHENOBENCH.items():
    if _name not in DatasetCatalog.list():
        register_coco_instances(_name, {"thing_classes": ["crop", "weed"]}, _json, _img_root)


class Trainer(DefaultTrainer):
    """Same as AdelaiDet's Trainer: AdetCheckpointer + DatasetMapperWithBasis."""

    def build_hooks(self):
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
        return ret

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    add_box_verd_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)  # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
