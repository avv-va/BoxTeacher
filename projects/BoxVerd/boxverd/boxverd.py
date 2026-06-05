"""
BoxVerd meta-architecture.

Inherits the CondInst architecture (ResNet+FPN backbone, FCOS proposal generator,
controller-generated dynamic per-instance mask head) and swaps in PoseSegModel's
loss + training style:

  * the per-instance mask head emits two channels (obj0, obj1) and is supervised
    by self-supervised, box-only "ridge" losses (see dynamic_mask_head.py);
  * a "shuffler" consistency augmentation forces the dense mask features to be a
    local function of the image: the image is spatially scrambled, run through
    backbone+mask_branch, then the resulting mask features are descrambled and
    obj0 is supervised on them. Toggle with MODEL.BOX_VERD.SHUFFLE_NUM (0 = off).

Box-only supervision comes from the box rectangle alone (BoxVerd._add_box_bitmasks);
unlike BoxInst there is no MODEL.BOXINST block, no projection/pairwise loss, and no
LAB color-similarity.

Inference is identical to CondInst (single pass, per-instance masks), so the
inference path is delegated to the parent; only the training forward is overridden.
"""
import logging

import torch
from torch import nn

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.structures.instances import Instances

from adet.modeling.condinst.condinst import CondInst

from .dynamic_mask_head import BoxVerdMaskHead
from .shuffler import Shuffler

__all__ = ["BoxVerd"]

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class BoxVerd(CondInst):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Replace CondInst's single-channel mask head with the two-channel
        # BoxVerd head, and rebuild the controller to emit the head's (now larger)
        # number of dynamic parameters.
        self.mask_head = BoxVerdMaskHead(cfg)
        in_channels = self.proposal_generator.in_channels_to_top_module
        self.controller = nn.Conv2d(
            in_channels, self.mask_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

        self.shuffle_num = cfg.MODEL.BOX_VERD.SHUFFLE_NUM

        self.to(self.device)

    def forward(self, batched_inputs):
        # Inference is unchanged from CondInst (single pass, per-instance masks):
        # the parent calls mask_head with 3 args, which hits the head's inference
        # branch (mask_feats_deshuf defaults to None).
        if not self.training:
            return super().forward(batched_inputs)

        original_images = [x["image"].to(self.device) for x in batched_inputs]

        images_norm = [self.normalizer(x) for x in original_images]
        images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)

        features = self.backbone(images_norm.tensor)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        # Build the per-instance box bitmasks (the box rectangle) that the obj0/obj1
        # ridge losses train against. Built at the padded resolution so the
        # mask_out_stride downsample aligns with mask_feats. BoxVerd does this itself
        # rather than via the BoxInst path: no LAB color-similarity is needed.
        self._add_box_bitmasks(
            gt_instances, images_norm.tensor.size(-2), images_norm.tensor.size(-1)
        )

        mask_feats, sem_losses = self.mask_branch(features, gt_instances)

        proposals, proposal_losses = self.proposal_generator(
            images_norm, features, gt_instances, self.controller
        )

        # Shuffler consistency: descrambled-shuffled mask features for obj0.
        mask_feats_deshuf = None
        if self.shuffle_num > 0:
            mask_feats_deshuf = self._shuffled_mask_feats(images_norm.tensor, mask_feats)

        mask_losses = self._forward_mask_heads_train_bv(
            proposals, mask_feats, mask_feats_deshuf, gt_instances
        )

        losses = {}
        losses.update(sem_losses)
        losses.update(proposal_losses)
        losses.update(mask_losses)
        return losses

    def _add_box_bitmasks(self, instances, im_h, im_w):
        """Attach per-instance box bitmasks to each image's gt_instances.

        For every GT box this rasterizes the filled box rectangle at the padded
        image resolution (`gt_bitmasks_full`, used by FCOS center sampling) and on
        the mask_out_stride grid (`gt_bitmasks`, the foreground the obj0/obj1 ridge
        losses are computed over). This is the box-rectangle half of CondInst's
        add_bitmasks_from_boxes; BoxVerd never needs the BoxInst LAB color-similarity,
        so it (and the image-mask / bottom-pixels-removed machinery feeding it) is
        dropped.
        """
        stride = self.mask_out_stride
        start = int(stride // 2)
        for per_im_gt_inst in instances:
            per_im_boxes = per_im_gt_inst.gt_boxes.tensor
            per_im_bitmasks = []
            per_im_bitmasks_full = []
            for per_box in per_im_boxes:
                bitmask_full = torch.zeros((im_h, im_w), device=self.device).float()
                bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0
                bitmask = bitmask_full[start::stride, start::stride]

                assert bitmask.size(0) * stride == im_h
                assert bitmask.size(1) * stride == im_w

                per_im_bitmasks.append(bitmask)
                per_im_bitmasks_full.append(bitmask_full)

            per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
            per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)

    def _shuffled_mask_feats(self, images_tensor, mask_feats):
        """Spatially shuffle the (normalized) image, re-run backbone+mask_branch,
        then descramble the resulting mask features so they align with the
        unshuffled grid. The shuffler is built at the mask-feature resolution and
        scaled up by the mask stride for the image (mirrors PoseSegModel)."""
        out_stride = self.mask_branch.out_stride
        H_m, W_m = mask_feats.shape[-2:]
        assert images_tensor.size(-2) == H_m * out_stride and images_tensor.size(-1) == W_m * out_stride, (
            "image size must be divisible by mask_branch out_stride ({}) for "
            "shuffler alignment; got image {}x{} vs mask feats {}x{}".format(
                out_stride, images_tensor.size(-2), images_tensor.size(-1), H_m, W_m
            )
        )

        shuffler = Shuffler(tile_shape=(H_m, W_m), num_oper=self.shuffle_num)
        img_shuffler = shuffler.scale((out_stride, out_stride))

        shuffled_img = img_shuffler.shuffle(images_tensor)
        feats_shuf = self.backbone(shuffled_img)
        mask_feats_shuf, _ = self.mask_branch(feats_shuf, None)
        mask_feats_deshuf = shuffler.unshuffle(mask_feats_shuf)
        return mask_feats_deshuf

    def _forward_mask_heads_train_bv(self, proposals, mask_feats, mask_feats_deshuf, gt_instances):
        # Same proposal selection as CondInst._forward_mask_heads_train, but the
        # mask head is called with the extra `mask_feats_deshuf` for the obj0 loss.
        pred_instances = proposals["instances"]

        assert (self.max_proposals == -1) or (self.topk_proposals_per_im == -1), \
            "MAX_PROPOSALS and TOPK_PROPOSALS_PER_IM cannot be used at the same time."
        if self.max_proposals != -1:
            if self.max_proposals < len(pred_instances):
                inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
                logger.info("clipping proposals from {} to {}".format(
                    len(pred_instances), self.max_proposals
                ))
                pred_instances = pred_instances[inds[:self.max_proposals]]
        elif self.topk_proposals_per_im != -1:
            num_images = len(gt_instances)

            kept_instances = []
            for im_id in range(num_images):
                instances_per_im = pred_instances[pred_instances.im_inds == im_id]
                if len(instances_per_im) == 0:
                    kept_instances.append(instances_per_im)
                    continue

                unique_gt_inds = instances_per_im.gt_inds.unique()
                num_instances_per_gt = max(int(self.topk_proposals_per_im / len(unique_gt_inds)), 1)

                for gt_ind in unique_gt_inds:
                    instances_per_gt = instances_per_im[instances_per_im.gt_inds == gt_ind]

                    if len(instances_per_gt) > num_instances_per_gt:
                        scores = instances_per_gt.logits_pred.sigmoid().max(dim=1)[0]
                        ctrness_pred = instances_per_gt.ctrness_pred.sigmoid()
                        inds = (scores * ctrness_pred).topk(k=num_instances_per_gt, dim=0)[1]
                        instances_per_gt = instances_per_gt[inds]

                    kept_instances.append(instances_per_gt)

            pred_instances = Instances.cat(kept_instances)

        pred_instances.mask_head_params = pred_instances.top_feats

        loss_mask = self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            pred_instances, gt_instances, mask_feats_deshuf
        )

        return loss_mask
