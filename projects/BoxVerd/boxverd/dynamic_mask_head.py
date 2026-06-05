"""
BoxVerd's per-instance dynamic mask head.

Subclasses AdelaiDet's CondInst DynamicMaskHead, keeping its controller-generated
dynamic-convolution machinery, but:

  * produces TWO output channels per instance (obj0, obj1) instead of one, and
  * replaces BoxInst's projection + pairwise losses with PoseSegModel's
    self-supervised, box-only "ridge" losses (ported from
    ultralytics2/ultralytics/nn/tasks.py, class PoseSegModel).

obj0 loss (calc_object0_loss): inside each instance's box (the foreground), the
network's own confident predictions along the row/column-max "ridge" become
positive targets; they are 1px-dilated and trained with weighted BCE. obj1 loss
(calc_object1_loss): a second channel is trained against soft pseudo-labels
derived from obj0's normalized predictions. Both are dense per-instance maps over
the stride-`mask_out_stride` grid; `gt_bitmasks` (the box rectangle, built by
CondInst.add_bitmasks_from_boxes) plays the role of PoseSegModel's `cls_mask`.

Per the design, PoseSegModel's `_extend_to_all_strides` multi-scale concat is
dropped (per-instance masks are single-scale here), and the shuffler consistency
lives in the meta-arch (boxverd.py): obj0 is evaluated on the descrambled-shuffled
mask features it passes in via `mask_feats_deshuf`.
"""
import torch
from torch import nn
from torch.nn import functional as F

from adet.utils.comm import compute_locations, aligned_bilinear
from adet.modeling.condinst.dynamic_mask_head import DynamicMaskHead


def build_box_verd_mask_head(cfg):
    return BoxVerdMaskHead(cfg)


class BoxVerdMaskHead(DynamicMaskHead):
    def __init__(self, cfg):
        super().__init__(cfg)

        # two per-instance output channels: obj0 (ridge self-supervision) + obj1
        # (pseudo-label-from-obj0). 1 in the parent.
        self.out_channels = 2

        self.warmup_iters = cfg.MODEL.BOX_VERD.WARMUP_ITERS
        self.ridge_thresh = cfg.MODEL.BOX_VERD.RIDGE_THRESH
        self.inflate_weight = cfg.MODEL.BOX_VERD.INFLATE_WEIGHT
        self.seg_weight = cfg.MODEL.BOX_VERD.SEG_WEIGHT
        self.infer_channel = 0 if cfg.MODEL.BOX_VERD.INFER_CHANNEL == "obj0" else 1
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # Recompute dynamic-conv parameter counts: the final layer now emits
        # `out_channels` per instance instead of 1. num_gen_params changes, so the
        # meta-arch's controller is rebuilt to match (see boxverd.py).
        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * self.out_channels)
                bias_nums.append(self.out_channels)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

    # ------------------------------------------------------------------ #
    # dynamic conv with a multi-channel final layer
    # ------------------------------------------------------------------ #
    def _parse_dynamic_params(self, params):
        assert params.dim() == 2
        assert params.size(1) == sum(self.weight_nums) + sum(self.bias_nums)
        num_insts = params.size(0)
        num_layers = len(self.weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, self.weight_nums + self.bias_nums, dim=1
        ))
        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * self.channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * self.channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * self.out_channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * self.out_channels)

        return weight_splits, bias_splits

    def mask_heads_forward_with_coords(self, mask_feats, mask_feat_stride, instances):
        # Mirrors DynamicMaskHead.mask_heads_forward_with_coords but parses
        # multi-channel dynamic params and reshapes to (n_inst, out_channels, H, W).
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = self._parse_dynamic_params(mask_head_params)

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, self.out_channels, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits

    # ------------------------------------------------------------------ #
    # PoseSegModel-style self-supervised losses (per-instance)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _inflate(x):
        """Inflate a binary (N, C, H, W) tensor by one pixel in H and W.

        Verbatim port of PoseSegModel._inflate."""
        inflated = x.clone()
        inflated[:, :, 1:, :] = torch.maximum(inflated[:, :, 1:, :], x[:, :, :-1, :])  # shift down
        inflated[:, :, :-1, :] = torch.maximum(inflated[:, :, :-1, :], x[:, :, 1:, :])  # shift up
        inflated[:, :, :, 1:] = torch.maximum(inflated[:, :, :, 1:], x[:, :, :, :-1])  # shift right
        inflated[:, :, :, :-1] = torch.maximum(inflated[:, :, :, :-1], x[:, :, :, 1:])  # shift left
        return inflated

    def _binary_loss(self, logits, target, weights=None):
        loss_per_pixel = self.bce(logits, target)
        if weights is not None:
            loss_per_pixel = loss_per_pixel * weights
        return loss_per_pixel.mean() * self.seg_weight

    def calc_object0_loss(self, obj0_logits, box_bitmask):
        """obj0: grow each instance's mask from its own confident ridge inside the box.

        Port of PoseSegModel.calc_object0_loss with `cls_mask` -> per-instance
        box bitmask. Targets are derived from detached predictions (self-training)."""
        fg = obj0_logits.detach().sigmoid() * box_bitmask  # N, 1, H, W
        col_max = fg.max(dim=2, keepdim=True).values        # N, 1, 1, W
        row_max = fg.max(dim=3, keepdim=True).values        # N, 1, H, 1
        normalizer = torch.minimum(col_max, row_max) * box_bitmask
        positives = (fg > (self.ridge_thresh * normalizer)).float() * box_bitmask
        positives_inflated = self._inflate(positives) * box_bitmask
        weights = torch.maximum(
            torch.maximum(positives * 1.0, positives_inflated * self.inflate_weight),
            1.0 - box_bitmask,
        )
        return self._binary_loss(obj0_logits, positives_inflated, weights)

    def _object1_pseudo_label(self, obj0_logits, box_bitmask):
        """Normalized soft pseudo-label for obj1 from obj0 (port of
        PoseSegModel.prepare_object1_pseudo_label)."""
        pred = obj0_logits.detach().sigmoid()
        labeled = pred * box_bitmask
        col_max = labeled.max(dim=2, keepdim=True).values
        row_max = labeled.max(dim=3, keepdim=True).values
        normalizer = (torch.minimum(col_max, row_max) + 1e-4) * box_bitmask
        background = 1.0 - box_bitmask
        normalizer = torch.maximum(normalizer, background)
        return labeled / normalizer

    def calc_object1_loss(self, obj1_logits, obj0_logits, box_bitmask):
        """obj1: distill obj0's normalized soft prediction (port of
        PoseSegModel.calc_object1_loss, minus the multi-stride concat)."""
        pseudo_gt = self._object1_pseudo_label(obj0_logits, box_bitmask)
        return self._binary_loss(obj1_logits, pseudo_gt)

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #
    def __call__(self, mask_feats, mask_feat_stride, pred_instances,
                 gt_instances=None, mask_feats_deshuf=None):
        if self.training:
            self._iter += 1

            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            # box rectangle per matched instance -> (n_inst, 1, H, W) foreground
            box_bitmask = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            losses = {}
            if len(pred_instances) == 0:
                dummy = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                losses["loss_obj0"] = dummy
                losses["loss_obj1"] = dummy
                return losses

            mask_logits = self.mask_heads_forward_with_coords(
                mask_feats, mask_feat_stride, pred_instances
            )
            # obj0 is supervised on the descrambled-shuffled features (consistency
            # regularizer); falls back to the unshuffled features when shuffling is off.
            if mask_feats_deshuf is not None:
                mask_logits_deshuf = self.mask_heads_forward_with_coords(
                    mask_feats_deshuf, mask_feat_stride, pred_instances
                )
                obj0_logits = mask_logits_deshuf[:, 0:1]
            else:
                obj0_logits = mask_logits[:, 0:1]
            obj1_logits = mask_logits[:, 1:2]

            if self.warmup_iters <= 0:
                warmup_factor = 1.0
            else:
                warmup_factor = min(self._iter.item() / float(self.warmup_iters), 1.0)

            losses["loss_obj0"] = self.calc_object0_loss(obj0_logits, box_bitmask) * warmup_factor
            losses["loss_obj1"] = self.calc_object1_loss(obj1_logits, obj0_logits, box_bitmask) * warmup_factor
            return losses
        else:
            if len(pred_instances) > 0:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                ch = self.infer_channel
                pred_instances.pred_global_masks = mask_logits[:, ch:ch + 1].sigmoid()

            return pred_instances
