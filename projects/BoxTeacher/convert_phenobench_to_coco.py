#!/usr/bin/env python3
"""Convert the phenobench-yolo dataset to COCO instance-segmentation JSON for
BoxTeacher (box-supervised).

The YOLO labels here are in pose format: `cls cx cy w h kx ky v` (8 fields),
all normalized to [0,1]. Only the first five (cls + box) are used; the keypoint
(kx,ky == center) and visibility are ignored.

BoxTeacher trains with BOXINST.ENABLED=True, which derives masks from the boxes
(adet/modeling/condinst/condinst.py: add_bitmasks_from_boxes). It needs no real
masks. But MODEL.MASK_ON=True means the data mapper calls annotations_to_instances,
which expects a `segmentation` field -- so each box is written as a rectangle
polygon. The polygon shape is never used by the loss; the box is.

Run on the host (PIL is available there):

    python projects/BoxTeacher/convert_phenobench_to_coco.py \
        --yolo-root /home/ava/data/phenobench-yolo \
        --out-dir   datasets/phenobench/annotations

Produces datasets/phenobench/annotations/{train,val}.json. The image_root is set
at registration time, so file_name holds only the basename.
"""
import argparse
import json
import os
from os import path as osp

from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_image(images_dir, stem):
    """Find the image file matching a label stem, trying common extensions."""
    for ext in IMG_EXTS:
        cand = osp.join(images_dir, stem + ext)
        if osp.isfile(cand):
            return cand
    return None


def convert_split(yolo_root, split, categories):
    images_dir = osp.join(yolo_root, "images", split)
    labels_dir = osp.join(yolo_root, "labels", split)
    if not osp.isdir(images_dir) or not osp.isdir(labels_dir):
        raise FileNotFoundError(f"missing images/ or labels/ for split '{split}'")

    coco = {"images": [], "annotations": [], "categories": categories}
    img_id, ann_id = 1, 1
    n_boxes, n_skipped, n_empty = 0, 0, 0

    label_files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".txt"))
    for lf in label_files:
        stem = osp.splitext(lf)[0]
        img_path = find_image(images_dir, stem)
        if img_path is None:
            print(f"  WARN: no image for label {lf}; skipping")
            continue
        with Image.open(img_path) as im:
            W, H = im.size

        coco["images"].append(
            {"id": img_id, "file_name": osp.basename(img_path), "width": W, "height": H}
        )

        rows = 0
        with open(osp.join(labels_dir, lf)) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue  # need at least cls cx cy w h
                cls = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])

                # normalized center/size -> absolute top-left xywh, clipped.
                bw, bh = w * W, h * H
                x = (cx - w / 2.0) * W
                y = (cy - h / 2.0) * H
                x0 = max(0.0, min(x, W))
                y0 = max(0.0, min(y, H))
                x1 = max(0.0, min(x + bw, W))
                y1 = max(0.0, min(y + bh, H))
                bw, bh = x1 - x0, y1 - y0
                if bw <= 1.0 or bh <= 1.0:
                    n_skipped += 1
                    continue

                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls + 1,  # YOLO 0-indexed -> COCO 1-indexed
                        "bbox": [x0, y0, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                        # box as a rectangle polygon (CW): satisfies the loader.
                        "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
                    }
                )
                ann_id += 1
                rows += 1

        n_boxes += rows
        if rows == 0:
            n_empty += 1
        img_id += 1

    print(
        f"  {split}: {len(coco['images'])} images, {n_boxes} boxes, "
        f"{n_empty} empty images, {n_skipped} degenerate boxes skipped"
    )
    return coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-root", default="/home/ava/data/phenobench-yolo")
    ap.add_argument("--out-dir", default="datasets/phenobench/annotations")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    # classes.txt: one class name per line, in YOLO id order.
    with open(osp.join(args.yolo_root, "classes.txt")) as fh:
        names = [ln.strip() for ln in fh if ln.strip()]
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    print(f"categories: {categories}")

    os.makedirs(args.out_dir, exist_ok=True)
    for split in args.splits:
        coco = convert_split(args.yolo_root, split, categories)
        out = osp.join(args.out_dir, f"{split}.json")
        with open(out, "w") as fh:
            json.dump(coco, fh)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
