import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

def load_alpha_mask(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path} (resolved: {Path(path).resolve()})")
    return img[:, :, 3] == 255

def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive a background mask as the pixel-wise inverse of the union of "
                     "one or more layer masks. Correct when background means 'everything not "
                     "covered by the given layer(s)' (e.g. a staff-only classifier where "
                     "background = NOT(staff)) -- not appropriate if some pixels should be "
                     "excluded from every mask (partially-annotated, multi-layer datasets)."
    )
    parser.add_argument("--images", nargs="+", required=True,
                        help="Path(s) to source image(s), used only for output file naming/validation")
    parser.add_argument("--layer-masks", nargs="+", required=True,
                        help="Flat list, chunked by len(images) per layer (same convention as calvo_independent_train.py)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write {image_stem}_bg.png files")
    return parser.parse_args()

def main():
    args = parse_args()
    n = len(args.images)
    assert len(args.layer_masks) % n == 0, f"--layer-masks count must be a multiple of {n}"

    layer_mask_groups = [
        args.layer_masks[i*n:(i+1)*n]
        for i in range(len(args.layer_masks)//n)
    ]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for idx in range(n):
        image_stem = Path(args.images[idx]).stem
        masks = [load_alpha_mask(group[idx]) for group in layer_mask_groups]

        union = masks[0]
        for m in masks[1:]:
            assert m.shape == union.shape, f"Layer mask shape mismatch for image #{idx} ({image_stem})"
            union = np.logical_or(union, m)
        background = ~union

        img = cv2.imread(args.images[idx], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {args.images[idx]} (resolved: {Path(args.images[idx]).resolve()})")
        assert img.shape[:2] == background.shape, (
            f"Image/mask shape mismatch for image #{idx} ({image_stem}): "
            f"image is {img.shape[:2]}, mask is {background.shape}"
        )

        # Cutout: original pixels where background, white elsewhere -- matches the
        # ground-truth convention (e.g. inputtest/078bg.png), not just a bare alpha mask.
        alpha = background.astype(np.uint8) * 255
        masked = img.copy()
        masked[alpha == 0] = (255, 255, 255)
        b, g, r = cv2.split(masked)
        rgba = cv2.merge((b, g, r, alpha))

        out_path = str(Path(args.output_dir) / f"{image_stem}_bg.png")
        cv2.imwrite(out_path, rgba)
        print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()