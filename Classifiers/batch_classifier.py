import argparse, os, sys
from itertools import product
import cv2, numpy as np
from Paco_classifier import recognition_engine as recognition

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def main():
    parser = argparse.ArgumentParser(
        description="Run Paco classifier on a folder of images with all model combinations")
    parser.add_argument("--image-dir",          required=True,
                        help="Folder of input images")
    parser.add_argument("--background-models",  required=True, nargs="+",
                        help="One or more background .h5 model paths")
    parser.add_argument("--layer-models",       required=True, nargs="+",
                        help="One or more layer .h5 model paths")
    parser.add_argument("--height",             type=int, default=256)
    parser.add_argument("--width",              type=int, default=256)
    parser.add_argument("--output-dir",         required=True,
                        help="Root directory for output (one subdirectory per combination)")
    args = parser.parse_args()

    image_files = sorted(
        f for f in os.listdir(args.image_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not image_files:
        sys.exit(f"Error: no images found in {args.image_dir!r}")

    combos = list(product(args.background_models, args.layer_models))
    total_combos = len(combos)
    total_images = len(image_files)
    print(f"Found {total_images} image(s), {total_combos} model combination(s) "
          f"-> {total_combos * total_images} total runs\n")

    for combo_idx, (bg_path, layer_path) in enumerate(combos):
        bg_stem = os.path.splitext(os.path.basename(bg_path))[0]
        layer_stem = os.path.splitext(os.path.basename(layer_path))[0]
        if total_combos == 1:
            combo_dir = args.output_dir
        else:
            combo_dir = os.path.join(args.output_dir, f"{bg_stem}__{layer_stem}")
        os.makedirs(combo_dir, exist_ok=True)
        model_paths = [bg_path, layer_path]

        for img_idx, img_file in enumerate(image_files):
            print(f"Combo {combo_idx + 1}/{total_combos} "
                  f"— image {img_idx + 1}/{total_images}: {img_file}")

            image = cv2.imread(os.path.join(args.image_dir, img_file), 1)
            if image is None:
                print(f"  Warning: could not read {img_file!r}, skipping")
                continue

            image_base = os.path.splitext(img_file)[0]
            analyses = recognition.process_image_msae(
                image, model_paths, args.height, args.width, mode='logical')

            for id_label, _ in enumerate(model_paths):
                label_range = np.array(id_label, dtype=np.uint8)
                mask = cv2.inRange(analyses, label_range, label_range)
                masked = cv2.bitwise_and(image, image, mask=mask)
                masked[mask == 0] = (255, 255, 255)
                alpha = np.ones(mask.shape, dtype=mask.dtype) * 255
                alpha[mask == 0] = 0
                b, g, r = cv2.split(masked)
                rgba = cv2.merge((b, g, r, alpha))
                fname = (f'background_layer_{image_base}.png' if id_label == 0
                         else f'layer_{id_label}_{image_base}.png')
                cv2.imwrite(os.path.join(combo_dir, fname), rgba)

    print("\nBatch complete.")


if __name__ == '__main__':
    main()
