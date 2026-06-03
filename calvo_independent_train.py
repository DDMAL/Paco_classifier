import argparse
import tempfile
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

import isolated_training_sae as training
from Paco_classifier.data_loader import DataContainer, Data, FileSelectionMode, SampleExtractionMode

def load_alpha_mask(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path} (resolved: {Path(path).resolve()})")
    return img[:, :, 3] == 255

def parse_args():
    parser = argparse.ArgumentParser(description="Run Calvo training script on images")
    parser.add_argument("--images",  nargs="+", required=True,  help="Path(s) to source image(s)")
    parser.add_argument("--background-mask", nargs="+", required=True, help="Path(s) to background mask(s) - one bg RGBA PNG per image")
    parser.add_argument("--layer-masks", nargs="+", required=True, help="Flat list, chunked by len(images) per layer")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for .h5 model files")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--max-samples-per-class", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ram-limit", type=float, required=True, help="in GB, passed to DataContainer")
    parser.add_argument("--regions-mask", nargs="+", default=None, help="One selected-regions RGBA PNG per image")
    parser.add_argument("--pretrained-models", nargs="+", default=None,
                        help="Paths to existing .h5 models to fine-tune, one per label (background first, then layers in order)")
    return parser.parse_args()



def main():
    args = parse_args()

    # Validation
    n = len(args.images)
    assert len(args.background_mask) == n, f"--background-mask must have {n} entries"
    assert len(args.layer_masks) % n == 0, f"--layer-masks count must be a multiple of {n}"
    if args.regions_mask:
        assert len(args.regions_mask) == n, f"--regions-mask must have {n} entries"

    # Chunk flat layer-mask list into L groups of N (all pages of layer 1, then layer 2, etc.)
    layer_mask_groups = [
        args.layer_masks[i*n:(i+1)*n]
        for i in range(len(args.layer_masks)//n)
    ]
    num_labels = 1 + len(layer_mask_groups) # background + N layers

    if args.pretrained_models and len(args.pretrained_models) != num_labels:
        print(f"Error: --pretrained-models must have {num_labels} entries (background + {num_labels - 1} layer(s)), got {len(args.pretrained_models)}.", file=sys.stderr)
        sys.exit(1)
    if args.height != 256 or args.width != 256:
        print(f"Warning: height={args.height} width={args.width} - models are typically trained at 256x256.", file=sys.stderr)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tmpdir = tempfile.mkdtemp()
    try:
        # data prep loop
        #sizes[idx] stores (img_nbytes, {label_i: mask_nbytes}) to avoid re-loading .npy for size
        sizes = {}
        for idx in range(n):
            image_stem = Path(args.images[idx]).stem
            img = cv2.imread(args.images[idx], cv2.IMREAD_COLOR)
            np.save(f"{tmpdir}/{image_stem}_image.npy", img)

            regions_mask = load_alpha_mask(args.regions_mask[idx]) if args.regions_mask else None

            masks = [load_alpha_mask(args.background_mask[idx])] + [
                load_alpha_mask(group[idx]) for group in layer_mask_groups
            ]
            _GB = 1024 ** 3
            mask_sizes = {}
            for i, mask in enumerate(masks):
                if regions_mask is not None:
                    mask = np.logical_and(mask, regions_mask)
                np.save(f"{tmpdir}/{image_stem}_label_{i}.npy", mask)
                mask_sizes[i] = mask.nbytes / _GB
            sizes[idx] = (img.nbytes / _GB, mask_sizes)

        # build DataContainer
        # layer_name must be a string key: "0" for bg, "1", "2", etc. for layers
        # getTrain() iterates sorted(inputs.meta.keys()) skipping "Image", so string
        # ints preserve order for reasonable layer counts. 
        container = DataContainer(ram_limit=args.ram_limit)
        for idx in range(n):
            image_stem = Path(args.images[idx]).stem
            img_nbytes, mask_nbytes = sizes[idx]
            x = Data(x_name=image_stem, path=f"{tmpdir}/{image_stem}_image.npy", size=img_nbytes)
            for i in range(num_labels):
                y = Data(x_name=image_stem, path=f"{tmpdir}/{image_stem}_label_{i}.npy", size=mask_nbytes[i])
                container.addXYPair(image_stem, str(i), x, y)
        
        # Train
        output_path = {
            str(i): str(Path(args.output_dir) / f"model_{i}.h5")
            for i in range (num_labels)
        }
        pretrained = {}
        if args.pretrained_models:
            for i, path in enumerate(args.pretrained_models):
                key = "Background Model" if i == 0 else f"Model {i}"
                pretrained[key] = path
        else:
            for i in range(num_labels):
                key = "Background Model" if i == 0 else f"Model {i}"
                candidate = Path(args.output_dir) / f"model_{i}.h5"
                if candidate.exists():
                    print(f"Found existing model for label {i}, fine-tuning: {candidate}")
                    pretrained[key] = str(candidate)
        if not pretrained:
            pretrained = None
        training.train_msae(
            inputs=container,
            num_labels = num_labels,
            height=args.height,
            width=args.width,
            output_path=output_path,
            file_selection_mode=FileSelectionMode.RANDOM,
            sample_extraction_mode=SampleExtractionMode.RANDOM,
            epochs=args.epochs,
            number_samples_per_class=args.max_samples_per_class,
            batch_size=args.batch_size,
            models=pretrained,
        )

        for path in output_path.values():
            print(f"Wrote: {path}")

    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()   
