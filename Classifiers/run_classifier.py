import argparse, os, sys, cv2, numpy as np
from Paco_classifier import recognition_engine as recognition

def main():
    # parse args -> image_path, background_model, layer-models, height, width, threshold
    parser = argparse.ArgumentParser(description="Run Paco classifier on a single image")
    parser.add_argument("--image",            required=True,  help="Path to input image")
    parser.add_argument("--background-model", required=True,  help="Path to background .h5 model")
    parser.add_argument("--layer-models",     required=True,  nargs="+", help="One or more layer .h5 model paths")
    parser.add_argument("--height",           type=int, default=256)
    parser.add_argument("--width",            type=int, default=256)
    parser.add_argument("--threshold",        type=int, default=50)
    parser.add_argument("--output-dir",       default=None,   help="Directory for output PNGs (default: image dir)")
    args = parser.parse_args()

    image = cv2.imread(args.image, 1)
    if image is None:
        sys.exit(f"Error: could not read image at {args.image!r}")

    output_dir = args.output_dir if args.output_dir is not None else os.path.dirname(os.path.abspath(args.image))
    os.makedirs(output_dir, exist_ok=True)

    model_paths = [args.background_model] + args.layer_models

    analyses = recognition.process_image_msae(image, model_paths, args.height, args.width, mode='logical')

    image_base = os.path.splitext(os.path.basename(args.image))[0]
    for id_label, _ in enumerate(model_paths):
        label_range = np.array(id_label, dtype=np.uint8)
        mask = cv2.inRange(analyses, label_range, label_range)

        original_masked = cv2.bitwise_and(image, image, mask=mask)
        original_masked[mask == 0] = (255, 255, 255)

        alpha_channel = np.ones(mask.shape, dtype=mask.dtype) * 255
        alpha_channel[mask == 0] = 0
        b_channel, g_channel, r_channel = cv2.split(original_masked)
        original_masked_alpha = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

        if id_label == 0:
            filename = f'background_{image_base}.png'
        elif id_label == 1:
            filename = f'stafflines_{image_base}.png'
        else:
            filename = f'layer_{id_label}_{image_base}.png'
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, original_masked_alpha)
        print(f"Wrote {output_path}")


if __name__ == '__main__':
    main()