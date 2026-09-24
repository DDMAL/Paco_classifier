"""
Resolution-adaptive scaling helpers for classifier inference.

process_image_msae() slides a fixed 256x256 window across an image. If the
image is scanned at a much higher resolution than the training set, notation
features (staff-line thickness, notehead size) become larger in pixel terms
than what the model saw during training, degrading predictions. These helpers
downscale the image before classification and restore the resulting label map
to the original resolution afterward, so callers see no change in output shape.
"""

import cv2

# Derived from recognition_engine.process_image_msae's padding (25px each side)
# plus the 256x256 window itself: the padded image must be at least as large as
# the window in both dimensions, or the sliding-window loop
# (`range(0, img_height_pad, w_height-padding*2-1)` and
# `min(row, img_height_pad-w_height)`) becomes degenerate/negative.
_PADDING = 25

def _min_safe_dim(w_height, w_width):
    return max(w_height, w_width) - 2 * _PADDING

def compute_scale_ratio(width, height, w_height, w_width,
                        max_dimension=None, ratio=None):
    """
    Resolve a single scalar downscale ratio (never > 1.0 -- we only ever
    shrink, never upscale, since upscaling would invent detail rather than
    fix the scale mismatch).

    - If `ratio` is given, use it directly (clamped to <= 1.0).
    - Elif `max_dimension` is given, ratio = min(1.0, max_dimension / max(width, height)).
    - Else, 1.0 (no-op).

    The ratio is then raised back up if needed so neither resulting dimension
    drops below the window's minimum safe size.
    """
    if ratio is not None:
        scale_ratio = min(1.0, ratio)
    elif max_dimension is not None:
        scale_ratio = min(1.0, max_dimension / float(max(width, height)))
    else:
        scale_ratio = 1.0

    if scale_ratio < 1.0:
        min_safe = _min_safe_dim(w_height, w_width)
        floor_ratio = min_safe / float(min(width, height))
        if scale_ratio < floor_ratio:
            print(f"Requested scale ratio {scale_ratio:.4f} would shrink the "
                  f"image below the classifier's minimum safe size; raising "
                  f"to {floor_ratio:.4f}")
            scale_ratio = min(1.0, floor_ratio)

    return scale_ratio

def resize_image_down(image, ratio):
    """Downscale `image` by `ratio` (uniform on both axes, aspect-preserving).
    Uses INTER_AREA, the recommended cv2 filter for shrinking images."""
    return cv2.resize(image, None, fx=ratio, fy=ratio,
                       interpolation=cv2.INTER_AREA)

def restore_label_map(label_map, target_width, target_height):
    """Resize a categorical label map back to (target_width, target_height).
    Uses INTER_NEAREST -- label values are discrete integers, so any other
    interpolation would invent invalid intermediate label values."""
    return cv2.resize(label_map, (target_width, target_height),
                       interpolation=cv2.INTER_NEAREST)
