from __future__ import division
from Paco_classifier import image_scaling

import os
import threading

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import image_data_format


# Loaded .h5 models, keyed by (path, mtime_ns, size) so a swapped-in
# retrained checkpoint is picked up instead of silently serving the old
# weights. process_image_msae() used to call load_model() on BOTH models on
# every single call, i.e. once per page for a server that classifies pages
# one at a time -- measured at ~4.3s on a cold page cache and ~0.35s warm.
#
# _MODEL_CACHE_LOCK guards the CACHE and the load itself, so two threads
# racing on the same page size can't both pay for the same deserialize.
# Inference is deliberately NOT serialized: `model(x, training=False)` on a
# functional Keras model reads the weights and mutates nothing on the model
# object (unlike, say, kraken's TorchSeqRecognizer, which stashes per-call
# state on itself), so concurrent callers can share one instance -- which is
# the ordinary way Keras models are served.
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()


def load_model_cached(model_path):
    """load_model(), memoized on the file's identity + mtime + size."""
    try:
        st = os.stat(model_path)
        key = (str(model_path), st.st_mtime_ns, st.st_size)
    except OSError:
        # Unstattable path: fall back to loading every time rather than
        # caching under a key that can't detect a change.
        return load_model(model_path)
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            model = load_model(model_path)
            # One entry per distinct weights file. The caller passes a fixed
            # pair of paths, so this never grows unboundedly in practice.
            _MODEL_CACHE[key] = model
        return model


class ClassificationCancelled(Exception):
    """Raised by process_image_msae() when its should_cancel callback
    reports True. Lets a caller running this on a background thread (e.g.
    a server request whose client disconnected) stop the sliding-window
    pass between patches instead of paying for the whole page regardless."""


def process_image(image, model_path, vspan, hspan):
    """
    Takes a document image and a pre-trained model path
    and returns the process image (with logical labels).
    """

    model = load_model(model_path)

    [height, width, channels] = image.shape

    output = np.zeros((height, width), 'uint8')

    for row in range(vspan, height-vspan-1):
        print(str(row) + ' / ' + str(height - vspan - 1))
        for col in range(hspan, width-hspan-1):
            sample = image[row-vspan:row+vspan+1, col-hspan:col+hspan+1]

            if image_data_format() == 'channels_first':
                sample = np.asarray(sample).reshape(1, 3, vspan*2 + 1, hspan*2 + 1)
            else:
                sample = np.asarray(sample).reshape(1, vspan*2 + 1, hspan*2 + 1, 3)

            prediction = model.predict(sample)[0]
            label = np.argmax(prediction)

            output[row][col] = label

    return output


def process_image_msae(image, model_paths, w_height, w_width, mode='masks',
                       resize_ratio=None, max_dimension=None, should_cancel=None,
                       progress_callback=None):
    """
    Takes a document image and pre-trained SAE model paths
    and returns a single image with logical labels.

    should_cancel, if given, is a zero-arg callable polled once per patch
    (each row/col step of the sliding window below) -- if it ever returns
    True, raises ClassificationCancelled immediately instead of finishing
    the rest of the page. Optional and defaults to None (no polling, same
    behavior as before this parameter existed) so every existing caller
    keeps working unchanged.

    progress_callback, if given, is called as progress_callback(row, total)
    once per sliding-window row step (mirrors should_cancel's shape) --
    `row` is exactly the pixel offset already printed below ("N / total"),
    `total` is img_height_pad. Optional and defaults to None so every
    existing caller keeps working unchanged. Lets a caller running this on
    a server (paco-classifier-service) relay real progress to its own
    caller instead of only the bare "N / total" console print this
    function already did.
    """

    num_labels = len(model_paths)
    padding = 25

    orig_height, orig_width = image.shape[:2]
    scale_ratio = image_scaling.compute_scale_ratio(
        orig_width, orig_height, w_height, w_width,
        max_dimension=max_dimension, ratio=resize_ratio)
    if scale_ratio < 1.0:
        image = image_scaling.resize_image_down(image, scale_ratio)
        print(f"Resizing input {orig_width}x{orig_height} -> "
              f"{image.shape[1]}x{image.shape[0]} (ratio={scale_ratio:.4f}) "
              f"before classification")
        
    #Including padding at the edges due to the unreliability of the model's predictions along the borders.
    image_with_padding = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    [img_height_pad, img_width_pad, channels_pad] = image_with_padding.shape

    sae_models = []
    for id_label in range(num_labels):
        sae_models.append(load_model_cached(model_paths[id_label]))

    [img_height, img_width, channels] = image.shape

    if mode == 'masks':
        output_images = []

        for id_label in range(num_labels):
            output_images.append(np.zeros((img_height_pad, img_width_pad)))

    elif mode == 'logical':
        output_image = np.zeros((img_height_pad+padding*2, img_width_pad+padding*2), 'uint8')

    # One BATCH of patches per sliding-window row, rather than one model
    # call per patch per model. This used to run
    # `sae_models[id].predict(sample)` with a batch of exactly 1, twice per
    # patch -- 84 separate Keras calls for a 1064x1342 page, measured at
    # ~14s on a 2-CPU pod. Two separate costs were being paid 84 times:
    # Keras's own per-`predict()` machinery (dataset adapter, callback list,
    # a tf.function re-entry) and a matmul too small to use the available
    # cores. Batching a whole row fixes both.
    #
    # Batching per ROW, rather than over the whole page, bounds peak memory
    # without needing a chunk-size knob: a row is `cols` patches of
    # w_height*w_width*3 float64, so ~4.7MB for the 6-column page above and
    # ~23MB for a 6000px-wide one.
    #
    # The (row, col) arithmetic below is preserved EXACTLY as it was, and
    # deliberately so -- it has two quirks that a tidy-up would silently
    # change. `row` was reassigned inside the inner loop, so the clamp stuck
    # for the rest of that row (reproduced here as `row_eff`, computed once
    # per row); and the column loop ranges over the UNPADDED `img_width`
    # while the row loop uses the padded height. Output must stay
    # byte-identical, so neither is "fixed" here.
    for row in range(0, img_height_pad, w_height-padding*2-1):
        print(str(row) + ' / ' + str(img_height_pad))
        if progress_callback is not None:
            progress_callback(row, img_height_pad)

        # Modifying the row and column indices to always cover the right and bottom borders of the image.
        row_eff = min(row, img_height_pad-w_height)

        samples = []
        col_effs = []
        for col in range(0, img_width, w_width-padding*2-1):
            if should_cancel is not None and should_cancel():
                raise ClassificationCancelled()

            col_eff = min(col, img_width_pad -w_width)

            sample = image_with_padding[row_eff:row_eff+w_height, col_eff:col_eff+w_width]

            # Pre-process (check that training does the same!)
            sample = (255. - sample) / 255.

            samples.append(sample)
            col_effs.append(col_eff)

        if not samples:
            continue

        batch = np.asarray(samples)
        if image_data_format() == 'channels_first':
            batch = batch.reshape(len(samples), 3, w_height, w_width)
        else:
            batch = batch.reshape(len(samples), w_height, w_width, 3)

        # model(x, training=False) rather than model.predict(x): identical
        # inference-mode arithmetic (predict() sets training=False too), minus
        # the per-call dataset/callback plumbing that made up a real share of
        # the 84-call cost. np.asarray() because this returns an EagerTensor,
        # and the indexing below is numpy's.
        predictions = [np.asarray(model(batch, training=False)) for model in sae_models]

        for i, col_eff in enumerate(col_effs):
            if mode == 'masks':

                for id_label in range(num_labels):
                    output_images[id_label][row_eff+padding:row_eff+w_height-padding,col_eff+padding:col_eff+w_width-padding] = 100*predictions[id_label][i,padding:w_height-padding,padding:w_width-padding,0]

            elif mode == 'logical':
                patch_predictions = [predictions[id_label][i,:,:,0] for id_label in range(num_labels)]

                output_image[row_eff+padding:row_eff+w_height-padding,col_eff+padding:col_eff+w_width-padding] = np.argmax( patch_predictions, axis = 0 )[padding:w_height-padding, padding:w_width-padding]

    #Cutting the padding to obtain the same image resolution as the original image.
    if mode == 'masks':
        output_images_no_pad = [output_image[padding:w_height-padding, padding:w_height-padding] for output_image in output_images]
        return output_images_no_pad
    elif mode == 'logical':
        result = output_image[padding:padding+img_height, padding:padding+img_width]
        if scale_ratio < 1.0:
            result = image_scaling.restore_label_map(result, orig_width, orig_height)
        return result

