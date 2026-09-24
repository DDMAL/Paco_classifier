"""process_image_msae() batches its sliding window one row at a time.

The batching rewrite must be OUTPUT-IDENTICAL to the per-patch version it
replaced, and that is not obvious to inspect: the original reassigned `row`
inside the inner loop (so the bottom-edge clamp stuck for the rest of that
row) and iterated columns over the UNPADDED image width while iterating rows
over the padded height. This test pins both quirks by running the real
implementation against a from-scratch reimplementation of the original loop
and demanding an exact match.

No TensorFlow needed: the module's TF imports are stubbed before import, and
the "models" are deterministic functions of the patch, so any change to which
pixels land in which patch -- or to where a patch's output is written back --
shows up as a mismatch.
"""
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


def _install_tf_stub():
    """Minimal stand-ins for the tensorflow symbols the Paco_classifier
    package pulls in at import time, so this test runs in a plain numpy+cv2
    environment.

    `Paco_classifier/__init__.py` imports the training modules too, so the
    stub has to satisfy their top-level imports (layers, optimizers,
    callbacks) even though nothing here touches training. Everything is an
    inert placeholder -- if a test ever actually calls one of these, that is
    a bug in the test, not something to make work.
    """
    if "tensorflow" in sys.modules:
        return

    def _placeholder(*_args, **_kwargs):
        raise AssertionError("stubbed TensorFlow symbol called in a unit test")

    tf = types.ModuleType("tensorflow")
    submodules = {
        "tensorflow.keras": {},
        "tensorflow.keras.models": {"Model": _placeholder, "load_model": _placeholder},
        "tensorflow.keras.backend": {"image_data_format": lambda: "channels_last"},
        "tensorflow.keras.layers": {
            name: _placeholder for name in
            ("Dropout", "UpSampling2D", "Concatenate", "Conv2D",
             "MaxPooling2D", "Input", "Masking")
        },
        "tensorflow.keras.optimizers": {"Adam": _placeholder},
        "tensorflow.keras.callbacks": {"EarlyStopping": _placeholder,
                                       "ModelCheckpoint": _placeholder},
    }
    sys.modules["tensorflow"] = tf
    for dotted, attrs in submodules.items():
        mod = types.ModuleType(dotted)
        for name, value in attrs.items():
            setattr(mod, name, value)
        sys.modules[dotted] = mod
        parent_name, _, leaf = dotted.rpartition(".")
        setattr(sys.modules[parent_name], leaf, mod)


_install_tf_stub()

from Paco_classifier import recognition_engine  # noqa: E402


class FakeModel:
    """Deterministic, per-sample, position-sensitive stand-in for a loaded
    autoencoder. Returns (N, h, w, 1); the value encodes both the patch's own
    content and the pixel's offset within it, so a patch written to the wrong
    place, or built from the wrong crop, cannot coincidentally match."""

    def __init__(self, seed):
        self.seed = seed
        self.calls = 0
        self.batch_sizes = []

    def _compute(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        n, h, w, _ = batch.shape
        ramp = (np.arange(h)[:, None] * 31 + np.arange(w)[None, :] * 7).astype(np.float64)
        out = batch.sum(axis=3) * (self.seed + 1) + ramp[None, :, :] + self.seed
        return out[..., None]

    # The batched path the implementation now uses.
    def __call__(self, batch, training=False):
        assert training is False, "inference must run with training=False"
        self.calls += 1
        self.batch_sizes.append(len(batch))
        return self._compute(batch)

    # The per-patch path the original implementation used.
    def predict(self, sample):
        self.calls += 1
        self.batch_sizes.append(len(sample))
        return self._compute(sample)


def original_process_image_msae(image, models, w_height, w_width, padding=25):
    """Verbatim transcription of the pre-batching inner loop (mode='logical'),
    kept here as the oracle. Deliberately NOT refactored -- including the
    `row = min(...)` reassignment and the unpadded-width column range."""
    num_labels = len(models)
    image_with_padding = __import__("cv2").copyMakeBorder(
        image, padding, padding, padding, padding, __import__("cv2").BORDER_REPLICATE)
    img_height_pad, img_width_pad, _ = image_with_padding.shape
    img_height, img_width, _ = image.shape
    output_image = np.zeros((img_height_pad + padding * 2, img_width_pad + padding * 2), 'uint8')

    for row in range(0, img_height_pad, w_height - padding * 2 - 1):
        for col in range(0, img_width, w_width - padding * 2 - 1):
            row = min(row, img_height_pad - w_height)
            col = min(col, img_width_pad - w_width)
            sample = image_with_padding[row:row + w_height, col:col + w_width]
            sample = (255. - sample) / 255.
            sample = np.asarray(sample).reshape(1, w_height, w_width, 3)
            predictions = [models[i].predict(sample)[0, :, :, 0] for i in range(num_labels)]
            output_image[row + padding:row + w_height - padding,
                         col + padding:col + w_width - padding] = \
                np.argmax(predictions, axis=0)[padding:w_height - padding, padding:w_width - padding]
    return output_image[padding:padding + img_height, padding:padding + img_width]


def run_batched(image, models, w, mode='logical', **kwargs):
    """Call the real process_image_msae with `models` standing in for the
    two .h5 files it would otherwise load. Patches the cached loader rather
    than load_model itself, so the test exercises the same path production
    takes without touching the on-disk cache."""
    paths = [f"model-{i}.h5" for i in range(len(models))]
    by_path = dict(zip(paths, models))
    with mock.patch.object(recognition_engine, "load_model_cached", by_path.__getitem__):
        return recognition_engine.process_image_msae(
            image, paths, w, w, mode=mode, **kwargs)


def original_process_image_msae_masks(image, models, w_height, w_width, padding=25):
    """Verbatim transcription of the pre-batching mode='masks' branch.

    Every real caller passes mode='logical' (Classifiers/run_classifier.py,
    evaluation.py, both GUIs), so this branch is effectively vestigial -- and
    it carries a pre-existing bug the transcription keeps on purpose: the
    final crop indexes BOTH axes with w_height. This oracle exists to prove
    the batching rewrite did not change that branch's behaviour, not to
    bless it."""
    cv2 = __import__("cv2")
    num_labels = len(models)
    image_with_padding = cv2.copyMakeBorder(
        image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    img_height_pad, img_width_pad, _ = image_with_padding.shape
    _, img_width, _ = image.shape
    output_images = [np.zeros((img_height_pad, img_width_pad)) for _ in range(num_labels)]

    for row in range(0, img_height_pad, w_height - padding * 2 - 1):
        for col in range(0, img_width, w_width - padding * 2 - 1):
            row = min(row, img_height_pad - w_height)
            col = min(col, img_width_pad - w_width)
            sample = image_with_padding[row:row + w_height, col:col + w_width]
            sample = (255. - sample) / 255.
            sample = np.asarray(sample).reshape(1, w_height, w_width, 3)
            for i in range(num_labels):
                prediction = models[i].predict(sample)
                output_images[i][row + padding:row + w_height - padding,
                                 col + padding:col + w_width - padding] = \
                    100 * prediction[0, padding:w_height - padding, padding:w_width - padding, 0]
    return [o[padding:w_height - padding, padding:w_height - padding] for o in output_images]


class BatchedSlidingWindowTest(unittest.TestCase):
    # Sizes chosen so the bottom/right clamp actually fires (the page is not a
    # whole number of strides), which is the case the `row`/`col` quirks govern.
    SHAPES = [(300, 260), (256, 256), (700, 540), (120, 640)]
    W = 128
    PADDING = 25

    def _image(self, h, w):
        rng = np.random.default_rng(h * 1000 + w)
        return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    def test_output_is_identical_to_the_per_patch_implementation(self):
        for h, w in self.SHAPES:
            with self.subTest(shape=(h, w)):
                image = self._image(h, w)
                expected = original_process_image_msae(
                    image, [FakeModel(0), FakeModel(1)], self.W, self.W, self.PADDING)
                actual = run_batched(image, [FakeModel(0), FakeModel(1)], self.W)
                np.testing.assert_array_equal(actual, expected)

    def test_masks_mode_output_is_identical_too(self):
        for h, w in self.SHAPES:
            with self.subTest(shape=(h, w)):
                image = self._image(h, w)
                expected = original_process_image_msae_masks(
                    image, [FakeModel(0), FakeModel(1)], self.W, self.W, self.PADDING)
                actual = run_batched(image, [FakeModel(0), FakeModel(1)], self.W,
                                     mode='masks')
                self.assertEqual(len(actual), len(expected))
                for got, want in zip(actual, expected):
                    np.testing.assert_array_equal(got, want)

    def test_runs_one_batched_call_per_row_per_model(self):
        image = self._image(300, 260)
        models = [FakeModel(0), FakeModel(1)]
        run_batched(image, models, self.W)
        # One call per sliding-window row, each covering that row's columns --
        # not one call per patch, which is the whole point of the change.
        for m in models:
            self.assertEqual(m.calls, len(m.batch_sizes))
            self.assertTrue(all(n >= 1 for n in m.batch_sizes))
            self.assertGreater(max(m.batch_sizes), 1,
                               "patches are not being batched at all")
        self.assertEqual(models[0].batch_sizes, models[1].batch_sizes)

    def test_cancellation_still_raises_between_patches(self):
        image = self._image(300, 260)
        calls = {"n": 0}

        def should_cancel():
            calls["n"] += 1
            return calls["n"] > 3

        with self.assertRaises(recognition_engine.ClassificationCancelled):
            run_batched(image, [FakeModel(0), FakeModel(1)], self.W,
                        should_cancel=should_cancel)

    def test_progress_callback_still_fires_once_per_row_with_padded_total(self):
        image = self._image(300, 260)
        seen = []
        run_batched(image, [FakeModel(0), FakeModel(1)], self.W,
                    progress_callback=lambda r, t: seen.append((r, t)))
        self.assertTrue(seen)
        totals = {t for _, t in seen}
        self.assertEqual(len(totals), 1, "total must be constant across the pass")
        self.assertEqual(totals.pop(), image.shape[0] + self.PADDING * 2)
        rows = [r for r, _ in seen]
        self.assertEqual(rows, sorted(rows))
        # Reported UNCLAMPED, as before -- the UI progress bar depends on it
        # reaching the total, not on the internal bottom-edge clamp.
        self.assertEqual(rows[0], 0)


class ModelCacheTest(unittest.TestCase):
    """load_model_cached() exists to stop process_image_msae() re-reading both
    .h5 files on every page. It must still notice a swapped-in checkpoint."""

    def setUp(self):
        recognition_engine._MODEL_CACHE.clear()
        self.addCleanup(recognition_engine._MODEL_CACHE.clear)

    def test_repeated_loads_of_an_unchanged_file_hit_the_cache(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as fh:
            fh.write(b"weights"); fh.flush()
            with mock.patch.object(recognition_engine, "load_model",
                                   side_effect=lambda p: object()) as loader:
                first = recognition_engine.load_model_cached(fh.name)
                second = recognition_engine.load_model_cached(fh.name)
        self.assertIs(first, second)
        self.assertEqual(loader.call_count, 1)

    def test_a_rewritten_checkpoint_is_reloaded(self):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as fh:
            fh.write(b"weights"); fh.flush()
            path = fh.name
        self.addCleanup(os.unlink, path)
        with mock.patch.object(recognition_engine, "load_model",
                               side_effect=lambda p: object()) as loader:
            first = recognition_engine.load_model_cached(path)
            # Same path, different content and mtime -- the exact case a
            # retrained-weights swap produces.
            with open(path, "wb") as fh:
                fh.write(b"different weights")
            os.utime(path, (0, 0))
            second = recognition_engine.load_model_cached(path)
        self.assertIsNot(first, second)
        self.assertEqual(loader.call_count, 2)

    def test_an_unstattable_path_falls_back_to_loading_every_time(self):
        with mock.patch.object(recognition_engine, "load_model",
                               side_effect=lambda p: object()) as loader:
            recognition_engine.load_model_cached("/no/such/model.h5")
            recognition_engine.load_model_cached("/no/such/model.h5")
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(recognition_engine._MODEL_CACHE, {})


if __name__ == "__main__":
    unittest.main()
