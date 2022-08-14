from enum import Enum	
import random as rd	
import threading	

import numpy as np

class FileSelectionMode(Enum):
    RANDOM,     \
    SHUFFLE,    \
    DEFAULT     \
    = range(3)

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return FileSelectionMode[s]
        except KeyError:
            raise ValueError()

class SampleExtractionMode(Enum):
    RANDOM,     \
    SEQUENTIAL  \
    = range(2)

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return SampleExtractionMode[s]
        except KeyError:
            raise ValueError()

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe."""

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

def appendNewSample(gr, gt, row, col, patch_height, patch_width, gr_chunks, gt_chunks, index):
    gr_sample = gr[
            row : row + patch_height, col : col + patch_width
        ]  # Greyscale image
    gt_sample = gt[
        row : row + patch_height, col : col + patch_width
    ]  # Ground truth
    gr_chunks[index] = gr_sample
    gt_chunks[index] = gt_sample

def extractRandomSamplesClass(gr, gt, patch_height, patch_width, batch_size, gr_chunks, gt_chunks):
    potential_training_examples = np.where(gt[:-patch_height, :-patch_width] == 1)

    num_coords = len(potential_training_examples[0])

    if num_coords >= batch_size:

        index_coords_selected = [
            np.random.randint(0, num_coords) for _ in range(batch_size)
        ]
        x_coords = potential_training_examples[0][index_coords_selected]
        y_coords = potential_training_examples[1][index_coords_selected]
    else:
        x_coords = [
            np.random.randint(0, gr.shape[0]) for _ in range(batch_size)
        ]

        y_coords = [
            np.random.randint(0, gr.shape[1]) for _ in range(batch_size)
        ]

    for i in range(batch_size):
        row = x_coords[i]
        col = y_coords[i]
        appendNewSample(gr, gt, row, col, patch_height, patch_width, gr_chunks, gt_chunks, i)


def extractRandomSamples(inputs, idx_file, idx_label, patch_height, patch_width, batch_size, sample_extraction_mode):
    gr = inputs["Image"][0][idx_file]
    gt = inputs[idx_label][0][idx_file]

    gr_chunks = np.zeros(shape=(batch_size, patch_width, patch_height, 3))
    gt_chunks = np.zeros(shape=(batch_size, patch_width, patch_height))

    extractRandomSamplesClass(gr, gt, patch_height, patch_width, batch_size, gr_chunks, gt_chunks)

    return gr_chunks, gt_chunks  # convert into npy before yielding


def get_stride(patch_height, patch_width):
    return patch_height // 2, patch_width // 2

@threadsafe_generator  # Credit: https://anandology.com/blog/using-iterators-and-generators/
def createGeneratorSequentialExtraction(inputs, idx_file, idx_label, patch_height, patch_width, batch_size):
    
    hstride, wstride = get_stride(patch_height, patch_width)
    
    gr_chunks = np.zeros(shape=(batch_size, patch_width, patch_height, 3))
    gt_chunks = np.zeros(shape=(batch_size, patch_width, patch_height))

    gr = inputs["Image"][0][idx_file]
    gt = inputs[idx_label][0][idx_file]
    count = 0
    for row in range(0, gr.shape[0] - patch_height, hstride):
        for col in range(0, gr.shape[1] - patch_width, wstride):
            appendNewSample(gr, gt, row, col, patch_height, patch_width, gr_chunks, gt_chunks, count)
            count +=1
            if count % batch_size == 0:
                yield gr_chunks, gt_chunks
                gr_chunks = np.zeros(shape=(batch_size, patch_width, patch_height, 3))
                gt_chunks = np.zeros(shape=(batch_size, patch_width, patch_height))
                count = 0

@threadsafe_generator  # Credit: https://anandology.com/blog/using-iterators-and-generators/
def createGenerator(inputs, idx_label, patch_height, patch_width, batch_size, file_selection_mode, sample_extraction_mode):
    # Check other file_selection mode and sample_extraction mode
    list_idx_files = inputs[idx_label][1]
    print ("Creating Generator: {}".format(idx_label))

    while True:
        if file_selection_mode == FileSelectionMode.RANDOM:
            list_idx_files = [np.random.randint(len(inputs[idx_label][1]))]
        elif file_selection_mode == FileSelectionMode.SHUFFLE:
            rd.shuffle(list_idx_files)
        for idx_file in list_idx_files:
            if sample_extraction_mode == SampleExtractionMode.RANDOM:
                yield extractRandomSamples(inputs, idx_file, idx_label, patch_height, patch_width, batch_size, sample_extraction_mode)
            elif sample_extraction_mode == SampleExtractionMode.SEQUENTIAL:
                for i in createGeneratorSequentialExtraction(inputs, idx_file, idx_label, patch_height, patch_width, batch_size):
                    yield i
            else:
                raise Exception(
                    'The sample extraction mode, {} {} does not exist.\n'.format(sample_extraction_mode, type(sample_extraction_mode))
                )

def getTrain(inputs, num_labels, patch_height, patch_width, batch_size, file_selection_mode, sample_extraction_mode):
    generator_labels = []

    for idx_label in inputs:
        if idx_label == "Image":
            continue
        generator_label = createGenerator(
            inputs, idx_label, patch_height, patch_width, batch_size, file_selection_mode, sample_extraction_mode
        )
        generator_labels.append(generator_label)

    return generator_labels