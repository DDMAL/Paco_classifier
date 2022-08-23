from enum import Enum
import random as rd
import threading	
from typing import Optional

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

class Data():
    def __init__(self, x_name:str, path:str, size:Optional[int] = None, count:int=0):
        self.x_name = x_name # Only for Y
        self.path = path
        self.size = size
        self.img = None
        self.count = count # Only for Y

    def loadImg(self):
        self.img = np.load(self.path)

    def delImg(self):
        del self.img
        self.img = None

    def hasImg(self):
        return isinstance(self.img, np.ndarray)

    def __eq__(self, other):
        return other.path == self.path


class DataContainer():
    def __init__(self, ram_limit:int):
        self.meta = {"Image":{}}
        self.ram_limit = ram_limit
        self.current_ram = 0

    def _getCurrentRAM(self, layer_name:str):
        image_list = self.meta["Image"]
        rams = [y.size + image_list[y.x_name].size for y in self.meta[layer_name]["working"]]
        return sum(rams)

    def addXYPair(self, x_name:str, layer_name:str, x:Data, y:Data):
        # append x
        if x_name not in self.meta["Image"].keys():
            self.meta["Image"][x_name] = x

        # append y
        if layer_name not in self.meta.keys():
            self.meta[layer_name] = {"working":[], "pending":[]}
        self.meta[layer_name]["pending"].append(y)

    def reloadPendingList(self, layer_name:str):
        assert len(self.meta[layer_name]["working"]) != 0

        pending_list = self.meta[layer_name]["pending"]
        working_list = self.meta[layer_name]["working"]

        # Extract the last Y from the pending list and put it in the working list
        pending_list_pop_idx = 0
        if len(pending_list) != 0:
            # Pop the last Y from the pending liist
            new_Y:Data = pending_list[pending_list_pop_idx]
            new_X:Data = self.meta["Image"][new_Y.x_name]
            assert new_X.hasImg() == False, "Memory Leak: The x.img of data ({}) in pending list is not None".format(new_Y.path) 
            assert new_Y.hasImg() == False, "Memory Leak: The y.img of data ({}) in pending list is not None".format(new_Y.path) 
            
            required_ram = new_Y.size + new_X.size

            # Remove Xs/Ys in working list to get enough space to load new X/Y
            rd.shuffle(working_list)
            released_ram = 0
            removed_from_working_list = []
            while released_ram < required_ram and len(working_list) > 0:
                old_Y:Data = working_list.pop()
                old_X:Data = self.meta["Image"][old_Y.x_name]

                released_ram += old_Y.size + old_X.size
                old_Y.delImg()
                old_X.delImg()
                removed_from_working_list.append(old_Y)

            # We've removed all Xs/Ys in the working list but still need more RAM.
            if len(working_list) == 0 and self.ram_limit - self.current_ram + released_ram < required_ram:
                working_list += removed_from_working_list
                for Y in working_list:
                    X:Data = self.meta["Image"][Y.x_name]
                    X.loadImg()
                    Y.loadImg()
                raise ValueError("Not enought RAM, Release {} RAM but require {} RAM.".format(released_ram, required_ram))

            # Now load X/Y and move Y into moving list
            new_Y.loadImg()
            new_X.loadImg()
            working_list.append(new_Y)
            pending_list.pop(pending_list_pop_idx)

            self.current_ram -= released_ram
            self.current_ram += required_ram

            # Move data removed from the working list into pending list
            pending_list += removed_from_working_list

            # Load more images from the pending list to fill up the remaining RAM
            remaining_ram = released_ram - required_ram
            head_path = ""
            while len(pending_list) != 0:
                Y:Data = pending_list[-1]
                X:Data = self.meta["Image"][Y.x_name]
                required_ram = Y.size + X.size

                # Y fits to the RAM. Move it into the working list
                if required_ram < remaining_ram:
                    X.loadImg()
                    Y.loadImg()
                    self.meta[layer_name]["working"].append(Y)

                    pending_list.pop()
                    remaining_ram -= required_ram
                else:
                    # The first time we get sth that does not fit into RAM. Keep track of its path
                    if len(head_path) == 0:
                        head_path = Y.path
                        pending_list.insert(0, Y)
                        pending_list.pop()
                    else:
                        # All images in the pending list can't fit into RAM.
                        if head_path == Y.path:
                            break
                        else:
                            # else we just pop the data and push it back
                            pending_list.insert(0, Y)
                            pending_list.pop()

        # Update count
        for Y in pending_list:
            Y.count += 1

        assert self.ram_limit > self._getCurrentRAM(layer_name), "Exceed RAM limit: {} with loading {} RAM".format(self.ram_limit, self._getCurrentRAM(layer_name))


    def initWorkingList(self, layer_name:str):
        # Try to load all Ys in the pending list
        assert len(self.meta[layer_name]["working"]) == 0
        assert self.current_ram == 0

        pending_list = self.meta[layer_name]["pending"]
        required_ram = 0
        head_path = ""
        while len(pending_list) != 0:
            Y:Data = pending_list[-1]
            X:Data = self.meta["Image"][Y.x_name]
            required_ram = Y.size + X.size

            # The first time we get sth that does not fit into RAM. Keep track of its path
            if required_ram + self.current_ram < self.ram_limit:
                X.loadImg()
                Y.loadImg()
                self.meta[layer_name]["working"].append(Y)

                pending_list.pop()
                self.current_ram += required_ram
            else:
                # All images in the pending list can't fit into RAM.
                if len(head_path) == 0:
                    head_path = Y.path
                    pending_list.insert(0, Y)
                    pending_list.pop()
                else:
                    # Break if we meet the head again
                    if head_path == Y.path:
                        break
                    else:
                        # else we just pop the data and push it back
                        pending_list.insert(0, Y)
                        pending_list.pop()

        # All Xs/Ys exceed the memory limit
        if len(self.meta[layer_name]["working"]) == 0:
            raise ValueError(f"Require {required_ram} Gb RAM and exceed the limit {self.ram_limit} Gb RAM.")

        # Update count
        for Y in pending_list:
            Y.count += 1
    
    def delWorkingList(self, layer_name:str):
        working_list = self.meta[layer_name]["working"]
        while len(working_list) != 0:
            Y:Data = working_list.pop()
            X:Data = self.meta["Image"][Y.x_name]

            X.delImg()
            Y.delImg()
            self.meta[layer_name]["pending"].append(Y)
        self.current_ram = 0


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
            np.random.randint(0, gr.shape[0] + 1 - patch_height) for _ in range(batch_size)
        ]

        y_coords = [
            np.random.randint(0, gr.shape[1] + 1 - patch_width) for _ in range(batch_size)
        ]

    for i in range(batch_size):
        row = x_coords[i]
        col = y_coords[i]
        try:
            appendNewSample(gr, gt, row, col, patch_height, patch_width, gr_chunks, gt_chunks, i)
        except ValueError as e:
            error_msg = "Try try to extract (row={}, col={}, h={}, w={})".format(row, col, patch_height, patch_width)
            raise ValueError(error_msg) from e


def extractRandomSamples(inputs, idx_file, idx_label, patch_height, patch_width, batch_size, sample_extraction_mode):
    gt = inputs.meta[idx_label]["working"][idx_file].img
    gr_name = inputs.meta[idx_label]["working"][idx_file].x_name
    gr = inputs.meta["Image"][gr_name].img

    gr_chunks = np.zeros(shape=(batch_size, patch_width, patch_height, 3))
    gt_chunks = np.zeros(shape=(batch_size, patch_width, patch_height))

    try:
        extractRandomSamplesClass(gr, gt, patch_height, patch_width, batch_size, gr_chunks, gt_chunks)
    except ValueError as e:
        error_msg = "Failed to load y_path:{}, y_dim:{}, x_name:{}, x_path{}, x_dim:{}".format(inputs.meta[idx_label]["working"][idx_file].path, 
                                                    inputs.meta[idx_label]["working"][idx_file].img.shape,
                                                    inputs.meta[idx_label]["working"][idx_file].x_name,
                                                    inputs.meta["Image"][gr_name].path,
                                                    inputs.meta["Image"][gr_name].img.shape)
        raise ValueError(error_msg) from e

    return gr_chunks, gt_chunks  # convert into npy before yielding


def get_stride(patch_height, patch_width):
    return patch_height // 2, patch_width // 2

@threadsafe_generator  # Credit: https://anandology.com/blog/using-iterators-and-generators/
def createGeneratorSequentialExtraction(inputs, idx_file, idx_label, patch_height, patch_width, batch_size):
    
    hstride, wstride = get_stride(patch_height, patch_width)
    
    gr_chunks = np.zeros(shape=(batch_size, patch_width, patch_height, 3))
    gt_chunks = np.zeros(shape=(batch_size, patch_width, patch_height))

    gt = inputs.meta[idx_label]["working"][idx_file].img
    gr_name = inputs.meta[idx_label]["working"][idx_file].x_name
    gr = inputs.meta["Image"][gr_name].img
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
def createGenerator(inputs, layer_name, patch_height, patch_width, batch_size, file_selection_mode, sample_extraction_mode):
    # Check other file_selection mode and sample_extraction mode
    #list_idx_files = inputs[idx_label][1]
    #print ("Creating Generator: {}".format(idx_label))
    for key in inputs.meta.keys():
        if key == "Image":
            continue
        print ("Clean up layer {}".format(key))
        inputs.delWorkingList(key)
    print ("Init generator for layer {}".format(layer_name))
    inputs.initWorkingList(layer_name)

    while True:
        list_idx_files = [i for i in range(len(inputs.meta[layer_name]["working"]))]
        if file_selection_mode == FileSelectionMode.RANDOM:
            list_idx_files = [np.random.randint(len(inputs.meta[layer_name]["working"]))]
        elif file_selection_mode == FileSelectionMode.SHUFFLE:
            rd.shuffle(list_idx_files)
        for idx_file in list_idx_files:
            if sample_extraction_mode == SampleExtractionMode.RANDOM:
                yield extractRandomSamples(inputs, idx_file, layer_name, patch_height, patch_width, batch_size, sample_extraction_mode)
            elif sample_extraction_mode == SampleExtractionMode.SEQUENTIAL:
                for i in createGeneratorSequentialExtraction(inputs, idx_file, layer_name, patch_height, patch_width, batch_size):
                    yield i
            else:
                raise Exception(
                    'The sample extraction mode, {} {} does not exist.\n'.format(sample_extraction_mode, type(sample_extraction_mode))
                )
        inputs.reloadPendingList(layer_name)

def getTrain(inputs, patch_height, patch_width, batch_size, file_selection_mode, sample_extraction_mode):
    generator_labels = []

    for idx_label in inputs.meta.keys():
        if idx_label == "Image":
            continue
        generator_label = createGenerator(
            inputs, idx_label, patch_height, patch_width, batch_size, file_selection_mode, sample_extraction_mode
        )
        generator_labels.append(generator_label)

    return generator_labels
