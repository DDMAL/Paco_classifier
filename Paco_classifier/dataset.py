import tensorflow as tf
import cv2
import numpy as np

IMAGE_KEY = "Image"
REGION_KEY = "rgba PNG - Selected regions"

# = = = = =
def visualizeX(img_x, filename):
    img_x[img_x==-1]=0.0
    img_x = (img_x*255).astype(np.uint8)
    cv2.imwrite (filename, img_x)

def visualizeY(img_y, filename):
    ret = img_y.astype(np.float64)
    ret = (ret*255).astype(np.uint8)
    cv2.imwrite (filename, ret)
# = = = = =

def loadImg(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED,)  # 4-channel
    return img

def fakeMask(mask):
    W, H = mask.shape
    ret = np.zeros_like(mask)
    ret[:W//2, :H//2] = 1.0
    ret = (ret==1.0)
    return ret

def loadMask(path):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED,)  # 4-channel
    TRANSPARENCY = 3
    mask = (mask[:, :, TRANSPARENCY] == 255)

    return fakeMask(mask)
    #return mask

def processXImg(img_gt, img_region):
    """
    Follow the chaotic way to process x in training_engine_sae.get_image_with_gt
    Return:
        <img_gt>: np.ndarray of shape (H, W, 3) and type np.float64 in range [0.0, 1.0]
                  for pixels within the region, -1.0 for pixels outside of the region.
    """
    img_gt = img_gt[..., :3]
    img_gt = (255.-img_gt) / 255.

    #Deactivate the training process for pixels outside the region mask
    l = np.where((img_region == 0))
    img_gt[l] = -1
    return img_gt

def processYLayer(layer_y, img_region):
    """
    Follow the chaotic way in training_engine_sae.load_gt_image 

    Return:
        Y: np.ndarray of type bool with shape (W, H, 4). Labelled pixels are True.
    """
    TRANSPARENCY = 3
    layer_y = (layer_y[:, :, TRANSPARENCY] == 255)
    return np.logical_and(layer_y, img_region)

def createWriter(n_layer:int):
    writers = [tf.io.TFRecordWriter("./Groceries/{:02d}.tfrecords".format(i)) for i in range(n_layer)]
    return writers

def detail(x, y):
    print (f"x: {x.shape} {x.dtype} {np.min(x)}/{np.max(x)}")
    print (f"y: {y.shape} {y.dtype} {np.min(y)}/{np.max(y)}")

def preprocess(inputs):
    img_region_path_list = [(_img['resource_path'], _region['resource_path']) for _img, _region in zip(inputs[IMAGE_KEY], inputs[REGION_KEY])]
    layer_key_list = [k for k in sorted(inputs.keys()) if k not in [IMAGE_KEY, REGION_KEY]]

    # Create Tfboard writer
    writers = createWriter(len(layer_key_list))

    for idx, (img_path, region_path) in enumerate(img_region_path_list):
        img_x = loadImg(img_path)
        mask  = loadMask(region_path)
        img_x = processXImg(img_x, mask) # X

        #visualizeX(img_x, filename=f"./Groceries/{idx}.png")

        for layer_idx, layer_key in enumerate(layer_key_list):
            layer_path = inputs[layer_key][idx]['resource_path']
            img_y = loadImg(layer_path)
            img_y = processYLayer(img_y, mask) # Y

            # img_x, mask, img_y
            #print (f"Y: {np.min(img_y)}, {np.max(img_y)}, {img_y.dtype}, {img_y.shape}")
            #visualizeY(img_y, filename=f"./Groceries/{idx}-{layer_idx}.png")
            detail(img_x, img_y)

            # Write img_x and img_y to tf.TfRecord
            features = dict()
            print ("Unpack x")
            features['x'] = tf.train.Feature(float_list=tf.train.FloatList(value=img_x.flatten()))
            features['x_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=img_x.shape))
            print ("Unpack y")
            features['y'] = tf.train.Feature(float_list=tf.train.FloatList(value=img_y.flatten()))
            features['y_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=img_y.shape))
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writers[layer_idx].write(example.SerializeToString())

    # Close Writer
    for writer in writers:
        writer.close()
            