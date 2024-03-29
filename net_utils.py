import numpy as np
import keras.backend as K

from utils import decode_position

def sigmoid_func(x):
    return 1 / (1+np.exp(-x))

def exp_func(x):
    return np.exp(x)

def softmax_func(x):
    return np.exp(x) / np.sum(np.exp(x))

def decode(img, feature_map, size=(13,13), num_classes=80, num_anchors = 5):
    height, width = size
    assert feature_map.flatten().shape[0] % height*width*num_anchors*(num_classes+5) == 0
    feature_map = np.reshape(feature_map, (height, width, num_anchors, num_classes+5))

    xy_offset = feature_map[:, :, :, 0:2]
    wh_offset = feature_map[:, :, :, 2:4]
    obj_probs = feature_map[:, :, :, 4]
    class_probs = feature_map[:, :, :, 5:]

    result = np.zeros(shape=(height, width, num_anchors, num_classes+9))
    for he in range(height):
        for wi in range(width):
            for a in range(num_anchors):
                x,y = xy_offset[he][wi][a]
                w,h = wh_offset[he][wi][a]
                class_prob = class_probs[he][wi][a].tolist()
                obj_prob = obj_probs[he][wi][a]
                cell = (he, wi)
                real_position = np.array(decode_position(img, cell, x, y, w, h, S=size[0])).flatten().tolist()
                real_position = np.array(real_position + [obj_prob] + class_prob)
                result[he][wi][a] = real_position
                print(he, wi, a)
    return result

def yolo_head(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters.

    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.

    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # Static implementation for fixed models.
    # TODO: Remove or add option for static implementation.
    # _, conv_height, conv_width, _ = K.int_shape(feats)
    # conv_dims = K.variable([conv_width, conv_height])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    # Static generation of conv_index:
    # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])
    # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.
    # conv_index = K.variable(
    #     conv_index.reshape(1, conv_height, conv_width, 1, 2))
    # feats = Reshape(
    #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)

    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.softmax(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs




if __name__ == '__main__':
    img = np.random.randint(0, 255, (416, 416, 3))
    feature_map = np.array([1/2, 1/2, 13/416, 13/416, 0.5] + np.random.uniform(0, 1, size=(80,)).tolist())
    print(feature_map.shape)
    feature_map = np.repeat(np.expand_dims(feature_map, axis=0), 5, axis=0)
    print(feature_map.shape)
    feature_map = np.repeat(np.expand_dims(feature_map, axis=0), 13, axis=0)
    print(feature_map.shape)
    feature_map = np.repeat(np.expand_dims(feature_map, axis=0), 13, axis=0)
    print(feature_map.shape)
    result = decode(img, feature_map)
    print(result.shape)