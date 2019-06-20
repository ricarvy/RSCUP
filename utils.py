import cv2
import numpy as np
import gc
import matplotlib.pyplot as plt
import configs as cfg


import logging
### 环境初始化
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
gc.enable()

### 读入八个坐标点值
### [x0, y0, x1, y1, x2, y2, x3, y3] -> [(x0, y0), (x1,y1), (x2, y2), (x3, y3)]
def position_transfer(l):
    assert len(l) == 8
    l_cp = l.copy()
    l_cp = np.array(l_cp).reshape((-1, 2))
    l_result = []
    for i, l in enumerate(l_cp):
        l_result.append(tuple(l))
    return l_result

### 从txt中读取信息
def read_boxes_from_txt(txt_path):
    '''

    :param txt_path:
    :return: json like object,
    {
        'large_vehicle':[[(341, 292),...,(346, 457), 0], [(341, 292),...,(346, 457), 1]...],
        'small_vehicle':[[(341, 292),...,(346, 457), 0], [(341, 292),...,(346, 457), 0]...]
    }
    '''
    boxes = {}
    box = []
    difficulty = 0
    for line_idx, line in enumerate(open(txt_path, 'r')):
        if line_idx >= 2:
            data = np.array(line.split(' '))
            b = data[:8].astype('int')
            tag = data[-2]
            difficulty = data[-1].astype('int')
            box.append(position_transfer(b))
            box.append(difficulty)
            if tag not in boxes.keys():
                boxes[tag] = []
            boxes[tag].append(box)
            box = []
    return boxes

### 计算任意形状的两个四边形的IOU
def cal_iou(pts1, pts2):
    '''
    Warning: The order should be up-left --> up_right --> down_right --> dowm_left
    :param pts1:like [(134,123),(156,124),(235,353),(123,564)]
    :param pts2: like [(134,123),(156,124),(235,353),(123,564)]
    :return: IOU value
    '''
    # image = cv2.imread(image_path)
    image = np.zeros((1080, 1920, 3), np.uint8)
    pts1 = np.array([[pt1, pt2] for pt1, pt2 in pts1]) - 1
    pts2 = np.array([[pt1, pt2] for pt1, pt2 in pts2]) - 1
    original_grasp_bboxes = np.expand_dims(pts1, 0)
    prediction_grasp_bboxes = np.expand_dims(pts2, 0)
    im = np.zeros(image.shape[:2], dtype="uint8")
    im1 = np.zeros(image.shape[:2], dtype="uint8")
    original_grasp_mask = cv2.fillPoly(im, [original_grasp_bboxes], (255,255,255))
    prediction_grasp_mask = cv2.fillPoly(im1, [prediction_grasp_bboxes], (255,255,255))
    masked_and = cv2.bitwise_and(original_grasp_mask, prediction_grasp_mask)
    masked_or = cv2.bitwise_or(original_grasp_mask, prediction_grasp_mask)
    or_area = np.sum(np.float32(np.greater(masked_or, 0)))
    and_area = np.sum(np.float32(np.greater(masked_and, 0)))
    IOU = and_area / or_area
    return IOU


def decode_position(img, cell, x, y, w, h, S=13):
    '''

    :param img: 3D array like (wi, hi, Channels) for the img
    :param cell: 2D array like (xcol, yrow)
    :param x: position x(after nomalization)
    :param y: position y(after nomalization)
    :param w: ratio w = wb/wi
    :param h: ratio h = hb/hi
    :return:object like [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
    :default S: 7
    '''
    ### xs should be constructed like [x0, x1, x2, x3]
    # xs = []
    ### ys should be constructed like [y0, y1, y2, y3]
    # ys = []
    ### prepare for your work..

    cell0,cell1=cell

    xc = (x + cell0) * img.shape[0] / S
    yc = (y + cell1) * img.shape[1] / S
    wb = w * img.shape[0]
    hb = h * img.shape[1]

    p1 = [xc - 0.5 * wb, yc + 0.5 * hb]
    p2 = [xc + 0.5 * wb, yc + 0.5 * hb]
    p3 = [xc - 0.5 * wb, yc - 0.5 * hb]
    p4 = [xc + 0.5 * wb, yc - 0.5 * hb]

    xs = [p1[0], p2[0], p3[0], p4[0]]
    ys = [p1[1], p2[1], p3[1], p4[1]]
    ### end your work
    result = []
    for xy in zip(xs, ys):
        result.append(xy)
    return result


if __name__ == '__main__':
    img = np.random.randint(0, 255, (416, 416, 3))
    S = 13
    test_cell_position = (3,4)
    test_x = 0.5
    test_y = 0.5
    test_w = 13 / 416
    test_h = 13 / 416
    result = decode_position(img, cell=test_cell_position, x=test_x, y=test_y, w=test_w, h=test_h,S=S)
    print(result)