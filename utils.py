import cv2
import numpy as np
import gc


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
    dificulty = 0
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
def cal_iou(image_path, pts1, pts2):
    image = cv2.imread(image_path)
    pts1 = [[pt1, pt2] for pt1, pt2 in pts1]
    pts2 = [[pt1, pt2] for pt1, pt2 in pts2]
    original_grasp_bboxes = np.expand_dims(pts1, 0)
    prediction_grasp_bboxes = np.expand_dims(pts2, 0)
    # original_grasp_bboxes = np.array([[[361, 260.582], [301, 315], [320, 336], [380, 281.582]]], dtype=np.int32)
    # prediction_grasp_bboxes = np.array([[[301, 290.582], [321, 322], [310, 346], [380, 291.582]]], dtype=np.int32)
    im = np.zeros(image.shape[:2], dtype="uint8")
    im1 = np.zeros(image.shape[:2], dtype="uint8")
    original_grasp_mask = cv2.fillPoly(im, original_grasp_bboxes, 255)
    prediction_grasp_mask = cv2.fillPoly(im1, prediction_grasp_bboxes, 255)
    masked_and = cv2.bitwise_and(original_grasp_mask, prediction_grasp_mask, mask=im)
    masked_or = cv2.bitwise_or(original_grasp_mask, prediction_grasp_mask)

    or_area = np.sum(np.float32(np.greater(masked_or, 0)))
    and_area = np.sum(np.float32(np.greater(masked_and, 0)))
    IOU = and_area / or_area
    return IOU

if __name__ == '__main__':
    a = np.array([(1,2),(3,4),(5,6),(7,8)])
    b = a.copy()
    c = cal_iou(None, a, b)