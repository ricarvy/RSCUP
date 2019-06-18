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
            dificulty = data[-1].astype('int')
            box.append(position_transfer(b))
            box.append(dificulty)
            if tag not in boxes.keys():
                boxes[tag] = []
            boxes[tag].append(box)
            box = []
    return boxes

