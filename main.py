import numpy as np
import cv2
import matplotlib.pyplot as plt
import gc

from utils import position_transfer, read_boxes_from_txt, cal_iou
from img_utils import drawRect, show_img_with_boxes, cal_centeroid, confirm_cell_index


import logging
### 环境初始化
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
gc.enable()

if __name__ == '__main__':
    no = 'P10466'
    base_dir = 'data/rssrai2019_object_detection/train/'
    txt_path = f'{base_dir}labelTxt/labelTxt/{no}.txt'
    img_path = f'{base_dir}images/part6/{no}.png'
    img = cv2.imread(img_path)
    # show_img_with_boxes(img_path, txt_path, categories=None, save=False, show=True, use_modify=True)
    bbox = read_boxes_from_txt(txt_path)
    centeroids = cal_centeroid(bbox)
    result = confirm_cell_index(img.shape[:-1], 32, centeroids)
    print(result)


