import numpy as np
import cv2
import matplotlib.pyplot as plt
import gc

from utils import position_transfer, read_boxes_from_txt, cal_iou
from img_utils import drawRect, show_img_with_boxes

import logging
### 环境初始化
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
gc.enable()

if __name__ == '__main__':

    txt_path = 'data/txt/P9895.txt'
    img_path = 'data/img/test_img.png'
    # show_img_with_boxes(img_path, txt_path, save=True)
    iou = cal_iou(np.array([(0,0),(250,0),(250,400),(0,400)]),
                  np.array([(0,0),(500,0),(500,200),(0,200)]))
    print(iou)


