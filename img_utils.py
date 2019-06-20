import cv2
import re
import pandas as pd
import gc
import numpy as np

from utils import read_boxes_from_txt

import logging
### 环境初始化
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
gc.enable()

### 画线函数，四条线组成一个框
def drawRect(img, pts, color, lineWidth=1, use_modify=False):
    assert len(pts) == 4 and type(pts[0]) == tuple
    if use_modify:
        pts_temp = np.array(pts)
        pts_x, pts_y = pts_temp[:,0], pts_temp[:,1]
        x_min, y_min = np.min(pts_x), np.min(pts_y)
        x_max, y_max = np.max(pts_x), np.max(pts_y)
        pts = [(x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)]
    for i in range(4):
        cv2.line(img, pts[i % 4], pts[(i+1) % 4], color, lineWidth)

### 将16进制的颜色转换成RGB
### #0000FF -> (0, 0, 255)
def hextoRgb(hex):
    assert len(hex) == 7 and hex[0] == '#'
    hex = hex[1:]
    opt = re.findall(r'(.{2})', hex)
    return tuple([int(o, 16) for o in opt])

### 显示单张描绘出边界框的图片
def show_img_with_boxes(img_path, txt_path, categories=None, save=False, show=False, use_modify=False):
    boxes_counter = 0
    color_csv = 'data/color.csv'
    boxes = read_boxes_from_txt(txt_path)
    img = cv2.imread(img_path)
    if not categories:
        for k in boxes.keys():
            for data in boxes[k]:
                pts = data[0]
                color = hextoRgb(read_color_from_csv_acord_cat(str(k), color_csv))
                drawRect(img, pts, color=color, use_modify=use_modify)
                boxes_counter += 1
        logging.info(f'show img {img_path} with {len(boxes.keys())} categories: {boxes.keys()} and {boxes_counter} boxes')
    else:
        assert len(categories) != 0
        for k in categories:
            if k not in boxes.keys():
                logging.info(f'Categories input {k} not in {txt_path}!')
            else:
                for data in boxes[k]:
                    pts = data[0]
                    color = hextoRgb(read_color_from_csv_acord_cat(str(k), color_csv))
                    drawRect(img, pts, color=color, use_modify=use_modify)
                    boxes_counter += 1
        logging.info(f'show img {img_path} with {len(categories)} categories and {boxes_counter} boxes')
    if save:
        cv2.imwrite('data/output/test.png', img)
    if show:
        cv2.imshow(img_path, img)
        cv2.waitKey()
        cv2.destroyAllWindows()


### 根据输入的类别匹配边界框颜色
def read_color_from_csv_acord_cat(cat, csv_path):
    assert type(cat) == str
    color_df = pd.read_csv('data/color.csv', encoding='gbk')
    color_df['cat'] = color_df['cat'].apply(lambda x:x.split('(')[-1][:-1])
    return color_df[color_df['cat'] == cat]['color'].values[0]




# if __name__ == '__main__':
#     a = read_color_from_csv_acord_cat('large-vehicle', 'data/color.csv')
#     print(a)

def compute_area(points):
    point_num = len(points)
    if(point_num < 3): return 0.0
    s = points[0][1] * (points[point_num-1][0] - points[1][0])
    for i in range(1, point_num):
        s += points[i][1] * (points[i-1][0] - points[(i+1)%point_num][0])
    return abs(s/2.0)

if __name__ == '__main__':
    poly=[[1,1], [1,3],[3,3], [3,1]]
    print(compute_area(poly))