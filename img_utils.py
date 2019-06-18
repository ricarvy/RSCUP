import cv2
import re
import pandas as pd
import gc

from utils import read_boxes_from_txt

import logging
### 环境初始化
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
gc.enable()

### 画线函数，四条线组成一个框
def drawRect(img, pts, color, lineWidth=1):
    assert len(pts) == 4 and type(pts[0]) == tuple
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
def show_img_with_boxes(img_path, txt_path, save=False):
    boxes_counter = 0
    color_csv = 'data/color.csv'
    boxes = read_boxes_from_txt(txt_path)
    img = cv2.imread(img_path)
    for k in boxes.keys():
        for data in boxes[k]:
            pts = data[0]
            color = hextoRgb(read_color_from_csv_acord_cat(str(k), color_csv))
            drawRect(img, pts, color=color)
            boxes_counter += 1
    if save:
        cv2.imwrite('data/output/test.png', img)
    logging.info(f'show img {img_path} with {len(boxes.keys())} categories and {boxes_counter} boxes')
    cv2.imshow('test', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


### 根据输入的类别匹配边界框颜色
def read_color_from_csv_acord_cat(cat, csv_path):
    assert type(cat) == str
    color_df = pd.read_csv('data/color.csv', encoding='gbk')
    color_df['cat'] = color_df['cat'].apply(lambda x:x.split('(')[-1][:-1])
    return color_df[color_df['cat'] == cat]['color'].values[0]




if __name__ == '__main__':
    a = read_color_from_csv_acord_cat('large-vehicle', 'data/color.csv')
    print(a)

