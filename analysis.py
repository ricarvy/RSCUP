import numpy as np
import pandas as pd
import os
from utils import read_boxes_from_txt
from img_utils import compute_area, cal_centeroid

def cat_dict_handler(cat_dict, op_name):
    cat_dict_temp = cat_dict.copy()
    assert op_name in ['min', 'max', 'mean', 'std']
    if op_name == 'min':
        op = np.min
    elif op_name == 'max':
        op = np.max
    elif op_name == 'mean':
        op = np.mean
    else:
        op = np.std
    for k in cat_dict_temp.keys():
        cat_dict_temp[k] = op(cat_dict_temp[k])
    return cat_dict_temp


def cat_list2dict(cats):
    cat_dict = dict()
    for cat in cats:
        cat_dict[cat] = []
    return cat_dict

def gen_categories(file_path):
    df = pd.read_csv(file_path, encoding='gbk')
    df['cat'] = df['cat'].apply(lambda x:str(x).split('(')[-1][:-1])
    return df['cat'].values

### 获取训练集内所有txt物体类别的图形面积信息
def cal_info_of_cats(file_path, txt_base_path):
    cats = gen_categories(file_path)
    cat_dict = cat_list2dict(cats)

    txt_single_list = os.listdir(txt_base_path)
    txt_path_list = [os.path.join(txt_base_path, p) for p in txt_single_list]
    for txt_path in txt_path_list:
        boxes = read_boxes_from_txt(txt_path=txt_path)
        for k in boxes.keys():
            for box in boxes[k]:
                points = np.array(box[0]).tolist()
                area = compute_area(points)
                cat_dict[k].append(area)
    cat_dict_max = cat_dict_handler(cat_dict, 'max')
    cat_dict_min = cat_dict_handler(cat_dict, 'min')
    cat_dict_mean = cat_dict_handler(cat_dict, 'mean')
    cat_dict_std = cat_dict_handler(cat_dict, 'std')
    return cat_dict, [cat_dict_max, cat_dict_min, cat_dict_mean, cat_dict_std]


if __name__ == '__main__':
    file_path = 'data/color.csv'
    txt_base_path = 'data/rssrai2019_object_detection/train/labelTxt/labelTxt'
    cat_dict, cat_dict_info = cal_info_of_cats(file_path, txt_base_path)
    print(cat_dict_info)

