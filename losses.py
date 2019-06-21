import keras.backend as K
import numpy as np
import cv2

from img_utils import cal_centeroid, confirm_cell_index
from utils import decode_position, cal_iou, read_boxes_from_txt
from analysis import gen_categories

### 将二位坐标转为一维
def twoD2oneD(pt, S):
    x, y = pt
    return x*S+y

### 计算MSE
def calculateMSE(y_true, y_pred):
    return sum([(y_true - y_pred)**2 for y_true, y_pred in zip(y_true, y_pred)])/len(y_pred)

def cal_centeroid_loss(lambda_coord, true_p, pred_p):
    [x_t, y_t, w_t, h_t] = true_p
    [x_p, y_p, w_p, h_p] = pred_p
    ### 对预测中心坐标做损失
    center_loss = lambda_coord * ((x_t - x_p)**2 + (y_t-y_p)**2)
    ### 对预测的边界宽高做损失
    box_loss = lambda_coord * ((np.sqrt(w_t) - np.sqrt(w_p))**2 + (np.sqrt(h_t) - np.sqrt(h_p))**2)
    return center_loss, box_loss



def cal_loss(img, y_preds, ground_truths, categories):
    '''

    :param img: 3D array like (W, H, C)
    :param y_preds: nd array like (S*S, B, (5+class_num))
    :param ground_truths: json like object,
    {
        'large_vehicle':[[(341, 292),...,(346, 457), 0], [(341, 292),...,(346, 457), 1]...],
        'small_vehicle':[[(341, 292),...,(346, 457), 0], [(341, 292),...,(346, 457), 0]...]
    }
    :param categories: categories list like [small_vehicle, ...]
    :return: total losses
    '''
    loss = 0
    class_num = len(categories)
    W, H, C = img.shape
    S, _, B= int(np.sqrt(y_preds.shape[0])), int(np.sqrt(y_preds.shape[0])), int(y_preds.shape[1])
    ### define parameters lambda_coord, lambda_noob
    lambda_coord, lambda_noob = 0.5, 0.5

    ### 1.对预测的中心坐标做损失
    ground_truths_centeroid_idx = confirm_cell_index((W, H),S, cal_centeroid(ground_truths))
    for k in ground_truths_centeroid_idx.keys():
        idxs = ground_truths_centeroid_idx[k]
        for i, idx in enumerate(idxs):
            idx_x, idx_y = idx
            idx_flat = idx_x*S+idx_y
            res_bbox = y_preds[idx_flat, :, :]
            ground_truth_box = ground_truths[k][i]
            ious = []
            boxes = res_bbox
            boxes = np.array(boxes).reshape((B, class_num+5))
            for box in boxes:
                x_pred, y_pred, w_pred, h_pred, conf, class_prob_array = box[0], box[1], box[2], box[3], box[4], box[5:]
                pred_position = decode_position(img, cell=idxs, x=x_pred, y=y_pred, w=w_pred, h=h_pred, S=S)
                iou = cal_iou(ground_truth_box, pred_position)
                ious.append(iou)
        confirm_iou_idx = np.argmax(ious)
        confirm_box = ious[confirm_iou_idx]
        confirm_box = np.array(confirm_box).reshape(-1,)
        box_pts = decode_position(img, cell=idx, x=confirm_box[0], y=confirm_box[1],
                                  w=confirm_box[2], h=confirm_box[3], S=S)
        ground_truth_box_pts = ground_truth_box[0]

        ### 计算中心点和边界框loss
        centerAndBox_loss = cal_centeroid_loss(lambda_coord, ground_truth_box_pts, box_pts)

        ### 计算类别损失
        assert k in categories
        k_idx = categories.index(k)
        true_prob_array = np.zeros((class_num, ))
        true_prob_array[k_idx] = 1
        class_loss = calculateMSE(true_prob_array, class_prob_array)

        ### 计算置信度损失
        ground_truth_box_pts_idx = [confirm_cell_index((W, H), S,p) for p in np.array(ground_truth_box_pts)]
        ground_truth_box_pts_idx_1d = [twoD2oneD(pt, S) for pt in ground_truths_centeroid_idx]

        confidence_true = np.zeros(shape=(S, S))
        confidence_true[ground_truth_box_pts_idx_1d] = 1

        confidence_pred = np.array(y_pred[:, confirm_iou_idx, 4]).reshape((S, S))
        confidence_loss = calculateMSE(confidence_true, confidence_pred)
        loss = centerAndBox_loss + class_loss + confidence_loss
    return loss
if __name__ == '__main__':
    no = 'P10466'
    base_dir = 'data/rssrai2019_object_detection/train/'
    txt_path = f'{base_dir}labelTxt/labelTxt/{no}.txt'
    img_path = f'{base_dir}images/part6/{no}.png'
    img = cv2.imread(img_path)
    # show_img_with_boxes(img_path, txt_path, categories=None, save=False, show=True, use_modify=True)
    ground_truths = read_boxes_from_txt(txt_path)
    y_preds = np.random.randint(0, 10, size=(32*32, 5, 23)) / 10
    cats =gen_categories('data/color.csv')
    loss = cal_loss(img, y_preds, ground_truths, cats)