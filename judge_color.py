from get_object import get_object
import cv2
import numpy as np


def judge_light_color(obj_bgr):
    #  将图片转成HSV的
    obj_hsv = cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2HSV)
    #  从H通道中提取出颜色的平均值
    H = obj_hsv[:, :, 0]
    H = np.squeeze(np.reshape(H, (1, -1)))

    # print(H)  # 这里要注意红色H值得范围很特别，特别小和特别大都是红色，直接取平均值会出现判断错误的情况。
    H_value = np.array([])
    for i in H:
        if i > 0 and i < 124:
            H_value = np.append(H_value, i)
        elif i > 155:  # 将大于155的红色的H值，都看成是1，方便处理
            H_value = np.append(H_value, 1.0)
    H_avg = np.average(H_value)

    #  根据这个值来判断是什么颜色， RYG
    if H_avg <= 18 or H_avg >= 156:
        color = "R"
    elif H_avg >= 18 and H_avg <= 34:
        color = "Y"
    elif H_avg >= 35 and H_avg <= 100:
        color = "G"
    else:
        color = "X"

    return color


def judge_light_position(trafficLight, cx, cy):
    h, w, d = trafficLight.shape
    # print(trafficLight.shape)
    position = "0"
    if max(w / h, h / w) > 2:  # 根据位置来判断颜色
        if h > w:  # 竖着三个灯
            if cy < h / 3:  # 灯的中心在上方
                position = "1"
            elif cy < 2 * h / 3:  # 灯的中心在中间
                position = "2"
            else:  # 灯的中心在下方
                position = "3"

        else:  # 横着三个灯
            if cx < w / 3:  # 灯的中心在左方
                position = "1"
            elif cx < 2 * w / 3:  # 灯的中心在中间
                position = "2"
            else:  # 灯的中心在右方
                position = "3"
    return position


def judge_color(trafficLight, obj_bgr, cx, cy):
    position = judge_light_position(trafficLight, cx, cy)  # 判断位置
    color = judge_light_color(obj_bgr)  # 判断颜色
    if (position == "1" and color == "R") or (position == "2" and color == "Y") or (position == "3" and color == "G"):
        color_conf = 1.0
    elif position == "2" and color == "R":
        color = "Y"  # 如何在中间就纠正为黄色
        color_conf = 0.9
    elif position == "1" and color == "Y":
        color = "R"  # 如何在中间就纠正为黄色
        color_conf = 0.9
    else:
        color_conf = 0.8
    return color, color_conf


if __name__ == '__main__':
    trafficLight = cv2.imread('C:\\Users\\qcdz-003\\Pictures\\light\\023.jpg', cv2.IMREAD_COLOR)
    obj_bgr, cnt_max, cx, cy = get_object(trafficLight)

    color, color_conf = judge_color(trafficLight, obj_bgr, cx, cy)   # 一行代码

    print("color, color_conf = ", color, color_conf)
    print("The end")
