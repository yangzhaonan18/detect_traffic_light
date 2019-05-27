# -*- coding=utf-8 -*-
# py37
# opencv 3.4.1
# win10 + pycharm

import time

import cv2

from get_object import get_object
from judge_diretcton import judge_direction
from judge_color import judge_color


# judge_light 是一个总函数，同时调用了三个函数:get_object，judge_direction，udge_color
def judge_light(trafficLight):
    obj_bgr, cnt_max, cx, cy = get_object(trafficLight)  # 通过颜色来获取红绿灯区域。（目前只考虑了一个红绿灯的情况）
    if cnt_max is None:
       return "X", "X", 0
    direction, direction_conf = judge_direction(obj_bgr, cnt_max)  # 判断方向
    # print("\nResult : \ndirection, ratio = ", direction, direction_conf)
    color, color_conf = judge_color(trafficLight, obj_bgr, cx, cy)  # 判断颜色

    # print("color, color_conf = ", color, color_conf)
    conf = round(direction_conf * color_conf, 2)  # 将反向和颜色置信度的成绩作为最后的置信度。

    return color, direction, conf  # 单个字符， 单个字符， 小于1的小数


if __name__ == "__main__":
    trafficLight = cv2.imread('C:\\Users\\qcdz-003\\Pictures\\light\\001.jpg', cv2.IMREAD_COLOR)
    cv2.imshow("trafficLight", trafficLight)
    start = time.time()
    color, direction, conf = judge_light(trafficLight)  # 最终只使用这一行代码。
    end = time.time()
    print("judge_light()  use time = ", end - start)  # 0.00297
    print("color, direction, conf = ", color, direction, conf)
    print("the end !")
    cv2.waitKey()
