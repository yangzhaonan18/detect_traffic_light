# -*- conding=utf-8 -*-
# py37
# 判断红绿灯的形状和方向

from main_2_get_object import get_object
import cv2
import numpy as np


def cal_circle_xy(frame, x, y, radius):
    x1 = x - radius if x - radius > 0 else 0
    x2 = x + radius if x + radius < frame.shape[1] else frame.shape[1]  # cv里面横坐标是x 是shape[1]
    y1 = y - radius if y - radius > 0 else 0
    y2 = y + radius if y + radius < frame.shape[0] else frame.shape[0]  # cv里面纵坐标是y 是shape[0]
    return int(x1), int(x2), int(y1), int(y2)


def cal_point(SomeBinary, x, y, radius):  # 返回最大方向的编号int

    x = int(x)
    y = int(y)
    x1, x2, y1, y2 = cal_circle_xy(SomeBinary, x, y, radius)
    S00 = SomeBinary[y1:y, x1:x]  # 计算面积时，使用二值图，左上
    S01 = SomeBinary[y1:y, x:x2]  # 右上
    S10 = SomeBinary[y:y2, x1:x]  # 左下
    S11 = SomeBinary[y:y2, x:x2]  # 右下

    SS00 = np.sum(S00)
    SS01 = np.sum(S01)
    SS10 = np.sum(S10)
    SS11 = np.sum(S11)

    value = [SS00, SS01, SS10, SS11]
    value.sort(reverse=True)  # 将面积大的放在前面

    print("\nSS00, SS01 , SS10, SS11 = ", SS00, SS01, SS10, SS11)
    if SS00 in value[0:2] and SS10 in value[0:2]:
        return "R"  # right
    elif SS01 in value[0:2] and SS11 in value[0:2]:  # 箭头右侧需要补齐的东西多
        return "L"  # left
    elif SS10 in value[0:2] and SS11 in value[0:2]:
        return "D"  # direct
    else:
        return "X"  # circle


def judge_direction(obj_bgr, cnt_max):

    cnt = np.array(cnt_max)
    ((x, y), radius) = cv2.minEnclosingCircle(cnt)  # 确定面积最大的轮廓的外接圆  返回圆心坐标和半径
    if radius < 3:
        print("\nminEnclosingCircle radius = ", radius, "< 5 , so NO direction !! ", )
        return "X", 0.0

    area = cv2.contourArea(cnt)  # 轮廓面积
    hull = cv2.convexHull(cnt)  # 计算出凸包形状(计算边界点)
    hull_area = cv2.contourArea(hull)  # 计算凸包面积
    solidity = round(float(area) / hull_area, 2)  # 自己的面积 / 凸包面积  凸度
    circularity = round(hull_area / (np.pi * pow(radius, 2)), 2)  # 自己的面积/外接圆的面积

    print("solidity = ", solidity)
    print("circularity = ", circularity)

    direction = "X"  # X  表示是未知方向，或者不能正确判断
    ratio = 0
    threshold01 = 0.9  # solidity  远大越可能是圆
    threshold03 = 0.5  # solidity * circularity  越小越可能是箭头

    if solidity > threshold01 and circularity > 0.5:  # 当solidity度很大,且圆形度不是很小时 （考虑椭圆的情况），直接判断为圆
        direction = "C"  # 判断 为圆形灯
        ratio = solidity  # 将solidity度作为 圆灯的概率返回
    elif solidity > threshold01 and circularity <= 0.5:  # 凸度很大，但圆形度很小时，说明不是圆
        direction = "X"
        ratio = 0.0
    else:  # 当solidity度处于中间值时，判断为箭头，并计算概率
        # （规定圆形度与凸度成绩小于0.7时，概率为1。大于0.7时，概率小于1）
        cnts_ColorThings = cv2.drawContours(obj_bgr.copy(), [cnt], -1, (255, 255, 255), -1)
        hull_ColorThings = cv2.drawContours(obj_bgr.copy(), [hull], -1, (255, 255, 255), -1)
        BinThings = ~cnts_ColorThings & hull_ColorThings & ~obj_bgr  # 找到凸包与原图之间的差

        # cv2.imshow("obj_bgr", obj_bgr)
        # cv2.imshow("cnts_ColorThings = ", cnts_ColorThings)
        # cv2.imshow("BinThings = ", BinThings)
        direction = cal_point(BinThings, int(x), int(y), radius)  # 判断方向（圆心和半径）根据凸包与原图之间的差，计算箭头的方向
        x = solidity * circularity
        print("solidity * circularity = ", x)
        if x < threshold03:   # 乘积小于这个值，判断为箭头，概率为1。
            ratio = 1
        else:
            ratio = (x - 1) / (threshold03 - 1)
    # 当圆形度特别小时，判断为处于异常，输出X 0
    return direction, ratio


if __name__ == '__main__':
    trafficLight = cv2.imread('C:\\Users\\qcdz-003\\Pictures\\light\\027.jpg', cv2.IMREAD_COLOR)
    obj_bgr, cnt_max = get_object(trafficLight)
    direction = judge_direction(obj_bgr, cnt_max)

    # cv2.imshow("trafficLight", trafficLight)
    # cv2.imshow("obj_bgr", obj_bgr)

    # print("direction", direction)
    cv2.waitKey()
    print("the end")
