# -*- coding=utf-8 -*-
# py37
# opencv 3.4.2
# win10 + pycharm

# 程序作用：判断红绿灯灯颜色和方向。
# 具体说明如下：
# 0. 为了方便复制程序，已经将所有的程序放入这个py文件中了，程序测试正常。修改前请备份，谢谢。
# 1. 输入是的一个只包含红绿灯的图片（注意，不是摄像头读取的整个图），输出是红绿灯的颜色+方向+置信度。
# 2. 颜色用字符表示：红色是"R"， 黄色是"Y", 绿色是"G", 未知或不能判断是"X"。
# 3. 方向用字符表示：圆饼灯是"C",左转箭头是"L",直行箭头是"D",右转箭头是"R",未知或不能判断是"X"。
# 4. 置信度用[0,1]之间的两位小数表示，数值越大，正确率越高。如0.86


import cv2
import time
import numpy as np

import matplotlib.pyplot as plt

# 将使用Kmeans聚类用于分割，最后生成提取红绿灯使用的mask。
def seg_kmeans_color(img, k=2):  # 使用的是opencv中的Kmean算法

    # 变换一下图像通道bgr->rgb，否则很别扭啊
    h, w, d = img.shape  # (595, 1148, 3)  # 0.47 聚类的时间和图像的面积大小成正比，即与图像的点数量成正比。
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    # 3个通道展平
    img_flat = img.reshape((img.shape[0] * img.shape[1], 3))  # 一个通道一列，共三列。每行都是一个点。
    img_flat = np.float32(img_flat)
    # print("len of img_flat = ", len(img_flat))
    # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # 聚类

    compactness, labels, centers = cv2.kmeans(img_flat, k, None, criteria, 10, flags)
    # print(" len of  labels = ", len(labels))
    # print(labels)  # 每个点的标签
    # print(centers) #  两类的中心，注意是点的值，不是点的坐标
    labels = np.squeeze(labels)
    img_output = labels.reshape((img.shape[0], img.shape[1]))
    L1 = max(int(img.shape[1] / 4), 2)  # 当图像只有四个像素时， L1等于1 会出现-1：的异常异常，所以需要改为2，

    # 取每个四个角上的点
    boundary00 = np.squeeze(img_output[0:L1, 0: L1]. reshape((1, -1)))
    boundary01 = np.squeeze(img_output[0:L1, - L1:]. reshape((1, -1)))
    boundary10 = np.squeeze(img_output[- L1:, 0:L1]. reshape((1, -1)))
    boundary11 = np.squeeze(img_output[- L1:, - L1-1]. reshape((1, -1)))

    # 取中间一个正方形内的点
    inter = img_output[L1: -L1, L1: -L1]
    boundary_avg = np.average(np.concatenate((boundary00, boundary01, boundary10, boundary11), axis=0))
    inter_avg = np.average(inter)
    # print("boundary_avg", boundary_avg)
    # print("inter_avg", inter_avg)

    if k == 2 and boundary_avg > inter_avg:  # 如果聚类使得边缘类是1，即亮的话，需要纠正颜色（所有的标签）。
        img_output = abs(img_output - 1)  # 将边缘类（背景）的标签值设置为0，中间类（中间类）的值设置为1.

    img_output = np.array(img_output, dtype=np.uint8)  # jiang

    return img_output


# 下面是实现 judge_color 函数的各个子函数
def judge_light_color(obj_bgr):
    #  将图片转成HSV的
    obj_hsv = cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2HSV)
    #  从H通道中提取出颜色的平均值
    H = obj_hsv[:, :, 0]
    H = np.squeeze(np.reshape(H, (1, -1)))
    print(H)  # 这里要注意红色H值得范围很特别，特别小和特别大都是红色，直接取平均值会出现判断错误的情况。
    H_value = np.array([])
    for i in H:
        if i > 0 and i < 124:
            H_value = np.append(H_value, i)
        elif i > 155:  # 将大于155的红色的H值，都看成是1，方便处理
            H_value = np.append(H_value, 1.0)
    H_avg = np.average(H_value)
    #  根据这个值来判断是什么颜色， RYG
    if H_avg <= 18:
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


# 下面是实现judge_direction函数的各个子函数
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

    # print("solidity = ", solidity)
    # print("circularity = ", circularity)

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


#  下面是实现 get_object函数的各个子函数。
def find_contours_max(img_bin_color):
    #  找最大轮廓的正矩形
    gray = cv2.cvtColor(img_bin_color, cv2.COLOR_BGR2GRAY)  # 转成灰色图像
    ret, BinThings = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 灰色图像二值化（变黑白图像）
    _, contours, hierarchy = cv2.findContours(BinThings, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # for i in range(len(contours) - 1):  # 绘制每一个轮廓
    #     cv2.drawContours(frame_copy, [contours[i+1]], -1, (0, 0, 255), 2)

    # if len(contours) > 0:
    #     cnt_max = max(contours, key=cv2.contourArea)  # 找到面积最大的轮廓
    #     color_aera = cv2.contourArea(cnt_max)

    if len(contours) > 0:
        cnt_max = max(contours, key=cv2.contourArea)  # 找到面积最大的轮廓
        # cv2.drawContours(img_bin_color, [cnt_max], -1, (0, 255, 255), 2)
        # cv2.imshow("img_bin_color", img_bin_color)
        # # ((x, y), radius) = cv2.minEnclosingCircle(np.array(cnt_max))
        # cv2.circle(img_bgr, (int(x), int(y)), int(radius), (0, 0, 255), 2)
        # print(((x, y), radius))

        # print("cnt_max", cnt_max)
        return cnt_max
    return None


def crop_max_region(frame_bgr, img_bin_color):  # frame_bgr, img_bgr
    """
    :param frame_bgr: 用于裁剪的原始图片
    :param img_bin_color: # 用于腐蚀，寻找颜色区域的
    :return: 在原图上裁剪下来的区域。
    """

    cnt_max = find_contours_max(img_bin_color)
    if cnt_max is not None:
        x, y, w, h = cv2.boundingRect(np.array(cnt_max))  # 正外界矩形
        # print("x, y, w, h = ", x, y, w, h)
        # cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame_crop = frame_bgr[y: y + h, x: x + w]

        # cv2.imshow("frame_crop6", frame_crop)
        cnt_max = find_contours_max(frame_crop)

        #  将原始图像翻转并判断形状和方向

        return frame_crop, cnt_max, x, y, w, h
    return None, None, None, None, None, None


def get_color_region(frame_bgr, frame_hsv, color="RYG"):
    # cv2.imshow("frame_bgr", frame_bgr)
    # colorLower = np.array([0, 43, 46], dtype=np.uint8)  # 非黑白灰的
    # colorUpper = np.array([180, 255, 255], dtype=np.uint8)
    try:
        redLower01 = np.array([0, 43, 46], dtype=np.uint8)  # 部分红 和橙黄绿
        redUpper01 = np.array([124, 255, 255], dtype=np.uint8)
        red_mask02_and_othersLower = np.array([156, 43, 46], dtype=np.uint8)   # 部分红
        red_mask02_and_othersUpper = np.array([180, 255, 255], dtype=np.uint8)

        red_mask01 = cv2.inRange(frame_hsv, redLower01,  redUpper01)
        red_mask02_and_others = cv2.inRange(frame_hsv, red_mask02_and_othersLower,  red_mask02_and_othersUpper)
        mask = None
        if color == "RYG":
            mask = red_mask01 + red_mask02_and_others
        # print("mask.shape = ", mask.shape)  # (26, 23)

        BinColors = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
        # cv2.imshow("BinColors", BinColors)
        # BinColors = cv2.GaussianBlur(BinColors, (5, 5), 0)  # 彩色图时 高斯消除噪音  # 很耗时，不建议使用

        # dst = cv2.erode(dst, None, iterations=2)  # 腐蚀操作
        kernel = np.ones((3, 3), np.uint8)
        # BinColors = cv2.morphologyEx(BinColors, cv2.MORPH_OPEN, kernel)  # 开运算
        # kernel = np.ones((10, 10), np.uint8)
        BinColors = cv2.morphologyEx(BinColors, cv2.MORPH_CLOSE, kernel)  # 闭运算

        # cv2.imshow("mask ", mask)  # 这是一个二值图
        # cv2.imshow("crop_frame ", crop_frame)  # 这是一个二值图

        # cv2.imshow("BinColors", BinColors)  # 原图
        # cv2.imshow("crop_frame", crop_frame)  # 原图
        # cv2.imshow("BinColors", BinColors)  # 其中的彩色区域
        # cv2.imshow("img_bin_color ", img_bin_color)

        # cv2.waitKey(0)
        return crop_max_region(frame_bgr, BinColors)
    except:
        return None, None, None, None, None, None


def get_object(frame_bgr):
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    frame_crop_bgr, _, x1, y1, w1, h1 = get_color_region(frame_bgr, frame_hsv, color="RYG")  # 获取彩色区域(裁剪一个矩形)
    if frame_crop_bgr is None:
        print("no object in get_object")
        return None, None, None, None
    frame_kmeans = seg_kmeans_color(frame_crop_bgr)  # k menan k=2

    # cv2.imshow("frame_crop_bgr", frame_crop_bgr)
    # cv2.imshow("frame_kmeans", frame_kmeans)

    frame_new_bgr = cv2.bitwise_and(frame_crop_bgr, frame_crop_bgr, mask=frame_kmeans)  # 使用Kmean之后的结果比直接使用颜色阈值分割更可靠。

    kernel = np.ones((3, 3), np.uint8)
    frame_new_bgr = cv2.morphologyEx(frame_new_bgr, cv2.MORPH_OPEN, kernel)
    frame_max_bgr, cnt_max, x2, y2, w2, h2 = crop_max_region(frame_new_bgr, frame_new_bgr)  # Kmeans之后的中间颜色区域提取。

    # light = frame_bgr[y1 + y2: y1 + y2 + h2, x1 + x2: x1 + x2 + w2]
    # cv2.imshow("light =", light)

    cx = int(x1 + x2 + w2 / 2)  # 灯的中心点坐标
    cy = int(y1 + y2 + h2 / 2)

    # 显示结果
    b, g, r = cv2.split(frame_bgr)
    frame_rgb = cv2.merge([r, g, b])
    frame_crop_rgb = frame_crop_bgr[:, :, (2, 1, 0)]
    frame_new_rgb = frame_new_bgr[:, :, (2, 1, 0)]
    frame_max_rgb = frame_max_bgr[:, :, (2, 1, 0)]

    # plt 显示图片RGB的顺序
    plt.subplot(161), plt.imshow(frame_rgb), plt.title('rgb')  # plt是RGB的顺序，CV是BGR的顺序。
    plt.subplot(162), plt.imshow(frame_crop_rgb), plt.title('crop_rgb')  # plt是RGB的顺序，CV是BGR的顺序。
    plt.subplot(163), plt.imshow(frame_kmeans, 'gray'), plt.title('kmeans')
    plt.subplot(164), plt.imshow(frame_new_rgb, 'gray'), plt.title('new_rgb')
    plt.subplot(165), plt.imshow(frame_max_rgb), plt.title('max_rgb')
    plt.show()

    # cv显示图片 BGR的通道顺序
    # cv2.imshow("frame", frame)  # 原图
    # cv2.imshow("crop_frame", frame_crop)  # 原图
    # cv2.waitKey(0)
    # frame_crop_gray = cv2.cvtColor(frame_crop_bgr, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("frame_crop_gray", frame_crop_gray)
    # cv2.waitKey()
    return frame_max_bgr, cnt_max, cx, cy


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
    trafficLight = cv2.imread('C:\\Users\\qcdz-003\\Pictures\\light\\025.jpg', cv2.IMREAD_COLOR)
    start = time.time()
    color, direction, conf = judge_light(trafficLight)  # 最终只使用这一行代码。
    end = time.time()
    print("\n####################")
    print("judge_light()  use time = ", end - start)  # 0.00297
    print("color, direction, conf = ", color, direction, conf)
    print("the end !")
