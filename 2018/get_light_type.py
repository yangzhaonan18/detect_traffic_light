#! -*- coding=utf-8 -*-


import cv2
import numpy as np


# judge_color

def find_mask(frame, color):
    blackLower01 = np.array([0, 0, 0], dtype=np.uint8)  # 黑的阈值 标准H：0:180 S:0:255 V:0:46:220
    blackUpper01 = np.array([180, 255, 90], dtype=np.uint8)
    blackLower02 = np.array([0, 0, 46], dtype=np.uint8)  # 灰的阈值 标准H：0:180 S:0:43 V:0:46:220
    blackUpper02 = np.array([180, 43, 45], dtype=np.uint8)  # 灰色基本没用

    redLower01 = np.array([0, 80, 80], dtype=np.uint8)  # 红色的阈值 标准H：0-10 and 160-179 S:43:255 V:46:255
    redUpper01 = np.array([10, 255, 255], dtype=np.uint8)
    redLower02 = np.array([156, 80, 80], dtype=np.uint8)  # 125 to 156
    redUpper02 = np.array([180, 255, 255], dtype=np.uint8)

    greenLower = np.array([40, 80, 80], dtype=np.uint8)  # 绿色的阈值 标准H：35:77 S:43:255 V:46:255
    greenUpper = np.array([95, 255, 255], dtype=np.uint8)  # V 60 调整到了150

    blueLower = np.array([105, 120, 46], dtype=np.uint8)  # 蓝H:100:124 紫色H:125:155
    blueUpper = np.array([130, 255, 255], dtype=np.uint8)

    yellowLower = np.array([24, 80, 80], dtype=np.uint8)  # 黄色的阈值 标准H：26:34 S:43:255 V:46:255
    yellowUpper = np.array([36, 255, 255], dtype=np.uint8)  # 有的图 黄色变成红色的了
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red1_mask = cv2.inRange(hsv, redLower01, redUpper01)  # 根据阈值构建掩膜, 红色的两个区域
        red2_mask = cv2.inRange(hsv, redLower02, redUpper02)
        red_mask = red1_mask + red2_mask

        black01_mask = cv2.inRange(hsv, blackLower01, blackUpper01)  # 根据阈值构建掩膜,黑色的区域
        black02_mask = cv2.inRange(hsv, blackLower02, blackUpper02)  # 根据阈值构建掩膜,黑色的区域
        black_mask = black01_mask + black02_mask

        yellow_mask = cv2.inRange(hsv, yellowLower, yellowUpper)  # 根据阈值构建掩膜, 黄色的区域
        green_mask = cv2.inRange(hsv, greenLower, greenUpper)  # 根据阈值构建掩膜, 绿色的区域

        blue_mask = cv2.inRange(hsv, blueLower, blueUpper)
        if color == "black":
            mask = black_mask
        elif color == "yellow":
            mask = yellow_mask
        elif color == "red":
            mask = red_mask
        elif color == "green":
            mask = green_mask
        elif color == "blue":
            mask = blue_mask
        elif color == "red+blue":
            mask = red_mask + blue_mask
        elif color == "yellow+green":
            mask = yellow_mask + green_mask
        elif color == "red+yellow+green":
            mask = red_mask + yellow_mask + green_mask
        else:
            mask = None
        return mask

    except:
        return None


def find_color_aera(Crop_frame, color):

    mask = find_mask(Crop_frame, color)
    # mask = cv2.dilate(mask, None, iterations=1)  # 膨胀操作，其实先腐蚀再膨胀的效果是开运算，去除噪点
    # mask = cv2.erode(mask, None, iterations=num)  # 腐蚀操作
    BinColors = cv2.bitwise_and(Crop_frame, Crop_frame, mask=mask)  # 提取感兴趣的颜色区域  背景黑色+彩色的图像

    dst = cv2.GaussianBlur(BinColors, (3, 3), 0)  # 彩色图时 高斯消除噪音
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)  # 转成灰色图像

    ret, BinThings = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 灰色图像二值化（变黑白图像）

    # cv2.imshow("BinThings", BinThings)
    # cv2.waitKey(0)

    _, contours, hierarchy = cv2.findContours(BinThings, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 边界是封闭的 del first BinThings,
    if len(contours) > 0:
        cnt_max = max(contours, key=cv2.contourArea)  # 找到面积最大的轮廓
        color_aera = cv2.contourArea(cnt_max)
        # cv2.imshow("BinThings", BinColors)
        # cv2.waitKey(0)
    else:
        color_aera = 0
    return color_aera


def judge_color(Crop_frame):
    yellow_area = find_color_aera(Crop_frame, "yellow")
    red_area = find_color_aera(Crop_frame, "red")
    green_area = find_color_aera(Crop_frame, "green")
    print("\nyellow_area,  red_area, green_area = ", yellow_area, red_area, green_area)
    ratio = 0.05  # 调参
    if max(yellow_area, red_area, green_area) > ratio * pow(min(Crop_frame.shape[0], Crop_frame.shape[1]), 2):  # 提取出的图像的面积不能太小

        if yellow_area > red_area and yellow_area > green_area:
            return "yellow"
        if red_area > yellow_area and red_area > green_area:
            return "red"
        if green_area > yellow_area and green_area > red_area:
            return "green"
    else:
        print("\n max area = ", max(yellow_area, red_area, green_area), " < ",  ratio, " * min^2 =", 0.1 * pow(min(Crop_frame.shape[0], Crop_frame.shape[1]), 2))
        return "NO"

# judge_direction


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


def find_cnt(Crop_frame, mask):  # 找轮廓

    mask = cv2.dilate(mask, None, iterations=1)
    BinColors = cv2.bitwise_and(Crop_frame, Crop_frame, mask=mask)  # 提取感兴趣的颜色区域  背景黑色+彩色的图像
    dst = cv2.GaussianBlur(BinColors, (3, 3), 0)  # 彩色图时 高斯消除噪音
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)  # 转成灰色图像
    ret, BinThings = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 灰色图像二值化（变黑白图像）
    _, contours, hierarchy = cv2.findContours(BinThings, cv2.RETR_CCOMP,  cv2.CHAIN_APPROX_SIMPLE)  # 边界是封闭的  linux  del first one
    cv2.imshow("asdf",BinThings )
    if len(contours) > 0:
        cnt_max = max(contours, key=cv2.contourArea)  # 找到面积最大的轮廓
        return cnt_max
    else:
        return None


def judge_direction(Crop_frame):  # 判断方向
    size = 50
    Crop_frame = cv2.resize(Crop_frame, (size, int(size * Crop_frame.shape[0] / Crop_frame.shape[1])),
                            interpolation=cv2.INTER_CUBIC)  # 将裁剪下来的图片调整到固定的尺寸
    # cv2.imshow("Crop_frame", Crop_frame)
    # cv2.waitKey(0)
    mask = find_mask(Crop_frame, "red+yellow+green")
    mask = cv2.erode(mask, None, iterations=2)  # 腐蚀操作
    cnt_max = find_cnt(Crop_frame, mask)  # 找到最大轮廓

    if cnt_max is None:
        print("\nno color contour, so NO direction !!")
        return "NO"
    solidity = 0.85
    direction = "NO"
    ilter_num = 1
    min_s = 0.8  # 比例最小值，低于这个值，直接判断为箭头信号灯
    max_s = 0.94  # 比例最大值，高于这个值，直接判断为圆形信号灯
    max_item = 4  # 最大腐蚀次数，比例介于最大值和最小值之间时，通过腐蚀来判断。
    while solidity > min_s and solidity < max_s and ilter_num < max_item:
        print("ilter_num = ", ilter_num)

        cnts = np.array(cnt_max)
        # cnts = cnt_max
        ((x, y), radius) = cv2.minEnclosingCircle(cnts)  # 确定面积最大的轮廓的外接圆  返回圆心坐标和半径
        if radius < 5:
            print("\nminEnclosingCircle radius = ", radius, "< 5 , so NO direction !! ", )
            return "NO"
        x = int(x)
        y = int(y)
        area = cv2.contourArea(cnts)  # 轮廓面积
        hull = cv2.convexHull(cnts)  # 计算出凸包形状(计算边界点)
        hull_area = cv2.contourArea(hull)  # 计算凸包面积
        solidity = float(area) / hull_area  # 轮廓面积 / 凸包面积
        print("\nsolidity = ", solidity)
        if solidity > max_s:
            direction = "C"  # circle
            break
        # elif solidity < min_s:
        #     direction = ""  # others type not light
        #     # print("direction = D ", solidity)
        #     break

        cnts_ColorThings = Crop_frame.copy()
        hull_ColorThings = Crop_frame.copy()
        cnts_ColorThings = cv2.drawContours(cnts_ColorThings, [cnts], -1, (255, 255, 255), -1)
        hull_ColorThings = cv2.drawContours(hull_ColorThings, [hull], -1, (255, 255, 255), -1)
        BinThings = ~cnts_ColorThings & hull_ColorThings & ~Crop_frame  # 找到凸包与原图之间的差



        direction = cal_point(BinThings, x, y, radius)  # （圆心和半径）根据凸包与原图之间的差，计算箭头的方向
        ilter_num += 1
        cnt_max = find_cnt(Crop_frame, mask)  # 找最大轮廓

        if cv2.contourArea(cnt_max) < size * size / 5:  # 腐蚀到，剩余面积很小时，结束。
            break
    return direction


def get_light_type(crop_frame):
    color = judge_color(crop_frame)  # 颜色
    direction = judge_direction(crop_frame)  # 方向
    print("\n color, direction = ", color, direction)

    if color == "red" and direction == "R":
        return "15"
    elif color == "green" and direction == "R":
        return "16"
    elif color == "yellow" and direction == "R":
        return "17"
    elif color == "red" and direction == "D":
        return "18"
    elif color == "green" and direction == "D":
        return "19"
    elif color == "yellow" and direction == "D":
        return "20"
    elif color == "red" and direction == "L":
        return "21"
    elif color == "green" and direction == "L":
        return "22"
    elif color == "yellow" and direction == "L":
        return "23"
    elif color == "red" and direction == "C":
        return "24"
    elif color == "green" and direction == "C":
        return "25"
    elif color == "yellow" and direction == "C":
        return "26"
    else:
        return "NO light or cant judge"
