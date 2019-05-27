import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from kmeans import seg_kmeans_color


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
        # cv2.drawContours(img_bgr, [cnt_max], -1, (0, 255, 255), 2)
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
        redUpper01 = np.array([100, 255, 255], dtype=np.uint8)
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
        kernel = np.ones((5, 5), np.uint8)
        img_bin_color = cv2.morphologyEx(BinColors, cv2.MORPH_OPEN, kernel)  # 开运算
        # kernel = np.ones((10, 10), np.uint8)
        img_bin_color = cv2.morphologyEx(img_bin_color, cv2.MORPH_CLOSE, kernel)  # 闭运算

        # cv2.imshow("mask ", mask)  # 这是一个二值图
        # cv2.imshow("crop_frame ", crop_frame)  # 这是一个二值图

        # cv2.imshow("BinColors", BinColors)  # 原图
        # cv2.imshow("crop_frame", crop_frame)  # 原图
        # cv2.imshow("BinColors", BinColors)  # 其中的彩色区域
        # cv2.imshow("img_bin_color ", img_bin_color)

        # cv2.waitKey(0)
        return crop_max_region(frame_bgr, img_bin_color)
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

    # # 显示结果
    # b, g, r = cv2.split(frame_bgr)
    # frame_rgb = cv2.merge([r, g, b])
    # frame_crop_rgb = frame_crop_bgr[:, :, (2, 1, 0)]
    # frame_new_rgb = frame_new_bgr[:, :, (2, 1, 0)]
    # frame_max_rgb = frame_max_bgr[:, :, (2, 1, 0)]
    #
    # # plt 显示图片RGB的顺序
    # # plt.subplot(161), plt.imshow(frame_rgb), plt.title('rgb')  # plt是RGB的顺序，CV是BGR的顺序。
    # # plt.subplot(162), plt.imshow(frame_crop_rgb), plt.title('crop_rgb')  # plt是RGB的顺序，CV是BGR的顺序。
    # # plt.subplot(163), plt.imshow(frame_kmeans, 'gray'), plt.title('kmeans')
    # # plt.subplot(164), plt.imshow(frame_new_rgb, 'gray'), plt.title('new_rgb')
    # # plt.subplot(165), plt.imshow(frame_max_rgb), plt.title('max_rgb')
    # # plt.show()

    # cv显示图片 BGR的通道顺序
    # cv2.imshow("frame", frame)  # 原图
    # cv2.imshow("crop_frame", frame_crop)  # 原图
    # cv2.waitKey(0)
    # frame_crop_gray = cv2.cvtColor(frame_crop_bgr, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("frame_crop_gray", frame_crop_gray)
    # cv2.waitKey()
    return frame_max_bgr, cnt_max, cx, cy


if __name__ == '__main__':
    frame_bgr = cv2.imread('C:\\Users\\qcdz-003\\Pictures\\light\\023.jpg', cv2.IMREAD_COLOR)

    frame_max_bgr, cnt = get_object(frame_bgr)
    # cv2.imshow("asdf", frame_max_bgr)
    cv2.waitKey()
