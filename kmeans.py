# -*- coding=utf-8 -*-
# K-means 图像分割 连续图像分割 （充分利用红绿灯的特点）

import cv2
import matplotlib.pyplot as plt
import numpy as np


def seg_kmeans_color(img, k=2):

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
    print("boundary_avg", boundary_avg)
    print("inter_avg", inter_avg)

    if k == 2 and boundary_avg > inter_avg:  # 如果聚类使得边缘类是1，即亮的话，需要纠正颜色（所有的标签）。
        img_output = abs(img_output - 1)  # 将边缘类（背景）的标签值设置为0，中间类（中间类）的值设置为1.

    img_output = np.array(img_output, dtype=np.uint8)  # jiang

    return img_output


if __name__ == '__main__':

    img = cv2.imread('C:\\Users\\qcdz-003\\Pictures\\light\\007.jpg', cv2.IMREAD_COLOR)

    img_output = seg_kmeans_color(img, k=2)  # 2 channel picture

    # print(img_output.shape)  # (244, 200)
    # 显示结果

    plt.subplot(121), plt.imshow(img), plt.title('input')
    plt.subplot(122), plt.imshow(img_output, 'gray'), plt.title('kmeans')
    plt.show()



