# 红绿灯检测

## 1.环境
- py37
- opencv 3.4.2
- win10 + pycharm

## 2.程序作用：判断红绿灯的颜色和方向。

### 具体说明如下：

0. 为了方便复制程序，已经将所有的程序放入这个py文件中了，程序测试正常。修改前请备份，谢谢。
1. 输入是的一个只包含红绿灯的图片（注意，不是摄像头读取的整个图），输出是红绿灯的颜色+方向+置信度。
2. 颜色用字符表示：红色是"R"， 黄色是"Y", 绿色是"G", 未知或不能判断是"X"。
3. 方向用字符表示：圆饼灯是"C",左转箭头是"L",直行箭头是"D",右转箭头是"R",未知或不能判断是"X"。
4. 置信度用[0,1]之间的两位小数表示，数值越大，正确率越高。如0.86

## 3. 函数说明

## 4. demo

https://www.bilibili.com/video/av53928579/
