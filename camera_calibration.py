#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------

    @   Author  :       Max_PJB
    @   date    :       2021/3/23 15:41
    @   IDE     :       PyCharm
    @   GitHub  :       https://github.com/JackyPJB
    @   Contact :       pengjianbiao@hotmail.com
-------------------------------------------------
    Description :       相机标定
-------------------------------------------------
"""

__author__ = 'Max_Pengjb'

import numpy as np
import cv2
import glob

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((6 * 8, 3), np.float32)  # 7x9的格子 此处参数根据使用棋盘格规格进行修改
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
# 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

objpoints = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob('calibrate/*.jpg')  # 文件存储路径，存储需要标定的摄像头拍摄的棋盘格图片
print(images)
for fname in images:
    img = cv2.imread(fname)
    print(img.shape)
    # img = np.rot90(img)
    # cv2.imshow('haha', img)
    # cv2.waitKey()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (6, 8), None)
    print(ret)
    if ret:
        objpoints.append(objp)

        # 在原角点的基础上寻找亚像素角点
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if corners2.any():
            img_points.append(corners2)
        else:
            img_points.append(corners)

        # 绘制角点并显示
        img = cv2.drawChessboardCorners(img, (6, 8), corners2, ret)
        cv2.imshow('img', img)
        # cv2.waitKey()
# 标定
img = cv2.imread('calibrate/1.jpg')
img_size = (img.shape[1], img.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, img_points, img_size, None, None)

print("ret:", ret)  # ret为bool值
print("mtx:\n", mtx)  # 内参数矩阵
print("dist:\n", dist)  # 畸变系数 distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)  # 旋转向量，外参数
print("tvecs:\n", tvecs)  # 平移向量，外参数
# return mtx, dist

cv2.imshow("before",img)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,img_size,1,img_size)
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# print(help(cv2.undistort))
cv2.imshow("after", dst)
cv2.waitKey()

cv2.destroyAllWindows()
