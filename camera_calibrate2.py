#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------

    @   Author  :       Max_PJB
    @   date    :       2021/3/23 16:39
    @   IDE     :       PyCharm
    @   GitHub  :       https://github.com/JackyPJB
    @   Contact :       pengjianbiao@hotmail.com
-------------------------------------------------
    Description :       
-------------------------------------------------
"""

__author__ = 'Max_Pengjb'

# coding:utf-8
import numpy as np
import cv2
import pickle


def calibrate_camera():
    # 将每个校准图像映射到棋盘角的数量
    objp_dict = {
        1: (7, 9),
        2: (7, 9),
        3: (7, 9),
        4: (7, 9),
        5: (7, 9),
        6: (7, 9),
        7: (7, 9),
        8: (7, 9),
        9: (7, 9),
    }

    # 用于校准的对象点和角点列表
    objp_list = []  # 存储3D点
    corners_list = []  # 存储2D点

    # 浏览所有图像并找到角点
    for k in objp_dict.keys():
        nx, ny = objp_dict[k]

        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        # 遍历每一幅棋盘格板，获取其对应的内角点数目，即 nx * ny。
        # 用数组的形式来保存每一幅棋盘格板中所有内角点的三维坐标。
        # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        # print(objp)，部分输出如下：
        # [[0. 0. 0.]
        #  [1. 0. 0.]
        #  [2. 0. 0.]
        #  [3. 0. 0.]
        #  [4. 0. 0.]
        #  [5. 0. 0.]
        #  [6. 0. 0.]
        #  [7. 0. 0.]
        #  [8. 0. 0.]
        #  [0. 1. 0.]
        #  [1. 1. 0.]
        #  [2. 1. 0.]
        #  ...

        fname = 'calibrate/%s.jpg' % str(k)
        img = cv2.imread(fname)
        cv2.imshow('origin', img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        print('ret:', ret, corners)
        cv2.waitKey()
        if ret:
            objp_list.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))

            # 在原角点的基础上寻找亚像素角点，其中，criteria是设置寻找亚像素角点的参数，
            # 采用的停止准则是最大循环次数30和最大误差容限0.001

            if corners2.any():
                corners_list.append(corners2)
            else:
                corners_list.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

            cv2.imshow('img', img)
            cv2.waitKey(5000)
            print('Found corners for %s' % fname)
        else:
            print('Warning: ret = %s for %s' % (ret, fname))

    cv2.destroyAllWindows()

    # 标定
    img = cv2.imread('test_images/straight_lines1.jpg')
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp_list, corners_list, img_size, None, None)

    # print ("ret:", ret)        # ret为bool值
    # print ("mtx:\n", mtx)      # 内参数矩阵
    # print ("dist:\n", dist )   # 畸变系数 distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    # print ("rvecs:\n", rvecs)  # 旋转向量，外参数
    # print ("tvecs:\n", tvecs ) # 平移向量，外参数
    return mtx, dist


if __name__ == '__main__':
    mtx, dist = calibrate_camera()
    save_dict = {'mtx': mtx, 'dist': dist}
    with open('calibrate_camera.p', 'wb') as f:
        pickle.dump(save_dict, f)
    # pickle提供了一个简单的持久化功能。可以将对象以文件的形式存放在磁盘上。
    # pickle模块只能在python中使用，python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化，
    # pickle序列化后的数据，可读性差，人一般无法识别。
    # pickle.dump(obj, file[, protocol])序列化对象，并将结果数据流写入到文件对象中。
    # 参数protocol是序列化模式，默认值为0，表示以文本的形式序列化。protocol的值还可以是1或2，表示以二进制的形式序列化。

    # 示例校准图像
    img = cv2.imread('camera_cal/calibration5.jpg')
    cv2.imshow("原图", img)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    print(help(cv2.undistort))
    cv2.imshow("校正后", dst)
    cv2.imwrite('example_images/undistort_calibration.png', dst)
    cv2.waitKey()
