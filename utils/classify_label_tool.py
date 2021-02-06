#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------

    @   Author  :       Max_PJB
    @   date    :       2021/2/6 17:23
    @   IDE     :       PyCharm
    @   GitHub  :       https://github.com/JackyPJB
    @   Contact :       pengjianbiao@hotmail.com
-------------------------------------------------
    Description :       
-------------------------------------------------
"""

__author__ = 'Max_Pengjb'

import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from pprint import pprint


# Python 读取一个目录下的所有文件
# recursion 设置为 false 就不循环读取子目录，默认为 True
def get_all_files(path, recursion=True, level=0):
    # 用一个数组保存所有的文件，作为返回
    all_files = []
    # 所有文件夹，第一个字段是次目录的级别
    dirList = []
    # 所有文件
    fileList = []
    # 返回一个列表，其中包含在目录条目的名称(google翻译)
    files = os.listdir(path)

    for f in files:
        if os.path.isdir(path + '/' + f):
            # 排除隐藏文件夹。因为隐藏文件夹过多
            if f[0] == '.':
                pass
            else:
                # 添加非隐藏文件夹
                dirList.append(f)
        if os.path.isfile(path + '/' + f):
            # 添加文件
            fileList.append(f)
            # 当一个标志使用，文件夹列表第一个级别不打印
    if recursion:
        # 如果需要递归，就循环递归
        for dl in dirList:
            # 打印至控制台，不是第一个的目录
            print('    ' * level + '-', dl)
            # 打印目录下的所有文件夹和文件，目录级别+1
            all_files.extend(get_all_files(path + '/' + dl, recursion, level + 1))
    for fl in fileList:
        # 打印文件
        print('    ' * level + '-', fl)
        # 加入文件
        all_files.append(path + '/' + fl)
    return all_files


# opencv往图片中写入中文,返回图片
def DrawChinese(img, text, positive, fontSize=50, fontColor=(255, 0, 0)):
    # args-(img:numpy.ndarray, text:中文文本, positive:位置, fontSize:字体大小默认20, fontColor:字体颜色默认绿色)
    cv2img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("simhei.ttf", fontSize, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text(positive, text, fontColor, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体格式
    cv2charimg = cv.cvtColor(np.array(pilimg), cv.COLOR_RGB2BGR)  # PIL图片转cv2 图片
    return cv2charimg


if __name__ == '__main__':
    base_dir = r'C:\Users\pengj\Desktop\毕业相关\Graduate_Program_Lane_Departure_Warning\utils'
    labels_dir = os.path.join(base_dir, 'list')
    save_txt = 'xx.txt'
    label_files = get_all_files(labels_dir, recursion=False)
    end_line_path = None
    if os.path.exists(save_txt):
        with open(save_txt, 'r') as xx:
            end_line_path = xx.readlines()[-1].strip('\n').split()[0]
    print('end_line_path: ', end_line_path)
    flag = False
    for file in label_files:
        if 'gt' in file:
            with open(file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                ori_img_path = line.split()[0]
                print('ori_img_path: ', ori_img_path)

                if end_line_path is not None and not flag:
                    if ori_img_path == end_line_path:
                        flag = True
                    continue
                img_path = os.path.join(base_dir, *ori_img_path.split('/'))
                print(img_path)
                # img = cv.imread('haha.jpg')
                img = cv.imread(img_path)
                img1 = DrawChinese(img, '正常：0 ，逆光：1', (50, 50))
                cv.imshow('haha', img1)
                k1 = cv.waitKey(0)
                k1 -= 48
                while k1 not in [0, 1]:
                    cv.imshow('haha', img1)
                    k1 = cv.waitKey(0)
                    k1 -= 48
                img2 = DrawChinese(img, '白天：0，夜晚：1，阴影：2', (50, 50))
                cv.imshow('haha', img2)
                k2 = cv.waitKey(0)
                k2 -= 48
                while k2 not in [0, 1, 2]:
                    cv.imshow('haha', img2)
                    k2 = cv.waitKey(0)
                    k2 -= 48

                with open('xx.txt', 'a') as ff:
                    ff.write(ori_img_path + ' ' + str(k1) + ' ' + str(k2) + '\n')
