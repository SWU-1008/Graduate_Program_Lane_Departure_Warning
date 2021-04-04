#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------

    @   Author  :       pengjb
    @   date    :       2021/3/13
    @   IDE     :       PyCharm
    @   GitHub  :       https://github.com/JackyPJB
    @   Contact :       pengjianbiao@hotmail.com
-------------------------------------------------
    Description :       
-------------------------------------------------
"""
import os


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


base_dir = '/home/lab1008/Desktop/datasets/CULane'

val_path = os.path.join(base_dir, 'list/val.txt')
train_path = os.path.join(base_dir, 'list/train.txt')

my_save_dir = os.path.join(base_dir, 'my_list/lane_task')

my_dir = os.path.join(base_dir, 'my_list')
my_files = get_all_files(my_dir, recursion=False)
val_imgs = set()
train_imgs = set()
with open(train_path, 'r') as tp:
    lines = tp.readlines()
for line in lines:
    line = line.strip('\n')
    train_imgs.add(line)

with open(val_path, 'r') as tp:
    lines = tp.readlines()
for line in lines:
    line = line.strip('\n')
    val_imgs.add(line)

for file_path in my_files:
    file_name = os.path.basename(file_path)
    img_type = file_name.split('_')
    # img_type = '_'.join(file_name.split('_')[1:])
    print('file_path,file_name,img_type', file_path, file_name, img_type)
    with open(file_path, 'r') as tp:
        lines = tp.readlines()
    for line in lines:
        # print(line)
        line = line.strip('\n')
        img_path, a = line.split()
        img_paths = img_path.split('/')
        img_path = '/' + os.path.join(*img_paths[-3:])
        # print('img_path', img_path)
        if img_path in val_imgs:
            save_file_name = '_'.join(['val'] + img_type[1:])
            save_path = os.path.join(my_save_dir, save_file_name)
            with open(save_path, 'a') as wp:
                wp.write(img_path + '\n')
        if img_path in train_imgs:
            # print('train')
            save_file_name = '_'.join(['train'] + img_type[1:])
            save_path = os.path.join(my_save_dir, save_file_name)
            with open(save_path, 'a') as wp:
                wp.write(img_path + '\n')

res = get_all_files(my_save_dir)
for r in res:
    print(r)