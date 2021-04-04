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
__author__ = 'Max_Pengjb'


import os
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.hub import load_state_dict_from_url
import torch
import torch.nn as nn
from torchvision.models import resnet
from attention_block import AttentionBlock
import numpy as np
import cv2
from tqdm import tqdm


class PreDataset(Dataset):

    def __init__(self, txt_path, base_dir):
        self.data = []
        self.origin = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # print(line)
            line = line.strip('\n')
            img_path = os.path.join(base_dir, os.path.join(*line.split('/')))
            self.data.append(img_path)
            self.origin.append(line)

    def __getitem__(self, item):
        img_path = self.data[item]
        # 如果是 png 图片，PIL 读取的会是四通道的，所以需要 转成 RGB
        # img = torch.Tensor(np.array(Image.open(img_path).convert("RGB"))/255.0).permute(2,0,1)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (288, 512))  # 图片太大，压缩一下
        img = torch.Tensor(img / 255.0).permute(2, 0, 1)  # opencv 默认是读取三通道的[BGR]格式
        # opencv 读取的图像为BGR,转为RGB
        # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        # print('img shape', img.shape)
        return img, self.origin[item]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    from my_resnet import resnet50
    from torchvision.models import resnet

    SPLIT_FILES = {
        0: "my_list/lane_task/test_normal.txt",
        1: 'my_list/lane_task/test_night.txt',
        2: "my_list/lane_task/test_shadow.txt",
        3: 'my_list/lane_task/test_hlight.txt',
        4: 'my_list/lane_task/test_hlight_night.txt',
        5: 'my_list/lane_task/test_hlight_shadow.txt'
    }
    split_txts = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    view = False
    # my_resnet_50 = resnet50(num_classes=6)  # 加了attention 的
    my_resnet_50 = resnet.resnet50(num_classes=6)  # 加了attention 的
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    last_epoch = 199
    # model_name = 'my_resnet_50'
    model_name = 'origin_resnet_50'
    pre_trained = './model_dicts/' + model_name + '_' + str(last_epoch) + '.pth'
    my_resnet_50.load_state_dict(torch.load(pre_trained))
    my_resnet_50.to(device)  # 加载进 gpu 如果有的话
    # 数据集
    base_dir = '/home/lab1008/Desktop/datasets/CULane'  # CULane 数据集位置
    predict_txt = '/home/lab1008/Desktop/datasets/CULane/list/test.txt'  # 预测文件地址
    # 加载数据
    predict_set = PreDataset(predict_txt, base_dir)
    # batch 大小
    batch_size = 16
    predict_loader = DataLoader(predict_set, batch_size=batch_size, shuffle=True, num_workers=4)
    my_resnet_50.eval()
    with torch.no_grad():
        for img_batch,orgin_paths in tqdm(predict_loader):
            # print(img_batch,orgin_paths)
            img_input = img_batch.to(device)
            logits = my_resnet_50(img_input)
            res = torch.argmax(logits,dim=1)
            imgs = img_batch.permute(0, 2, 3, 1).numpy()  # opencv 默认是读取三通道的[BGR]格式
            labels = res.clone().detach().cpu().numpy()
            for img,label,origin_path in zip(imgs,labels,orgin_paths):
                if view:
                    cv2.imshow(str(label),img)
                    k = cv2.waitKey(0)
                split_txts[label].append(origin_path)
    # 把每个类别的图片分别存放在不同的 txt 后缀的 label 文档
    for label, save_path in SPLIT_FILES.items():
        abs_save_path = os.path.join(base_dir, os.path.join(*save_path.split('/')))
        if os.path.exists(abs_save_path):
            continue
        else:
            with open(abs_save_path, 'w') as spt:
                spt.write('\n'.join(split_txts[label]))
