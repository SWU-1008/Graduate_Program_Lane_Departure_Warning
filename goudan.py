#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------

    @   Author  :       Max_PJB
    @   date    :       2021/2/7 21:40
    @   IDE     :       PyCharm
    @   GitHub  :       https://github.com/JackyPJB
    @   Contact :       pengjianbiao@hotmail.com
-------------------------------------------------
    Description :       
-------------------------------------------------
"""

__author__ = 'Max_Pengjb'

from torch import nn
import torch
import numpy


class WeightAttention(nn.Module):
    def __init__(self, bit, ms_num=4, use_gpu=False):
        super(WeightAttention, self).__init__()
        # self.weight = torch.full([ms_num, bit], 1 / ms_num)
        # self.weight = torch.full([ms_num, bit], 1 / bit)
        self.weight = torch.empty([ms_num, bit])
        nn.init.normal_(self.weight, 0.25, 1 / bit)
        print('weight normal mean: %2.2f std: %2.2f' % (0.25, 1 / bit))
        # self.weight = torch.randn(ms_num, bit)
        if use_gpu:
            self.weight = self.weight.cuda()
        self.weight = torch.nn.Parameter(self.weight)

    def forward(self, *input):
        hash_list = []
        for x in input:
            hash_list.append(x.unsqueeze(1))
        out = torch.cat(hash_list, dim=1)  # 4 64
        print('1111', out.shape)
        out = out * self.weight  # 4 64
        print('22222', self.weight.shape)
        print('33333', out.shape)
        out = torch.sum(out, dim=1)
        # out = self.fc(out)
        out = out.squeeze()
        print(out.shape)
        return out


if __name__ == '__main__':
    net = WeightAttention(64)
    # net = net.cuda()
    x1 = torch.ones(64, 64)
    x2 = torch.ones(64, 64)
    x3 = torch.ones(64, 64)
    x4 = torch.ones(64, 64)
    # x1 = x1.cuda()
    # x2 = x2.cuda()
    # x3 = x3.cuda()
    # x4 = x4.cuda()
    out = net(x1, x2, x3, x4)
    print(out.shape)
    print(net.weight)
    print(torch.mean(net.weight, dim=1))
    v1 = torch.randn((10,256, 56, 56))
    global_avgpool = nn.AdaptiveAvgPool2d(1)
    v1 = global_avgpool(v1)
    # print('v1v1', v1.shape)
    v1 = v1.view(v1.size(0), -1)
    print(v1.shape)
    cc = nn.Linear(256, 64)
    print(v1.shape)
    y1 = cc(v1)
    # v1 = torch.squeeze(v1)
    print('v1v1', y1.shape)
    # torch.reshape()
