#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------

    @   Author  :       Max_PJB
    @   date    :       2021/2/13 16:46
    @   IDE     :       PyCharm
    @   GitHub  :       https://github.com/JackyPJB
    @   Contact :       pengjianbiao@hotmail.com
-------------------------------------------------
    Description :       
-------------------------------------------------
"""

__author__ = 'Max_Pengjb'

import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, inplanes):
        super(AttentionBlock, self).__init__()
        W_, h_ = 32, 18
        self.pool = nn.AdaptiveAvgPool2d((h_, W_))
        self.Q = nn.Parameter(torch.FloatTensor(W_ * h_, inplanes // 8))
        self.K = nn.Parameter(torch.FloatTensor(W_ * h_, inplanes // 8))
        self.V = nn.Conv2d(in_channels=inplanes, out_channels=inplanes, kernel_size=1, stride=1)
        self.soft_max = nn.Softmax(dim=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        v = self.V(x).view(b, c, h * w)  # [b,c,N]
        x = self.pool(x)
        # print('attention block input x shape: ',x.shape)
        x = x.view(b, c, -1)  # [b,c,N']
        q = x @ self.Q  # [b,c,z]
        k = x @ self.K  # [b,c,z]
        a = q @ k.permute(0, 2, 1) / (c ** 0.5)  # [b,c,c]
        score = self.soft_max(a)  # [b,c,c]
        out = score @ v  # [b,c,N]
        return out.view(b, c, h, w)

# TODO multi head
# class MultiHeadAttention(nn.Module):
#     def __init__(self, inplanes, h, w, heads=1):
#         super(MultiHeadAttention, self).__init__()
#         self.heads = heads
#         self.attentions = []
#         for _ in range(heads):
#             self.attentions.append(AttentionBlock(inplanes, h, w))
#
#     def forward(self, x):
#         outs = torch.add([attention(x) for attention in self.attentions])
#         return outs


if __name__ == '__main__':
    inputs = torch.randn((1, 3, 1080, 1920))
    print(inputs)
    at = AttentionBlock(3)
    res = at(inputs)
    print(res, res.shape)
    torch.linspace(0,6,6).int()