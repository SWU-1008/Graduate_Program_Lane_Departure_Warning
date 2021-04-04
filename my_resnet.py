#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------

    @   Author  :       Max_PJB
    @   date    :       2021/2/10 17:41
    @   IDE     :       PyCharm
    @   GitHub  :       https://github.com/JackyPJB
    @   Contact :       pengjianbiao@hotmail.com
-------------------------------------------------
    Description :       
-------------------------------------------------
"""
import os
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader

__author__ = 'Max_Pengjb'

from torch.hub import load_state_dict_from_url
import torch
import torch.nn as nn
from torchvision.models import resnet
from attention_block import AttentionBlock
import numpy as np
import cv2

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.attention = AttentionBlock(planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


class MyDataset(Dataset):

    def __init__(self, txt_path, base_dir, transform=None):
        SPLIT_FILES = {
            '00': "my_list/train_normal.txt",
            '01': 'my_list/train_night.txt',
            '02': "my_list/train_shadow.txt",
            '10': 'my_list/train_hlight.txt',
            '11': 'my_list/train_hlight_night.txt',
            '12': 'my_list/train_hlight_shadow.txt'
        }
        split_txts = {'00': [], '01': [], '02': [], '10': [], '11': [], '12': []}
        classes_dict = {'00': 0, '01': 1, '02': 2, '10': 3, '11': 4, '12': 5}
        self.transfrom = transform
        self.data = []
        print(txt_path)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # print(line)
            line = line.strip('\n')
            img, a, b = line.split()
            img_path = os.path.join(base_dir, os.path.join(*img.split('/')))
            self.data.append((img_path, classes_dict[a + b]))
            split_txts[a + b].append(img_path + ' ' + str(classes_dict[a + b]))
        # 把每个类别的图片分别存放在不同的 txt 后缀的 label 文档
        for class_str, txt_path in SPLIT_FILES.items():
            split_txt_path = os.path.join(base_dir, os.path.join(*txt_path.split('/')))
            if os.path.exists(split_txt_path):
                continue
            else:
                with open(split_txt_path,'w') as spt:
                    spt.write('\n'.join(split_txts[class_str]))

    def __getitem__(self, item):
        img_path, label = self.data[item]
        # 如果是 png 图片，PIL 读取的会是四通道的，所以需要 转成 RGB
        # img = torch.Tensor(np.array(Image.open(img_path).convert("RGB"))/255.0).permute(2,0,1)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (288, 512))  # 图片太大，压缩一下
        img = torch.Tensor(img / 255.0).permute(2, 0, 1)  # opencv 默认是读取三通道的[BGR]格式
        # opencv 读取的图像为BGR,转为RGB
        # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        # print('img shape', img.shape)
        if self.transfrom:
            img = self.transfrom(img)
        return img, label
        # return img_path, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from tqdm import tqdm

    isMy = True  # 训练一个加了 attention 的 resnet 和一个原始的，进行对比
    # isMy = False
    last_epoch = 349
    if isMy:
        model_name = 'my_resnet_50'
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        my_resnet_50 = resnet50(num_classes=6)  # 加了attention 的
    else:
        model_name = 'origin_resnet_50'
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        my_resnet_50 = resnet.resnet50(num_classes=6)  # 原始的 resnet
    pre_trained = './model_dicts/' + model_name + '_' + str(last_epoch) + '.pth'
    my_resnet_50.load_state_dict(torch.load(pre_trained))
    my_resnet_50.to(device)  # 加载进 gpu 如果有的话
    # 查看一下参数量
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in my_resnet_50.parameters())))
    # 打印模型名称与shape
    for name, parameters in my_resnet_50.named_parameters():
        print(name, ':', parameters.size())

    # LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(pathname)s %(message)s "  # 配置输出日志格式
    LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s "  # 配置输出日志格式
    DATE_FORMAT = '%Y-%m-%d  %H:%M:%S %a '  # 配置输出时间的格式，注意月份和天数不要搞乱了
    logging.basicConfig(level=logging.INFO,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT,
                        # 不同的模型保存到不同的 log 文件
                        filename=model_name + ".log")  # 有了filename参数就不会直接输出显示到控制台，而是直接写入文件

    # 数据集
    base_dir = '/home/lab1008/Desktop/datasets/CULane'  # CULane 数据集位置
    train_val_txt = '/home/lab1008/Desktop/datasets/CULane/xx.txt'  # 训练集和验证集的标签文件
    # 加载数据
    train_val_set = MyDataset(train_val_txt, base_dir)
    test_txt = 'xx2.txt'  # todo 测试集的标签文件

    # 8:2 划分[训练集：验证集]
    train_size = int(0.8 * len(train_val_set))
    val_size = len(train_val_set) - train_size
    print('train_size,test_size', train_size, val_size)
    train_set, val_set = torch.utils.data.random_split(train_val_set, [train_size, val_size])
    # batch 大小
    batch_size = 8

    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    print('len(train_data),len(test_data)', len(train_data), len(val_data))
    # 测试集
    test_set = MyDataset(test_txt, base_dir)
    test_data = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # 加载进 GPU（if E）
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(my_resnet_50.parameters(), lr=5e-4)


    def train(model, data_loader, criteon, optimizer, model_name, isTrain=False, epoch_idx=None):
        if isTrain:
            model.train()
        else:
            model.eval()
        train_loss = 0
        num_correct = 0
        for img_batch, y_batch in tqdm(data_loader):
            img_batch, y_batch = img_batch.to(device), y_batch.to(device)
            # print('img_batch.shape, y_batch.shape', img_batch.shape, y_batch.shape)
            logits = model(img_batch)
            # logits = my_resnet_50(x)
            # print('logits.shape', logits, logits.shape)
            loss = criteon(logits, y_batch)
            # back prop
            if isTrain:
                # 做 val 的时候不需要做 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  # 梯度清零
            train_loss += float(loss.item())
            pred = logits.argmax(dim=1)
            num_correct += torch.eq(pred, y_batch).sum().float().item()
        acc = num_correct / len(data_loader.dataset)
        loss = train_loss / len(data_loader)
        if isTrain:
            assert epoch_idx is not None
            torch.save(model.state_dict(), './model_dicts/' + model_name + '_' + str(epoch_idx) + '.pth')
        return loss, acc


    # train
    epochs = 0
    start_epoch = last_epoch + 1
    for epoch in range(start_epoch, start_epoch + epochs):
        loss, acc = train(my_resnet_50, train_data, criteon, optimizer, model_name, True, epoch)
        logging.info("Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch, loss, acc))
        print("Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch, loss, acc))
        #  每 5 个 epoch 做一次 val
        if epoch % 2 == 0:
            # val 必须调用model.eval()，以便在运行推断之前将dropout和batch规范化层设置为评估模式。如果不这样做，将会产生不一致的推断结果。
            loss, acc = train(my_resnet_50, val_data, criteon, None, model_name, False)
            logging.info("Val Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch, loss, acc))
            print("Val Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch, loss, acc))

    # test
    # loss, acc = train(my_resnet_50, val_data, criteon, None, model_name, False)
    # logging.info("Test Loss: {:.6f}\t Acc: {:.6f}".format(loss, acc))

    # 混淆矩阵
    model_name = 'origin_resnet_50'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    or_resnet_50 = resnet.resnet50(num_classes=6)  # 原始的 resnet
    pre_trained = './model_dicts/' + model_name + '_' + str(199) + '.pth'
    or_resnet_50.load_state_dict(torch.load(pre_trained))
    or_resnet_50.to(device)  # 加载进 gpu 如果有的话
    my_cm = torch.zeros(6,6).to(device)
    or_cm = torch.zeros(6,6).to(device)
    my_resnet_50.eval()
    or_resnet_50.eval()
    with torch.no_grad():
        for img_batch, y_batch in tqdm(val_data):
            img_batch = img_batch.to(device)
            y_batch = y_batch.to(device)
            my_logits = my_resnet_50(img_batch)
            or_logits = or_resnet_50(img_batch)
            my_pres = torch.argmax(my_logits,dim=1)
            or_pres = torch.argmax(or_logits,dim=1)
            for i, j in zip(y_batch, my_pres):
                my_cm[i, j] += 1
            for i, j in zip(y_batch, or_pres):
                or_cm[i, j] += 1
    print(my_cm)
    print(or_cm)
