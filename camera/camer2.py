# -*- coding: utf-8 -*-

# @desc 使用 python 的 openCV 获取网络摄像头的数据

import cv2
import sys
import time
import numpy as np

# 读取视频流
cap = cv2.VideoCapture(1)
# 设置视频参数
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
h, w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print('w,h:', w, h)

center = w // 2
bot_h = h - 1
bot_w = 100
top_h = 100
top_w = 100
b = 1
print(cap.isOpened())

print(sys.version)

print(cv2.__version__)
top_l, top_r = [center - top_w, top_h], [center + top_w, top_h]
bot_l, bot_r = [center - bot_w, bot_h], [center + bot_w, bot_h]

while cap.isOpened():
    ret_flag, img_camera = cap.read()
    pts = np.array([[top_l, top_r, bot_r, bot_l]], np.int32)
    cv2.polylines(img_camera, pts, isClosed=True, color=(0, 0, 255), thickness=b)
    cv2.imshow("camera", img_camera)

    # 每帧数据延时 1ms, 延时为0, 读取的是静态帧
    k = cv2.waitKey(1)
    # print(k)
    if k == ord('w'):
        top_l[1] = max(0, top_l[1] - 1)
    elif k == ord('a'):
        top_l[0] = max(1, top_l[0] - 1)
    elif k == ord('s'):
        top_l[1] = min(h - 1, top_l[1] + 1)
    elif k == ord('d'):
        top_l[0] = min(w - 1, top_l[0] + 1)
    if k == ord('t'):
        top_r[1] = max(0, top_r[1] - 1)
    elif k == ord('f'):
        top_r[0] = max(1, top_r[0] - 1)
    elif k == ord('g'):
        top_r[1] = min(h - 1, top_r[1] + 1)
    elif k == ord('h'):
        top_r[0] = min(w - 1, top_r[0] + 1)
    if k == ord('i'):
        bot_l[1] = max(0, bot_l[1] - 1)
    elif k == ord('j'):
        bot_l[0] = max(1, bot_l[0] - 1)
    elif k == ord('k'):
        bot_l[1] = min(h - 1, bot_l[1] + 1)
    elif k == ord('l'):
        bot_l[0] = min(w - 1, bot_l[0] + 1)
    if k == ord('8'):
        bot_r[1] = max(0, bot_r[1] - 1)
    elif k == ord('4'):
        bot_r[0] = max(1, bot_r[0] - 1)
    elif k == ord('5'):
        bot_r[1] = min(h - 1, bot_r[1] + 1)
    elif k == ord('6'):
        bot_r[0] = min(w - 1, bot_r[0] + 1)

    elif k == ord('b'):
        b += 1
    elif k == ord(' '):
        name = str(time.time())
        cv2.imwrite(name + ".jpg", img_camera)
        with open(name + '.txt', 'w') as f:
            f.write(str(top_l) + str(top_r) + str(bot_l) + str(bot_r))
    if k == ord('q'):
        break

# 释放所有摄像头
cap.release()

# 删除窗口
cv2.destroyAllWindows()
