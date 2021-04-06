import random

import cv2
import numpy as np
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
import matplotlib.pyplot as plt

xs = [
    [6, 11, 14, 22, 29, 38, 44, 46, 53, 62, 70, 76, 86, 94, 102, 109, 114, 117, 125, 134, 140, 149, 157, 165, 173, 179,
     182, 189, 197, 204],
    [482, 476, 473, 466, 460, 457, 450, 444, 441, 434, 428, 425, 418, 412, 409, 402, 397, 393, 386, 381, 377, 369, 364,
     361, 353, 349, 345, 337, 333, 329, 321, 317, 313, 305, 299, 297, 289, 283, 273, 266, 258, 251]]
ys = [
    [186, 185, 182, 181, 179, 177, 176, 175, 173, 171, 169, 168, 166, 164, 162, 160, 160, 158, 156, 154, 152, 150, 149,
     147, 145, 144, 142, 141, 139, 138],
    [250, 249, 244, 242, 241, 236, 234, 233, 229, 226, 224, 221, 218, 217, 213, 210, 208, 205, 202, 200, 197, 194, 192,
     189, 186, 184, 182, 178, 176, 174, 170, 168, 166, 163, 160, 158, 155, 153, 148, 145, 141, 138]]

# IPM
ipm_img_W = 480
ipm_img_H = 640
pts1 = np.float32([[241.0, 279], [316.0, 283], [112.0, 479.0], [528.0, 479.0]])  # 2
# pts2 = np.float32([[0,0],[135*4, 0],[92*4,112*4],[54*4,112*4]])
# pts2 = np.float32([[855,2500],[1368, 2500],[855,112*50-100],[1368,112*50-100]])
pts2 = np.float32(
    [[ipm_img_W // 2 - 60, 0], [ipm_img_W // 2 + 60, 0],
     [ipm_img_W // 2 - 60, ipm_img_H], [ipm_img_W // 2 + 60, ipm_img_H]])
M = cv2.getPerspectiveTransform(np.array(pts1), np.array(pts2))

img_path = 'tt.jpg'
img = cv2.imread(img_path)
cv2.imshow('origin', img)
h, w = img.shape[:2]
lss = []
for line_x, line_y in zip(xs, ys):
    line = list(zip(line_x, line_y))
    lss.append(LineString(line))

# 网络结果是 256x512 的图上的，需要进行转换
image = np.zeros((256, 512))
lsoi = LineStringsOnImage(lss, shape=image.shape)
transformations = iaa.Sequential([Resize({'height': h, 'width': w})])
image, line_strings = transformations(image=image, line_strings=lsoi)
# line_strings.clip_out_of_image_()

poly_img = np.copy(img)
# 透视变换
dst = cv2.warpPerspective(poly_img, M, (ipm_img_W, ipm_img_H))
# cv2.imshow('ipm', dst)

lanes = []
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

for i, ls in enumerate(line_strings):
    kps = ls.to_keypoints()
    kpsoi = KeypointsOnImage(kps, img)
    kpsoi.draw_on_image(img, copy=False, color=colors[i], size=5)
    xys = kpsoi.to_xy_array()
    lanes.append(xys)
# cv2.imshow('point', img)

# cv2.polylines(img, lanes, isClosed=False, color=(0, 0, 255), thickness=3)
ptss = []  # 透视之后的关键点
for i, line in enumerate(lanes):
    # print('line', line)
    pst2 = cv2.perspectiveTransform(line.reshape(1, -1, 2), M)
    print(pst2)
    ptss.append(pst2[0])
    kpsoi = KeypointsOnImage.from_xy_array(pst2[0], dst.shape[:2])
    kpsoi.draw_on_image(dst, copy=False, color=colors[i], size=4)
# cv2.imshow('ipm_point', dst)

# 拟合
for i, xys in enumerate(ptss):
    # print(xys)
    p = np.polyfit(xys[:, 1], xys[:, 0], 2)
    a, b, c = p.tolist()
    print("求解的曲线是:")
    print("x = " + str(a) + "y**2 +" + str(b) + "y +" + str(c))
    # 画拟合直线
    y = np.linspace(0, ipm_img_H, ipm_img_H)  ##在0-15直接画100个连续点
    x = a * y * y + b * y + c  ##函数式
    pppsss = list(zip(x, y))
    cv2.polylines(dst, np.int32([pppsss]), isClosed=False, color=colors[i], thickness=3)
# cv2.imshow('ipm_poly', dst)

'''
勾画行车轨迹，只和拐弯的角度 theta 有关
'''
ta = random.randint(-10, 10)
while ta == 0:
    ta = random.randint(-10, 10)
theta = ta / 360 * np.pi
L = 85
H = 60
d = 10
zx = L / np.tan(theta) + ipm_img_W // 2
zy = L + d + ipm_img_H
R_l = np.sqrt(np.square(L) + np.square(L / np.tan(theta) + H / 2))
R_r = np.sqrt(np.square(L) + np.square(L / np.tan(theta) - H / 2))
y = np.linspace(0, ipm_img_H, ipm_img_H)  ##在0-15直接画100个连续点
x_l = np.sqrt(np.square(R_l) - np.square(y - zy)) + zx
x_l2 = -np.sqrt(np.square(R_l) - np.square(y - zy)) + zx
x_r = np.sqrt(np.square(R_r) - np.square(y - zy)) + zx
x_r2 = -np.sqrt(np.square(R_r) - np.square(y - zy)) + zx
yy = np.hstack([y, y])
# 处理左边
x_ls = np.hstack([x_l, x_l2])
xy_l = np.stack([x_ls, yy], axis=1)
xy_l = np.where(np.isnan(xy_l), -1, xy_l)
kps_l = KeypointsOnImage.from_xy_array(xy_l, dst.shape[:2])
kps_l.clip_out_of_image()
kps_l.draw_on_image(dst, copy=False, color=colors[2], size=1)
# 处理右边
x_rs = np.hstack([x_r, x_r2])
xy_r = np.stack([x_rs, yy], axis=1)
xy_r = np.where(np.isnan(xy_r), -1, xy_r)
kps_r = KeypointsOnImage.from_xy_array(xy_r, dst.shape[:2])
kps_r.clip_out_of_image()
kps_r.draw_on_image(dst, copy=False, color=colors[2], size=1)
speed = random.randint(5, 15)
cv2.putText(dst, 'speed: ' + str(speed / 10) + 'm/s', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
cv2.putText(dst, 'steer roll: ' + str(ta), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

cv2.imshow('ipm_trace', dst)

cv2.waitKey()
cv2.destroyAllWindows()