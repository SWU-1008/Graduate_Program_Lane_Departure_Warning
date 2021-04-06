import cv2
import numpy as np
from imgaug.augmentables.lines import LineString, LineStringsOnImage
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize

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
img_path = 'tt.jpg'
img = cv2.imread(img_path)
h, w = img.shape[:2]
lss = []
for line_x, line_y in zip(xs, ys):
    line = list(zip(line_x, line_y))
    lss.append(LineString(line))
image = np.zeros((256, 512))
lsoi = LineStringsOnImage(lss, shape=image.shape)
transformations = iaa.Sequential([Resize({'height': h, 'width': w})])
image, line_strings = transformations(image=image, line_strings=lsoi)
line_strings.clip_out_of_image_()
lanes = []
for ls in line_strings:
    lanes.append(np.int32(ls.coords))

cv2.polylines(img, lanes, isClosed=False, color=(0, 0, 255), thickness=3)
cv2.imshow('tt', img)
cv2.waitKey()
cv2.destroyAllWindows()
