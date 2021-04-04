import cv2
import numpy as np
#读入图片
img = cv2.imread('0.jpg')
H_rows, W_cols= img.shape[:2]
print(H_rows, W_cols)

# 原图中的四个角点pts1(对应好即可，左上、右上、左下、右下),与变换后矩阵位置pts2
# pts1 = np.float32([[0, 0],[1261, 0], [1261, 946], [0, 946]])
pts1 = np.float32([[277, 149],[323, 150], [55, 466], [568,470]])
# pts2 = np.float32([[0,0],[135*4, 0],[92*4,112*4],[54*4,112*4]])
pts2 = np.float32([[855,2500],[1368, 2500],[855,112*50-100],[1368,112*50-100]])

# 生成透视变换矩阵；进行透视变换
## 说明获取逆透视变换矩阵函数各参数含义 ；src：源图像中待测矩形的四点坐标；  sdt：目标图像中矩形的四点坐标
# cv2.getPerspectiveTransform(src, dst) → retval

M = cv2.getPerspectiveTransform(np.array(pts1), np.array(pts2))
print(M)
## 说明逆透视变换函数各参数含义
# cv2.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst
# src：输入图像;   M：变换矩阵;    dsize：目标图像shape;    flags：插值方式，interpolation方法INTER_LINEAR或INTER_NEAREST;
# borderMode：边界补偿方式，BORDER_CONSTANT or BORDER_REPLICATE;   borderValue：边界补偿大小，常值，默认为0
dst = cv2.warpPerspective(img, M, (2240,112*50))
dst2 = cv2.perspectiveTransform(np.float32(pts1).reshape(-1,1,2),M)
print(dst2)
#显示图片
cv2.namedWindow('dst',0)
cv2.namedWindow('original_img',0)
cv2.imshow("original_img",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

