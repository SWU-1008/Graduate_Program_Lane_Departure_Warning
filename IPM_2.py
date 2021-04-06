import time

import cv2
import numpy as np

# 读入图片
# img = cv2.imread('/home/swucar/Desktop/workspace/Graduate_Program_Lane_Departure_Warning/camera/1617530602.5335813.jpg')
# img = cv2.imread('/home/swucar/Desktop/workspace/Graduate_Program_Lane_Departure_Warning/camera/1617530255.782367.jpg')
img = cv2.imread('/home/swucar/Desktop/workspace/Graduate_Program_Lane_Departure_Warning/camera/1617529662.792434.jpg')
# img = cv2.imread('/home/swucar/Desktop/workspace/Graduate_Program_Lane_Departure_Warning/tt.jpg')

H_rows, W_cols = img.shape[:2]
print(H_rows, W_cols)
w = 480
h_top = 10
h = 680
# h_bot = 180
h_bot = 10
# 原图中的四个角点pts1(对应好即可，左上、右上、左下、右下),与变换后矩阵位置pts2
# pts1 = np.float32([[0, 0],[1261, 0], [1261, 946], [0, 946]])
# pts1 = np.float32([[298.0, 207], [338.0, 210], [120.0, 431.0], [489.0, 456.0]]) # 1
pts1 = np.float32([[241.0, 229],[316.0, 233],[112.0, 479.0],[528.0, 479.0]]) # 2

# pts2 = np.float32([[0,0],[135*4, 0],[92*4,112*4],[54*4,112*4]])
# pts2 = np.float32([[855,2500],[1368, 2500],[855,112*50-100],[1368,112*50-100]])
pts2 = np.float32(
    [[w // 2 - 60, h_top], [w // 2 + 60, h_top], [w // 2 - 60, h_top + h], [w // 2 + 60, h_top + h]])

# 生成透视变换矩阵；进行透视变换
## 说明获取逆透视变换矩阵函数各参数含义 ；src：源图像中待测矩形的四点坐标；  sdt：目标图像中矩形的四点坐标
# cv2.getPerspectiveTransform(src, dst) → retval

M = cv2.getPerspectiveTransform(np.array(pts1), np.array(pts2))
print(M)
## 说明逆透视变换函数各参数含义
# cv2.warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst
# src：输入图像;   M：变换矩阵;    dsize：目标图像shape;    flags：插值方式，interpolation方法INTER_LINEAR或INTER_NEAREST;
# borderMode：边界补偿方式，BORDER_CONSTANT or BORDER_REPLICATE;   borderValue：边界补偿大小，常值，默认为0
dst = cv2.warpPerspective(img, M, (w, h_top + h + h_bot))
dst2 = cv2.perspectiveTransform(np.float32(pts1).reshape(-1, 1, 2), M)
print(dst2)
# 显示图片
cv2.namedWindow('dst', 0)
cv2.namedWindow('original_img', 0)
cv2.imshow("original_img", img)
cv2.imshow("dst", dst)

'''
对视频进行逆透视
'''
# video = '/home/swucar/Desktop/workspace/PINet_new-master/CULane/haha.avi'
# cap = cv2.VideoCapture(video)
# while cap.isOpened():
#     prevTime = time.time()
#     ret, frame = cap.read()
#     H_rows, W_cols = frame.shape[:2]
#     print(H_rows, W_cols)
#     # frame = cv2.resize(frame, (512, 256)) / 255.0
#     # frame = np.rollaxis(frame, axis=2, start=0)
#     curTime = time.time()
#     sec = curTime - prevTime
#     fps = 1 / (sec)
#     s = "FPS : " + str(fps)
#     # cv2.putText(frame, s, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
#     M = cv2.getPerspectiveTransform(np.array(pts1), np.array(pts2))
#     ipm_dst = cv2.warpPerspective(frame, M, (w, h_top + h + h_bot))
#     cv2.imshow('frame', frame)
#     cv2.imshow('ipm_dst', ipm_dst)
#     ch = cv2.waitKey()
#     if ch == ord('q'):
#         break
#     elif ch == ord(' '):
#         continue
#     elif ch == ord('s'):
#         cv2.imwrite(str(curTime) + '.jpg', frame)
#         cv2.imwrite(str(curTime) + '_ipm_dst.jpg', ipm_dst)
# cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
