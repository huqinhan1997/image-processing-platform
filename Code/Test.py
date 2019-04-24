import numpy as np
import cv2 as cv
import random
# img = cv.imread('/Users/huqinhan/Desktop/数字图像/数字媒体技术/project1/images/photo1.jpg', 3)
# out_img = np.ones(img.shape, dtype=np.uint8)
# a = np.ones(img.shape, dtype=np.uint8)
#
# print(cv.add(out_img,a).shape)

# a = 2
# b = 10
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         out_img[i][j] = a*img[i][j]+b
# print(out_img)
# print("-------------------")
# print(img*a+b)

# A = np.ones((3,3,3),dtype=np.float16)
# # A[A>0]=2

# A= np.ones(img.shape,dtype=np.uint8)*101
# new = A+img
# print(img[0][0])
# print(new[0][0])

import cv2
import imutils
import numpy as np

# def c_and_b(arg):
#     ''''''
#     cnum = cv2.getTrackbarPos(trackbar_name1, wname)
#     bnum = cv2.getTrackbarPos(trackbar_name2, wname)
#     #print(bnum)
#     cimg = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#     # for i in range(img.shape[0]):
#     #     for j in range(img.shape[1]):
#     #         lst = 0.1*cnum*img[i, j] + bnum
#     #         cimg[i, j] = [int(ele) if ele < 255 else 255 for ele in lst]
#     #print(cimg[0][0])
#     new = np.array(img)
#     cimg = new*0.1*cnum+bnum
#     cimg[cimg > 255] = 255
#     cimg = cimg.astype(np.uint8)
#     cv2.imshow(wname, imutils.resize(cimg, 800))
#
# wname = 'brightness and contrast'
# trackbar_name1 = 'contrast'
# trackbar_name2 = 'brightness'
# img = cv.imread('/Users/huqinhan/Desktop/数字图像/数字媒体技术/project1/images/photo1.jpg', 3)
# height, width = img.shape[:2]
#
# img = cv2.resize(img, (int(width/height*400), 400), interpolation=cv2.INTER_CUBIC)
#
# cv2.namedWindow(wname)
# cv2.createTrackbar(trackbar_name1, wname, 10, 20, c_and_b)
# cv2.createTrackbar(trackbar_name2, wname, 0, 100, c_and_b)
# c_and_b(0)
#
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()
# sizes = [2,3,2]
# a = [np.random.rand(y,1) for y in sizes[1:]]
# print(a[0])

a = np.array(([1,2],[2,3],[4,5]))

print(a.transpose())

b=[0]
print(b[-1])
