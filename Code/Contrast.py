import cv2 as cv
import imutils
import numpy as np
import time

def Contrast(arg):
    #速度太慢了
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         grayscale = a*img[i][j]+b
    #         out_img[i][j] = [e if e<255 else 255 for e in grayscale]

    #A影响对比度，b影响亮度
    A = cv.getTrackbarPos(trackbar_name1,winname)
    b = cv.getTrackbarPos(trackbar_name2,winname)

    #将数字图像矩阵先声明为np.array,避免元素自动对255取模。在这里被坑了很久
    img_array = np.array(img)

    #核心公式：A使图像像素成倍数的增长或降低（A<1）改变了像素间的差值，所以对比度变化。
    #b改变像素的大小，在（0黑 255白）之间变化，所以改变的是图像的亮度。乘0.1是为了变化幅度不会太大
    out_img = (A*0.1)*img_array+b
    #用列表表达式来限制像素值最大=255
    out_img[out_img>255] = 255
    #格式化
    out_img = out_img.astype(np.uint8)
    cv.imshow(winname, imutils.resize(out_img, 800))


if __name__ =="__main__":
    start = time.clock()
    img = cv.imread('/Users/huqinhan/Desktop/数字图像/数字媒体技术/project1/images/photo1.jpg', 3)
    height, width = img.shape[:2]
    img = cv.resize(img, (int(width / height * 400), 400), interpolation=cv.INTER_CUBIC)

    #界面
    winname = 'contrast and brightness'
    trackbar_name1 = 'contrast'
    trackbar_name2 = 'brightness'
    cv.namedWindow(winname)
    cv.createTrackbar(trackbar_name1, winname, 10, 20, Contrast)
    cv.createTrackbar(trackbar_name2, winname, 0, 100, Contrast)
    Contrast(0)

    if cv.waitKey(0) == 27:
        cv.destroyAllWindows()
        print("Time used:", time.clock() - start)


