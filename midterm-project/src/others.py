import cv2
import numpy as np
from skimage.measure import compare_psnr

def Robert(img):
    # 第一步卷积核
    operatorFirst = np.array([[-1, 0], [0, 1]])
    # 第二步卷积核
    operatorSecond = np.array([[0, -1], [1, 0]])
    img2 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    img3 = img2.copy()
    # 两次叠加
    for i in range(1, img2.shape[0]):
        for j in range(1, img2.shape[1]):
            kernel = img2[i - 1:i + 2, j - 1:j + 2]
            img3[i, j] = np.abs(np.sum(kernel[1:, 1:] * operatorFirst)) + np.abs(np.sum(kernel[1:, 1:] * operatorSecond))
    img3 = img3 * (255 / np.max(img3))
    img3 = img3.astype(np.uint8)
    img3 = np.resize(img3,img.shape)
    cv2.imwrite("robert.jpg", img3)
    print(compare_psnr(img, img3))


def Laplacian(img, operatorType):
    # 4邻域情况的卷积核
    if operatorType == "fourfields":
        operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        img2 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        img3 = img2.copy()
        # 卷积操作
        for i in range(1, img2.shape[0] - 1):
            for j in range(1, img2.shape[1] - 1):
                kernel = img2[i - 1:i + 2, j - 1:j + 2]
                img3[i - 1, j - 1] = np.abs(np.sum(kernel * operator))
        # 归一化
        img3 = img3 * (255 / np.max(img3))
        img3 = img3.astype(np.uint8)
        img3 = np.resize(img3, img.shape)
        cv2.imwrite("laplacefour.jpg", img3)
        print(compare_psnr(img, img3))
    # 8邻域情况的卷积核
    elif operatorType == "eightfields":
        operator = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        img2 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        img3 = img2.copy()
        # 卷积操作
        for i in range(1, img2.shape[0] - 1):
            for j in range(1, img2.shape[1] - 1):
                kernel = img2[i - 1:i + 2, j - 1:j + 2]
                img3[i - 1, j - 1] = np.abs(np.sum(kernel * operator))
        # 归一化
        img3 = img3 * (255 / np.max(img3))
        img3 = img3.astype(np.uint8)
        img3 = np.resize(img3, img.shape)
        cv2.imwrite("laplaceeight.jpg", img3)
        print(compare_psnr(img, img3))
    else:
        raise ("type Error")

def Prewitt(img, operatorType):
    # 竖直方向卷积核
    if operatorType == "vertical":
        operator = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        img2 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        img3 = img2.copy()
        # 卷积操作
        for i in range(1, img2.shape[0] - 1):
            for j in range(1, img2.shape[1] - 1):
                kernel = img2[i - 1:i + 2, j - 1:j + 2]
                img3[i - 1, j - 1] = np.abs(np.sum(kernel * operator))
        # 归一化
        img3 = img3 * (255 / np.max(img3))
        img3 = img3.astype(np.uint8)
        img3 = np.resize(img3, img.shape)
        cv2.imwrite("prewittvertical.jpg", img3)
        print(compare_psnr(img, img3))
    # 水平方向卷积核
    elif operatorType == "horizonal":
        operator = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        img2 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        img3 = img2.copy()
        # 卷积操作
        for i in range(1, img2.shape[0] - 1):
            for j in range(1, img2.shape[1] - 1):
                kernel = img2[i - 1:i + 2, j - 1:j + 2]
                img3[i - 1, j - 1] = np.abs(np.sum(kernel * operator))
        # 归一化
        img3 = img3 * (255 / np.max(img3))
        img3 = img3.astype(np.uint8)
        img3 = np.resize(img3, img.shape)
        cv2.imwrite("prewitthorizonal.jpg", img3)
        print(compare_psnr(img, img3))
    else:
        raise ("type Error")

def Sobel(img, operatorType):
    # 竖直方向卷积核
    if operatorType == "vertical":
        operator = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        img2 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        img3 = img2.copy()
        # 卷积操作
        for i in range(1, img2.shape[0] - 1):
            for j in range(1, img2.shape[1] - 1):
                kernel = img2[i - 1:i + 2, j - 1:j + 2]
                img3[i - 1, j - 1] = np.abs(np.sum(kernel * operator))
        # 归一化
        img3 = img3 * (255 / np.max(img3))
        img3 = img3.astype(np.uint8)
        img3 = np.resize(img3, img.shape)
        cv2.imwrite("sobelvertical.jpg", img3)
        print(compare_psnr(img, img3))
    # 水平方向卷积核
    elif operatorType == "horizonal":
        operator = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        img2 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        img3 = img2.copy()
        # 卷积操作
        for i in range(1, img2.shape[0] - 1):
            for j in range(1, img2.shape[1] - 1):
                kernel = img2[i - 1:i + 2, j - 1:j + 2]
                img3[i - 1, j - 1] = np.abs(np.sum(kernel * operator))
        # 归一化
        img3 = img3 * (255 / np.max(img3))
        img3 = img3.astype(np.uint8)
        img3 = np.resize(img3, img.shape)
        cv2.imwrite("sobelhorizonal.jpg", img3)
        print(compare_psnr(img, img3))
    else:
        raise ("type Error")

def LoG(img):
    # 先通过高斯滤波降噪
    gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    # 再通过拉普拉斯算子做边缘检测
    laplace = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)
    LOG = cv2.convertScaleAbs(laplace)
    LOG = np.resize(LOG, img.shape)
    cv2.imwrite("LOG.jpg",LOG)
    print(compare_psnr(img, LOG))

if __name__ == "__main__":
    img = cv2.imread("background.jpg")
    grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    Robert(grayimg)
    # Prewitt(grayimg, "horizonal")
    # Prewitt(grayimg, "vertical")
    # Sobel(grayimg, "horizonal")
    # Sobel(grayimg, "vertical")
    # Laplacian(grayimg, "fourfields")
    # Laplacian(grayimg, "eightfields")
    # LoG(grayimg)
