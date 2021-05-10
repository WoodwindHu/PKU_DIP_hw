import cv2
import numpy as np


def MyErosion(img, kernel):
    height, width = img.shape
    kernel_sum = np.sum(kernel)
    # convolution
    result = np.zeros((height, width))
    for i in range(height-2):
        for j in range(width-2):
            result[i+1,j+1] = (np.sum(img[i:i+3,j:j+3]*kernel[::,::]) == (kernel_sum * 255))
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)

def MyDilation(img, kernel):
    height, width = img.shape
    kernel_sum = np.sum(kernel)
    # convolution
    result = np.zeros((height, width))
    for i in range(height-2):
        for j in range(width-2):
            result[i+1,j+1] = (np.sum(img[i:i+3,j:j+3]*kernel[::-1,::-1]) > 0)
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)

def MyOpening(img, kernel):
	return MyDilation(MyErosion(img, kernel), kernel)

def MyClosing(img, kernel):
	return MyErosion(MyDilation(img, kernel), kernel)


if __name__ == '__main__':
    img = cv2.imread('ImagesSet\word_bw.bmp', 0)
    cv2.imshow('img', img)
    kernel = np.random.randint(0, 2, (3, 3)).astype(np.uint8)
    erosion = MyErosion(img, kernel)
    cv2.imshow('erosion', erosion)
    dilation = MyDilation(img, kernel)
    cv2.imshow('dilation', dilation)
    opening = MyOpening(img, kernel)
    cv2.imshow('opening', opening)
    Closing = MyClosing(img, kernel)
    cv2.imshow('Closing', Closing)
    cv2.waitKey(0)