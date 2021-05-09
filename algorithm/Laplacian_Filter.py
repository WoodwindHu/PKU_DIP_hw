import cv2
import numpy as np
import math 

def MyLaplacian(img):
    # mask
    mask = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
    height, width = img.shape

    # pedding image
    img_ped = np.zeros((height+2, width+2))
    img_ped[1:-1, 1:-1] = img 
    img_ped[0,1:-1] = img[0,:]
    img_ped[-1,1:-1] = img[-1,:]
    img_ped[1:-1,0] = img[:,0]
    img_ped[1:-1,-1] = img[:,-1]
    img_ped[0,0] = img[0,0]
    img_ped[0,-1] = img[0,-1]
    img_ped[-1,0] = img[-1,0]
    img_ped[-1,-1] = img[-1,-1]

    # convolution
    result = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            result[i,j] = np.sum(img_ped[i:i+3,j:j+3]*mask)
    
    return result 

def MyLaplacianEnhance(img):
    result = MyLaplacian(img)
    result = (np.clip(img - result, 0, 255)).astype(np.uint8)
    return result

if __name__ == '__main__':
    img = cv2.imread('ImagesSet\moon.tif', 0)
    cv2.imshow('img', img)
    result = MyLaplacian(img)
    result0 = (np.clip(result, 0, 255)).astype(np.uint8)
    cv2.imshow('result0', result0)
    result = (np.clip(img - result, 0, 255)).astype(np.uint8)
    cv2.imshow('result', result)
    cv2.waitKey(0)
