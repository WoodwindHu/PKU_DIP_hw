import cv2
import numpy as np
import math 

def MyHE(img):
    img = img.astype(np.uint8)
    height, width = img.shape 
    h = np.zeros(256, dtype='int64')
    T = np.zeros(256)
    for i in range(height):
        for j in range(width):
            h[img[i,j]] += 1
    T[0] = h[0]
    for i in range(1, 256):
        T[i] = T[i-1] + h[i]
    T = T / (height * width)
    result = np.zeros((height,width), dtype='uint8')
    for i in range(height):
        for j in range(width):
            result[i,j] = T[img[i,j]] * 255
    h_result = np.zeros(256, dtype='int64')
    for i in range(height):
        for j in range(width):
            h_result[result[i,j]] += 1
    return result, T, h, h_result

def MyHE_RGB(img):
    if len(img.shape) == 2:
        return MyHE(img)[0]
    result = np.stack([MyHE(img[:,:,i])[0] for i in range(3)], axis=2)
    return result

def rgb2hsi(img):
    img = img.astype(np.int64)
    height = img.shape[0]
    width = img.shape[1]
    HSI = np.zeros_like(img, dtype=np.float64)
    for i in range(height):
        for j in range(width):
            num = 0.5*(img[i,j,0]-img[i,j,1]+img[i,j,0]-img[i,j,2])
            den = ((img[i,j,0]-img[i,j,1])**2+(img[i,j,0]-img[i,j,2])*(img[i,j,1]-img[i,j,2]))**0.5
            if den == 0:
                HSI[i,j,0] = 0
            else:
                theta = math.acos(num/den)
                HSI[i,j,0] = (theta if img[i,j,2] <= img[i,j,1] else 2*math.pi-theta)
            if img[i,j,0]+img[i,j,1]+img[i,j,2] == 0:
                HSI[i,j,1] = 0
            else:
                HSI[i,j,1] = 1 - 3/(img[i,j,0]+img[i,j,1]+img[i,j,2])*min(img[i,j,0],img[i,j,1],img[i,j,2])
            HSI[i,j,2] = (img[i,j,0]+img[i,j,1]+img[i,j,2])/3
    return HSI

def hsi2rgb(img):
    height = img.shape[0]
    width = img.shape[1]
    RGB = np.zeros_like(img, dtype=np.float64)
    for i in range(height):
        for j in range(width):
            if img[i,j,0] < 2*math.pi/3:
                RGB[i,j,0] = img[i,j,2]*(1+img[i,j,1]*math.cos(img[i,j,0])/math.cos(math.pi/3-img[i,j,0]))
                RGB[i,j,2] = img[i,j,2]*(1-img[i,j,1])
                RGB[i,j,1] = 3*img[i,j,2]-RGB[i,j,0]-RGB[i,j,2] 
            elif img[i,j,0] < 4*math.pi/3:
                RGB[i,j,1] = img[i,j,2]*(1+img[i,j,1]*math.cos(img[i,j,0]-2*math.pi/3)/math.cos(math.pi-img[i,j,0]))
                RGB[i,j,0] = img[i,j,2]*(1-img[i,j,1])
                RGB[i,j,2] = 3*img[i,j,2]-RGB[i,j,0]-RGB[i,j,1]
            else:
                RGB[i,j,2] = img[i,j,2]*(1+img[i,j,1]*math.cos(img[i,j,0]-4*math.pi/3)/math.cos(5*math.pi/3-img[i,j,0]))
                RGB[i,j,1] = img[i,j,2]*(1-img[i,j,1])
                RGB[i,j,0] = 3*img[i,j,2]-RGB[i,j,1]-RGB[i,j,2] 
    return RGB

def MyHE_HSI(HSI):
    I_= MyHE(HSI[:,:,2])[0]
    HSI_ = np.stack([HSI[:,:,0], HSI[:,:,1], I_], axis=2)
    result = hsi2rgb(HSI_)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


if __name__ == '__main__':
    img=cv2.imread('ImagesSet/histeq1.jpg',0)
    cv2.imshow('img', img)
    gray= MyHE(img)[0]
    cv2.imshow('gray', gray)
    img=cv2.imread('ImagesSet/histeqColor.jpg',1)
    cv2.imshow('img', img)
    rgb = MyHE_RGB(img)
    cv2.imshow('rgb', rgb)
    HSI = rgb2hsi(img)
    hsi = MyHE_HSI(HSI)
    cv2.imshow('hsi', hsi)
    cv2.waitKey(0)