#!/usr/env python

import cv2
from matplotlib import pyplot as plt
import numpy as np

def histBGRChannels(image):
    histB = cv2.calcHist([image],[0],None,[256],[0,256])
    histG = cv2.calcHist([image], [1], None, [256], [0, 256])
    histR = cv2.calcHist([image], [2], None, [256], [0, 256])
    return [histB,histG,histR]

#Function that given an image and threshold decides if a window is in a exudate

def convertColorToBlackWhite(colorImage):
    im_gray = cv2.cvtColor(colorImage,cv2.COLOR_BGR2GRAY)
    (thresh,im_bw) = cv2.treshold(im_gray, 128,255 , cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    im_bw = cv2.threshold(im_gray,thresh,255,cv2.THRESH_BINARY)[1]
    return im_bw

def exudateDetection(image,threshold,windowSize):
    row,col,chan = image.shape
    #exudate_image = np.zeros([row,col,3])
    exudate_image = np.zeros([row, col])
    for r in range(0,image.shape[0] - windowSize ,windowSize):
        for c in range(0,image.shape[1] - windowSize ,windowSize):
            windowImage = image[r:r+windowSize,c:c+windowSize]
            histogram = histBGRChannels(windowImage)


            #
            # array_of_BGR=[]
            #
            # if np.greater_equal(array_of_BGR,threshold).all():
            #     exudate_image[r:r+windowSize,c:c+windowSize] = np.ones([windowSize,windowSize])*255
            # # else:
            # #     exudate_image[r:r + windowSize, c:c + windowSize] = np.zeros([windowSize, windowSize])

            lower_color = np.array(threshold)
            upper_color = np.array([255,255,255])

            exudate_mask = cv2.inRange(windowImage,lower_color,upper_color)
            #exudate_window = cv2.bitwise_and(windowImage,windowImage,mask=exudate_mask)

            exudate_image[r:r + windowSize, c:c + windowSize] = exudate_mask #exudate_window

    #binaryImage = convertColorToBlackWhite(exudate_image)
    return exudate_image


if __name__ == "__main__":
    image = cv2.imread("1008_equalized2.jpg")
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = [20,155,170]
    windowSize = 20
    exudate_image = exudateDetection(image,threshold,windowSize)
    cv2.imwrite("1008_equalized2_exudates.jpg",exudate_image)