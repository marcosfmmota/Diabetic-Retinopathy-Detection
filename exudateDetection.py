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
def exudateDetection(image,threshold,windowSize):
    row,col,chan = image.shape
    exudate_image = np.zeros([row,col])

    for r in range(0,image.shape[0] - windowSize ,windowSize):
        for c in range(0,image.shape[1] - windowSize ,windowSize):
            windowImage = image[r:r+windowSize,c:c+windowSize]
            histogram = histBGRChannels(windowImage)
            array_of_median=[]

            for i in range(len(histogram)):
                array_of_median.append(np.median(histogram[i]))

            print(array_of_median)
            if np.greater_equal(array_of_median,threshold).all():
                exudate_image[r:r+windowSize,c:c+windowSize] = np.ones([windowSize,windowSize])*255
            # else:
            #     exudate_image[r:r + windowSize, c:c + windowSize] = np.zeros([windowSize, windowSize])

    cv2.imwrite("exudates.jpg",exudate_image)


if __name__ == "__main__":
    image = cv2.imread("1.jpg")
    threshold = [0,75,150]
    windowSize = 50
    exudateDetection(image,threshold,windowSize)
    # hist = histRGBChannels(image)
    # color = ('b','g','r')
    # print(hist)
    # for i,col in enumerate(color):
    #     plt.plot(hist[i],color = col)
    #     plt.xlim([0,256])
    # plt.show()

