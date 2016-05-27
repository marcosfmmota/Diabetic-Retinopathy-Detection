#!/usr/env python

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def histBGRChannels(image):
    histB = cv2.calcHist([image], [0], None, [256],[0,256])
    histG = cv2.calcHist([image], [1], None, [256], [0, 256])
    histR = cv2.calcHist([image], [2], None, [256], [0, 256])
    return [histB, histG, histR]

# Function that given an image and threshold decides if a window is in a exudate


def convertColorToBlackWhite(colorImage):
    im_gray = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
    (thresh,im_bw) = cv2.treshold(im_gray, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    im_bw = cv2.threshold(im_gray, thresh,255, cv2.THRESH_BINARY)[1]
    return im_bw


def maskWhiteCounter (mask_input):
    counter = 0
    for r in range(mask_input.shape[0]):
        for c in range(mask_input.shape[1]):
            if mask_input.item(r, c) == 255:
                counter+=1
    return counter


def exudateDetection(image, threshold, windowSize):
    row,col,chan = image.shape
    #exudate_image = np.zeros([row,col,3])
    exudate_image = np.zeros([row, col])
    for r in range(0, image.shape[0] - windowSize, windowSize):
        for c in range(0, image.shape[1] - windowSize, windowSize):
            windowImage = image[r:r+windowSize, c:c+windowSize]
            histogram = histBGRChannels(windowImage)

            #
            # array_of_BGR=[]
            #
            # if np.greater_equal(array_of_BGR,threshold).all():
            #     exudate_image[r:r+windowSize,c:c+windowSize] = np.ones([windowSize,windowSize])*255
            #  else:
            #      exudate_image[r:r + windowSize, c:c + windowSize] = np.zeros([windowSize, windowSize])

            lower_color = np.array(threshold)
            upper_color = np.array([70, 255, 255])

            exudate_mask = cv2.inRange(windowImage, lower_color, upper_color)
            # exudate_window = cv2.bitwise_and(windowImage,windowImage,mask=exudate_mask)
            # print(exudate_mask)

            exudate_image[r:r + windowSize, c:c + windowSize] = exudate_mask  # exudate_window

    #binaryImage = convertColorToBlackWhite(exudate_image)
    return exudate_image

def gettingOpticalDisk(image2):
    #erosion
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(image2, kernel, iterations = 1)


    #dilation

    dilation = cv2.dilate(erosion, kernel, iterations=1)

    return dilation





if __name__ == "__main__":
    pathFolder = "C:\\Users\\MarcosFelipe\\Documents\\RetinaNormalizedSet\\2\\"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
    threshold = [30, 155, 180]
    windowSize = 50
    exudateFolder = pathFolder+"exudate\\"
    array_exudate_pixels = []

    if not os.path.exists(exudateFolder):
        os.mkdir(exudateFolder)

    for file_name in filesArray:
        image = cv2.imread(pathFolder+file_name)
        #im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        exudate_image = exudateDetection(image, threshold, windowSize)
        file_name_no_extension = os.path.splitext(file_name)[0]
        counter = maskWhiteCounter(exudate_image)
        array_exudate_pixels.append(counter)
        cv2.imwrite(exudateFolder+file_name_no_extension+"_exudates.jpg",exudate_image)

    plt.hist(array_exudate_pixels,bins=20)
    plt.show()

    print(np.average(array_exudate_pixels))

    ## parte de teste

    # imagem2 = cv2.imread("1008_equalized2_exudates.jpg", 0)
    # imageAfterErosion = gettingOpticalDisk(imagem2)
    # cv2.imwrite("image after erosion.jpg", imagem2)
    #
    # ## fim dos testes