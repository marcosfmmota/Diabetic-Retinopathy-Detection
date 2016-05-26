import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


def createClare(channelB, channelG, channelR):

    # create a CLAHE object(Contrast Limited Adaptive Histogram Equalization)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(channelB)
    cv2.imwrite('bluechannel.jpg', cl1)

    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl2 = clahe.apply(channelG)
    cv2.imwrite('greenchanel.jpg', cl2)

    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl3 = clahe.apply(channelR)
    cv2.imwrite('redChannel.jpg', cl3)

    result = cv2.merge((cl1, cl2, cl3))

    dirname = "afterEqualization"
    os.mkdir(dirname)


    cv2.imwrite(os.path.join(dirname, "image after equalization.jpg"), result)

    return result

def medianFilter(image):

    median = cv2.medianBlur(image, 5, None)

    b = median[:, :, 0]
    g = median[:, :, 1]
    r = median[:, :, 2]

    cv2.imwrite("Median result.jpeg", median)
    return b,g,r



#img = cv2.imread()


def showHistogram(image ):
    # getting the histogram

    result = image

    color = ("b", "g", "r")

    for i, col in enumerate(color):
        histr = cv2.calcHist([result], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.show()


if __name__ == "__main__":

    imageSource = cv2.imread('932_left.jpeg')

    b,g,r = medianFilter(imageSource)

    afterEqualization = createClare(b,g,r)
    showHistogram(afterEqualization)



#reorder = cv2.merge((cl3,cl2,cl1)) #RGB



#cv2.imwrite("image after reorder.jpg",reorder)