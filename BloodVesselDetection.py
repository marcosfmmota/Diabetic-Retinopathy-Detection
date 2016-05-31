#!/usr/env python

import cv2
import numpy as np


def unsharp_image(image):
    gaussian_smooth = cv2.GaussianBlur(image, (5, 5), 10)
    unsharped = cv2.addWeighted(image, 3, gaussian_smooth, -0.5, 0)
    return unsharped


def gabor_wavelet(image):
    filters = []
    ksize = 7
    for theta in np.arange(0, np.pi, np.pi/16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)

    accum = np.zeros_like(image)
    for kern in filters:
        fimg = cv2.filter2D(image, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)

    return accum

def green_channel_from_image(image_name):
    color_image = cv2.imread(image_name)
    return color_image[:, :, 1]


def blood_vessel_detection(image_name):
    image_green = green_channel_from_image(image_name)
    enhanced_image = gabor_wavelet(image_green)
    resized = cv2.resize(enhanced_image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("gabel", resized)
    cv2.waitKey(0)
    image_unsharp = unsharp_image(enhanced_image)
    resized = cv2.resize(image_unsharp, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("gabel", resized)
    cv2.waitKey(0)
    vessels_edge = cv2.Canny(image_unsharp, 20, 50, apertureSize=3, L2gradient=True)
    kernel_closing = np.ones((9, 9), np.uint8)
    vessels_mask = cv2.morphologyEx(vessels_edge, cv2.MORPH_CLOSE, kernel_closing)
    return vessels_mask


if __name__ == "__main__":
    image_name = "1008_equalized2.jpg"
    blood_vessels = blood_vessel_detection(image_name)
    resized = cv2.resize(blood_vessels, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("blood_vessel", resized)
    cv2.waitKey(0)
