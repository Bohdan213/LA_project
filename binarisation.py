import cv2
import numpy as np


def get_bin_image(gray_img, show_mode=False):
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 199, 13)


    if show_mode:
        cv2.imshow('Input Image', gray_img)
        # cv2.imshow('Blurred Image', blurred)
        cv2.imshow('Binarized Image', binary_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return binary_img
