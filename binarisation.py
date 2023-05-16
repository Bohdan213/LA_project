import cv2
import numpy as np


def get_bin_image(gray_img, show_mode=False):
    # r = np.linalg.matrix_rank(gray_img)
    # u, d, v_t = np.linalg.svd(gray_img)
    # gray_img = u[:, :50] @ np.diag(d[:50]) @ v_t[:50, :]
    # gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # _, binary_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 199, 13)

    # binary_img = cv2.bitwise_not(binary_img)

    if show_mode:
        cv2.imshow('Input Image', gray_img)
        # cv2.imshow('Blurred Image', blurred)
        cv2.imshow('Binarized Image', binary_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return binary_img
