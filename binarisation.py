import cv2


def give_bin_image(gray_img, show_mode=False):

    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)  # making Gaussian Blur in order to reduce the noise

    threshold_value = 185  # threshold value
    max_value = 255
    tv, binary_img = cv2.threshold(blurred,
                                   threshold_value, max_value, cv2.THRESH_BINARY_INV)

    if show_mode:
        cv2.imshow('Input Image', gray_img)
        cv2.imshow('Blurred Image', blurred)
        cv2.imshow('Binarized Image', binary_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return binary_img
