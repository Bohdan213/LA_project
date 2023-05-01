import cv2


def get_bin_image(gray_img, threshold_value=170, show_mode=False):

    blurred = cv2.GaussianBlur(gray_img, (7, 7), cv2.BORDER_DEFAULT)  # making Gaussian Blur in order to reduce the noise
    # threshold_value = 170  # threshold value
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
