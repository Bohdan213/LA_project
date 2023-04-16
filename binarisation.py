def give_bin_image(img_path, show_mode=False):
    import cv2
    img = cv2.imread(img_path, 0)  # 0 because we load in gray scale mode

    threshold_value = 175  # threshold value
    max_value = 255
    tv, binary_img = cv2.threshold(img, threshold_value, max_value, cv2.THRESH_BINARY)

    if show_mode:
        cv2.imshow('Input Image', img)
        cv2.imshow('Binarized Image', binary_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return binary_img
