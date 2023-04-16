import cv2
import numpy as np


def give_CCA(binary_img, gray_img):
    analysis = cv2.connectedComponentsWithStats(binary_img, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = analysis
    output = np.zeros(gray_img.shape, dtype="uint8")
    for i in range(1, numLabels):

        # Area of the component
        area = stats[i, cv2.CC_STAT_AREA]  # the square of character
        # width = stats[i, cv2.CC_STAT_WIDTH]
        if 20 < area < 400:
            componentMask = (labels == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)

    cv2.imshow("Filtered Components", output)
    cv2.waitKey(0)
