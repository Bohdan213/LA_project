import cv2
import numpy as np


def get_CCA(binary_img, gray_img=None):
    analysis = cv2.connectedComponentsWithStats(binary_img, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = analysis
    if gray_img is not None:
        output = np.zeros(gray_img.shape, dtype="uint8")
        for i in range(1, numLabels):
            # Area of the component
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            if width > 20 or height > 20:
                componentMask = (labels == i).astype("uint8") * 255
                output = cv2.bitwise_or(output, componentMask)

        # cv2.imshow("Filtered Components", output)
        # cv2.waitKey(0)

    return analysis
