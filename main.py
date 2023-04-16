from binarisation import give_bin_image
from CCA import give_CCA
import cv2


img_path = "test_img.jpg"
gray_img = cv2.imread(img_path, 0)  # 0 because we load in gray scale mode
binary_img = give_bin_image(gray_img, True)
give_CCA(binary_img, gray_img)
