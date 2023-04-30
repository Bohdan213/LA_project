from binarisation import get_bin_image
from CCA import get_CCA
from DFZ import get_DFZ
import cv2
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


order_symbol = ['a', '1', '2', '6', '+']


# symbols_frame = pd.DataFrame({"v1": [], "v2": [], "v3": [],
#                                   "v4": [], "v5": [], "v6": [],
#                                   "v7": [], "v8": [], "v9": [],
#                                    "v10": [], "v11": [], "v12": [],
#                                    "v15": [], "v14": [], "v13": [],
#                                    "v16": [], "v17": [], "v18": [],
#                                    "v19": [], "v20": [], "v21": [],
#                                    "v22": [], "v23": [], "v24": [],
#                                    "v25": [], "value": []})

symbols_frame = pd.DataFrame({"v1": [], "v2": [], "v3": [],
                                  "v4": [], "v5": [], "v6": [],
                                  "v7": [], "v8": [], "v9": [],
                                   "v10": [], "v11": [], "v12": [],
                                   "v15": [], "v14": [], "v13": [],
                                   "v16": [], "value": []})


for symbol_num, symbol in enumerate(order_symbol):
    for idx in range(1, 2, 1):
        img_path = f"images/test_image_{symbol}_{idx}.jpg"
        gray_img = cv2.imread(img_path, 0)  # 0 because we load in gray scale mode
        binary_img = get_bin_image(gray_img, False)
        num_labels, labels, stats, centroids = get_CCA(binary_img, gray_img)
        for label in range(1, num_labels):

            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            if width > 20 or height > 20:
                component_mask = np.uint8(labels == label)

                # moments = cv2.moments(component_mask)
                # hu_moments = cv2.HuMoments(moments)
                # hu_moments = hu_moments.flatten()

                pixels_intensity = get_DFZ(labels, label, stats)
                pixels_intensity = np.array(pixels_intensity)
                pixels_intensity = np.append(pixels_intensity, symbol_num)
                symbols_frame.loc[len(symbols_frame)] = pixels_intensity


train_X = symbols_frame.iloc[:, 0:16]
train_Y = symbols_frame.value

# print(symbols_frame.loc[symbols_frame.value == 2])

symbols_model = DecisionTreeRegressor(random_state=1, max_depth=7)
symbols_model.fit(train_X, train_Y)


img_path = "images/image_validation_sentence.jpg"
gray_img = cv2.imread(img_path, 0)  # 0 because we load in gray scale mode
binary_img = get_bin_image(gray_img, False)
num_labels, labels, stats, centroids = get_CCA(binary_img, gray_img)

for label in range(1, num_labels):
    width = stats[label, cv2.CC_STAT_WIDTH]
    height = stats[label, cv2.CC_STAT_HEIGHT]

    x,y,w,h,_ = stats[label]

    print(x)
    if width > 20 or height > 20:

        pixels_intensity = get_DFZ(labels, label, stats)
        # component_mask = np.uint8(labels == label)
        #
        # moments = cv2.moments(component_mask)
        # hu_moments = cv2.HuMoments(moments)
        # hu_moments = hu_moments.flatten()

        validation = pd.DataFrame({"v1": [], "v2": [], "v3": [],
                                  "v4": [], "v5": [], "v6": [],
                                  "v7": [], "v8": [], "v9": [],
                                   "v10": [], "v11": [], "v12": [],
                                   "v15": [], "v14": [], "v13": [],
                                   "v16": []})

        validation.loc[len(validation)] = pixels_intensity
        prediction = symbols_model.predict(validation)
        print(prediction)
