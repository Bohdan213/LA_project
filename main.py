from binarisation import give_bin_image
from CCA import give_CCA
import cv2
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


order_symbol = ['a', '1', '2', '6', '+']


symbols_frame = pd.DataFrame({"mu1": [], "mu2": [], "mu3": [],
                              "mu4": [], "mu5": [], "mu6": [],
                              "mu7": [], "value": []})


for symbol_num, symbol in enumerate(order_symbol):
    for idx in range(1, 2, 1):

        img_path = f"test_image_{symbol}_{idx}.jpg"
        gray_img = cv2.imread(img_path, 0)  # 0 because we load in gray scale mode
        binary_img = give_bin_image(gray_img, False)
        num_labels, labels, stats, centroids = give_CCA(binary_img, gray_img)

        for label in range(1, num_labels):
            component_mask = np.uint8(labels == label)

            moments = cv2.moments(component_mask)
            hu_moments = cv2.HuMoments(moments)
            hu_moments = hu_moments.flatten()

            hu_moments = np.append(hu_moments, symbol_num)
            symbols_frame.loc[len(symbols_frame)] = hu_moments


train_features = ["mu1", "mu2", "mu3", "mu4", "mu5", "mu6", "mu7"]
train_X = symbols_frame[train_features]
train_Y = symbols_frame.value

# print(symbols_frame.loc[symbols_frame.value == 0])

symbols_model = DecisionTreeRegressor(random_state=1, max_depth=10)
symbols_model.fit(train_X, train_Y)


img_path = "image_validation+.jpg"
gray_img = cv2.imread(img_path, 0)  # 0 because we load in gray scale mode
binary_img = give_bin_image(gray_img, False)
num_labels, labels, stats, centroids = give_CCA(binary_img, gray_img)

# print(num_labels)

for label in range(1, num_labels):
    component_mask = np.uint8(labels == label)

    moments = cv2.moments(component_mask)
    hu_moments = cv2.HuMoments(moments)
    hu_moments = hu_moments.flatten()

    validation = pd.DataFrame({"mu1": [], "mu2": [], "mu3": [],
                              "mu4": [], "mu5": [], "mu6": [],
                              "mu7": []})

    validation.loc[len(validation)] = hu_moments
    prediction = symbols_model.predict(validation)
    print(prediction)
