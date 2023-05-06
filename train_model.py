import pandas as pd
from binarisation import get_bin_image
from CCA import get_CCA
from DFZ import get_DFZ
import numpy as np
import cv2
from sklearn.svm import SVC
from joblib import dump


order_symbol = ['a', 'b', 'c', '0', '1', '2', '3', '6', '+']

symbols_frame = pd.DataFrame({"v1": [], "v2": [], "v3": [],
                                  "v4": [], "v5": [], "v6": [],
                                  "v7": [], "v8": [], "v9": [],
                                   "v10": [], "v11": [], "v12": [],
                                   "v15": [], "v14": [], "v13": [],
                                   "v16": [], "v17": [], "v18": [],
                                   "v19": [], "v20": [], "v21": [],
                                   "v22": [], "v23": [], "v24": [],
                                   "v25": [], "value": []})


for symbol_num, symbol in enumerate(order_symbol):
    for idx in range(1, 4, 1):
        img_path = f"images/test/test_image_{symbol}_{idx}.jpg"
        gray_img = cv2.imread(img_path, 0)  # 0 because we load in gray scale mode
        binary_img = get_bin_image(gray_img, show_mode=False)
        num_labels, labels, stats, centroids = get_CCA(binary_img)
        for label in range(1, num_labels):

            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            if width > 30 or height > 30:
                component_mask = np.uint8(labels == label)
                pixels_intensity = get_DFZ(labels, label, stats, 5)
                pixels_intensity = np.array(pixels_intensity)
                pixels_intensity = np.append(pixels_intensity, ord(symbol))
                symbols_frame.loc[len(symbols_frame)] = pixels_intensity


train_X = symbols_frame.iloc[:, 0:25]
train_Y = symbols_frame.value

symbols_model = SVC()
symbols_model.fit(train_X, train_Y)
dump(symbols_model, 'model.joblib')
