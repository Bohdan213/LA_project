from binarisation import get_bin_image
from CCA import get_CCA
from DFZ import get_DFZ
import cv2
import numpy as np
from sklearn.svm import SVC
import pandas as pd



order_symbol = ['a', 'b', '1', '2', '6', '+']

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
    for idx in range(1, 3, 1):
        img_path = f"images/test/test_image_{symbol}_{idx}.jpg"
        gray_img = cv2.imread(img_path, 0)  # 0 because we load in gray scale mode
        binary_img = get_bin_image(gray_img, False)
        num_labels, labels, stats, centroids = get_CCA(binary_img)
        for label in range(1, num_labels):

            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            if width > 30 or height > 30:
                component_mask = np.uint8(labels == label)
                pixels_intensity = get_DFZ(labels, label, stats)
                pixels_intensity = np.array(pixels_intensity)
                pixels_intensity = np.append(pixels_intensity, ord(symbol))
                symbols_frame.loc[len(symbols_frame)] = pixels_intensity

train_X = symbols_frame.iloc[:, 0:16]
train_Y = symbols_frame.value

symbols_model = SVC()
symbols_model.fit(train_X, train_Y)

img_path = "images//validation/image_validation_sentence4.jpg"
gray_img = cv2.imread(img_path, 0)  # 0 because we load in gray scale mode
binary_img = get_bin_image(gray_img, False)
num_labels, labels, stats, centroids = get_CCA(binary_img, gray_img)

result = []

for label in range(1, num_labels):
    width = stats[label, cv2.CC_STAT_WIDTH]
    height = stats[label, cv2.CC_STAT_HEIGHT]

    x, y, w, h, _ = stats[label]

    if width > 30 or height > 30:
        pixels_intensity = get_DFZ(labels, label, stats)

        validation = pd.DataFrame({"v1": [], "v2": [], "v3": [],
                                   "v4": [], "v5": [], "v6": [],
                                   "v7": [], "v8": [], "v9": [],
                                   "v10": [], "v11": [], "v12": [],
                                   "v15": [], "v14": [], "v13": [],
                                   "v16": []})

        validation.loc[len(validation)] = pixels_intensity
        prediction = symbols_model.predict(validation)
        result.append((x, chr(int(prediction[0]))))

result.sort()
print(*[_[1] for _ in result])
