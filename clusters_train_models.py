import pandas as pd
from binarisation import get_bin_image
from CCA import get_CCA
from DFZ import get_DFZ
import numpy as np
import cv2
from sklearn.svm import SVC
from joblib import dump
from sklearn.cluster import KMeans

order_symbol = ['a', 'b', 'c', '0', '1', '2', '3', '6', '+']

symbols_frame = pd.DataFrame()

num_clusters = 3

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
                symbols_frame = symbols_frame.append(pd.Series(pixels_intensity), ignore_index=True)


X = symbols_frame.iloc[:, 0:25]
y = symbols_frame.iloc[:, -1]


kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(X)
dump(kmeans, 'kluster/kmeans.joblib')

print(len(X))
models = []
for i in range(num_clusters):

    X_cluster = X[cluster_labels == i]
    print(len(X_cluster))
    y_cluster = y[cluster_labels == i]

    model = SVC()
    model.fit(X_cluster, y_cluster)
    models.append(model)


for i, model in enumerate(models):
    dump(model, f'models/model_cluster_{i}.joblib')