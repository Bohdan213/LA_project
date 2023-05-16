import pandas as pd
from binarisation import get_bin_image
from CCA import get_CCA
from DFZ import get_DFZ
import numpy as np
import cv2
from sklearn.svm import SVC
from joblib import dump
from sklearn.cluster import KMeans
from profiles import get_profile


classes = ['a', 'b', 'c', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+',
                'x', 'y', 'd', 'e', 'z', '-', 'belongs', 'big_e', 'big_a']


classes_code = {}
for idx in range(len(classes)):
    classes_code[classes[idx]] = idx

symbols_frame = pd.DataFrame()

num_clusters = 1

for symbol_num, symbol in enumerate(classes):
    for idx in range(1, 4, 1):
        img_path = f"images/test/test_image_{symbol}_{idx}.jpg"
        print(img_path)
        gray_img = cv2.imread(img_path, 0)  # 0 because we load in gray scale mode
        binary_img = get_bin_image(gray_img, show_mode=False)
        num_labels, labels, stats, centroids = get_CCA(binary_img)
        for label in range(1, num_labels):
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            if width > 40 or height > 40:
                pixels_intensity = get_DFZ(labels, label, stats, 8)
                profiles = get_profile(label, labels, stats)
                image_features = [*pixels_intensity]
                image_features.extend(profiles)
                image_features.append(classes_code[symbol])
                symbols_frame = symbols_frame.append(pd.Series(image_features), ignore_index=True)

X = symbols_frame.iloc[:, 0:68]
y = symbols_frame.iloc[:, -1]

class_centroids = []

for class_label in classes:
    class_points = X[y == classes_code[class_label]]
    class_centroid = np.mean(class_points, axis=0)
    class_centroids.append(class_centroid)

class_centroids = np.array(class_centroids)

kmeans = KMeans(n_clusters=num_clusters)
class_cluster_labels = kmeans.fit_predict(class_centroids)

dump(kmeans, 'kluster/kmeans.joblib')

symbols_frame['Cluster'] = [class_cluster_labels[int(class_label)] for class_label in y]

models = []
for cluster in range(num_clusters):
    correspond_cluster = symbols_frame[symbols_frame['Cluster'] == cluster]
    X_cluster = correspond_cluster.iloc[:, 0:68]
    y_cluster = correspond_cluster.iloc[:, -2]
    model = SVC(C=10.0, kernel='rbf', gamma='scale', decision_function_shape='ovo')
    model.fit(X_cluster, y_cluster)
    models.append(model)

for i, model in enumerate(models):
    dump(model, f'models/model_cluster_{i}.joblib')


