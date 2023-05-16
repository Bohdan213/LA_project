from binarisation import get_bin_image
from CCA import get_CCA
from DFZ import get_DFZ
import cv2
import pandas as pd
from joblib import load
from profiles import get_profile
import time

classes = ['a', 'b', 'c', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+',
                'x', 'y', 'd', 'e', 'z', '-', 'belongs', 'big_e', 'big_a']

result = []
total_res = 0
total_cnt = 0

kmeans = load("kluster/kmeans.joblib")
num_clusters = kmeans.n_clusters
print(num_clusters)

models = []
for i in range(0, num_clusters):
    model = load(f"models/model_cluster_{i}.joblib")
    models.append(model)

a = time.time()
result = []
for symbol in classes:
    img_path = f"images/report_validation/image{symbol}.jpg"
    gray_img = cv2.imread(img_path, 0)  # 0 because we load in gray scale mode
    binary_img = get_bin_image(gray_img, show_mode=False)
    num_labels, labels, stats, centroids = get_CCA(binary_img)

    res = 0
    cnt = 0
    for label in range(1, num_labels):
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        x, y, w, h, _ = stats[label]

        if width > 30 or height > 30:
            pixels_intensity = get_DFZ(labels, label, stats, 7)
            profiles = get_profile(label, labels, stats)
            image_features = [*pixels_intensity]
            image_features.extend(profiles)
            validation = pd.DataFrame([image_features])
            kluster_prediction = kmeans.predict(validation.values)

            prediction = models[kluster_prediction[0]].predict(validation)
            if classes[int(prediction[0])] == symbol:
                res += 1
            cnt += 1
    total_res += res
    total_cnt += cnt
    res = res / cnt
    result.append((symbol, res))
print(time.time() - a)

print(result)
print(total_res / total_cnt)
