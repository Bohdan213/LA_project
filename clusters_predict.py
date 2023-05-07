from binarisation import get_bin_image
from CCA import get_CCA
from DFZ import get_DFZ
import cv2
import pandas as pd
from joblib import load


kmeans = load("kluster/kmeans.joblib")

models = []
for i in range(0, 3):
    model = load(f"models/model_cluster_{i}.joblib")
    models.append(model)

img_path = "images/validation/image_validation_sentence2.jpg"
gray_img = cv2.imread(img_path, 0)  # 0 because we load in gray scale mode
binary_img = get_bin_image(gray_img, show_mode=False)
num_labels, labels, stats, centroids = get_CCA(binary_img, gray_img)

result = []
for label in range(1, num_labels):
    width = stats[label, cv2.CC_STAT_WIDTH]
    height = stats[label, cv2.CC_STAT_HEIGHT]

    x, y, w, h, _ = stats[label]

    if width > 30 or height > 30:
        pixels_intensity = get_DFZ(labels, label, stats, 5)

        validation = pd.DataFrame([pixels_intensity])
        kluster_prediction = kmeans.predict(validation.values)

        prediction = models[kluster_prediction[0]].predict(validation)
        result.append((x, chr(int(prediction[0]))))

result.sort()
print(*[_[1] for _ in result])
