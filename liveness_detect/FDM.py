import numpy as np
import cv2
import keras
# Load mô hình đã train
model = keras.models.load_model(r"D:\Python\Projects\face-reidentification\liveness_detect\model_1.h5")

# Hàm dự đoán ảnh
def predict_image(image):
    if isinstance(image, np.ndarray):
        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)
        return "Fake" if prediction < 0.5 else "Real"
