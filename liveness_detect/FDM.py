import numpy as np
import cv2
import tensorflow as tf
import keras
# Load mô hình đã train
model = keras.models.load_model("fake_image_detector.h5")

# Hàm dự đoán ảnh
def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize về kích thước phù hợp
    image = image / 255.0  # Chuẩn hóa
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch

    prediction = model.predict(image)
    return "Ảnh giả" if prediction > 0.5 else "Ảnh thật"

# Kiểm tra với một ảnh
image_path = "face-reidentification/liveness_detect/dataset/train/real/LeNguyenQuocANh_AI1914 (9).jpg"
print(predict_image(image_path))
