
import numpy as np
import cv2
import keras
# Load mô hình đã train
model = keras.models.load_model(r"D:\Python\Projects\face-reidentification\liveness_detect\fake_image_detector.h5")

# Hàm dự đoán ảnh
def predict_image(image):
    # If input is already a numpy array (cropped face)
    if isinstance(image, np.ndarray):
        # Resize the image
        image = cv2.resize(image, (128, 128))
        # Normalize
        image = image / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)
        print(prediction)
        return "Fake" if prediction > 0.5 else "Real"
