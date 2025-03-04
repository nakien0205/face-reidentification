import tensorflow as tf
from keras import layers, models
from keras._tf_keras.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

# Đường dẫn đến dataset
train_dir = r"D:\Python\Projects\face-reidentification\liveness_detect\dataset\train"
test_dir = r"D:\Python\Projects\face-reidentification\liveness_detect\dataset\test"

# Load dữ liệu
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode="binary",
    class_names=["real", "fake"]  # Định nghĩa lại thứ tự nhãn
)

test_ds = image_dataset_from_directory(
    test_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode="binary",
    class_names=["real", "fake"]
)


# Chuẩn hóa dữ liệu
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Phân loại nhị phân (real hoặc fake)
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(train_ds, epochs=20, validation_data=test_ds, batch_size=8)

# Lưu mô hình sau khi train
model.save("fake_image_detector.h5")

