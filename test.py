import os
import cv2

faces_dir = "D:/BTL/face-reidentification/faces"

for class_name in os.listdir(faces_dir):
    class_path = os.path.join(faces_dir, class_name)
    
    if not os.path.isdir(class_path):  # Bỏ qua nếu không phải thư mục
        continue
    
    for student_name in os.listdir(class_path):
        student_path = os.path.join(class_path, student_name)

        if not os.path.isdir(student_path):  # Bỏ qua nếu không phải thư mục
            continue

        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)

            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):  # Chỉ đọc file ảnh
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Không thể mở file ảnh: {img_path}")
