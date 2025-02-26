import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time  # Thêm thư viện đo thời gian

# **BƯỚC 1: Chuẩn bị dữ liệu**
data_transforms = transforms.Compose([
    transforms.Resize((112, 112)),  
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder(root=r"D:\BTL\face-reidentification\faces\AI1914", transform=data_transforms)
test_dataset = datasets.ImageFolder(root=r"D:\BTL\face-reidentification\faces\AI1914", transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# **BƯỚC 2: Định nghĩa mô hình SCRFD**
class SCRFD(nn.Module):
    def __init__(self, num_classes=10):
        super(SCRFD, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(256 * 14 * 14, 512)  # Flatten đầu ra
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        logits = self.classifier(features)
        return logits, features  # Trả về logits và embeddings

# Khởi tạo mô hình
num_classes = len(train_dataset.classes)
model = SCRFD(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# **BƯỚC 3: Định nghĩa loss function và optimizer**
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# **BƯỚC 4: Huấn luyện và đánh giá mô hình**
num_epochs = 10
total_start_time = time.time()  # Đo tổng thời gian train

for epoch in range(num_epochs):
    epoch_start_time = time.time()  # Đo thời gian mỗi epoch

    ### 1️⃣ TRAINING PHASE ###
    model.train()
    train_loss, train_correct, total_train = 0, 0, 0
    train_start_time = time.time()  # Đo thời gian train
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_time = time.time() - train_start_time  # Kết thúc đo thời gian train
    train_acc = 100 * train_correct / total_train

    ### 2️⃣ TESTING PHASE ###
    model.eval()
    test_loss, test_correct, total_test = 0, 0, 0
    test_start_time = time.time()  # Đo thời gian test
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_time = time.time() - test_start_time  # Kết thúc đo thời gian test
    test_acc = 100 * test_correct / total_test

    epoch_time = time.time() - epoch_start_time  # Thời gian chạy 1 epoch

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Time: {train_time:.2f}s | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test Time: {test_time:.2f}s | "
          f"Epoch Time: {epoch_time:.2f}s")

# **BƯỚC 5: Lưu mô hình sau khi huấn luyện**
model_path = "scrfd_trained.pth"
torch.save(model.state_dict(), model_path)
total_train_time = time.time() - total_start_time  # Tổng thời gian huấn luyện
print(f"✅ Mô hình đã được lưu tại {model_path}")
print(f"⏳ Tổng thời gian train: {total_train_time:.2f}s")
