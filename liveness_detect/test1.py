import torchvision.models as models
import torch.optim as optim

# Load mô hình ResNet50
class LivenessModel(nn.Module):
    def __init__(self):
        super(LivenessModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 2)  # 2 lớp: Real (1) - Fake (0)

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LivenessModel().to(device)

# Cấu hình huấn luyện
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Lưu mô hình đã huấn luyện
torch.save(model.state_dict(), "liveness_model.pth")
print("Model trained and saved!")
