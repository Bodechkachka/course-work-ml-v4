import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Предварительно обученная модель MobileNetV2
model = models.mobilenet_v2(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
num_classes = 5
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_folder = "D:/skyrim-dataset/transfer-learning-train"
valid_folder = "D:/skyrim-dataset/transfer-learning-valid"

train_dataset = datasets.ImageFolder(root=train_folder, transform=transform)
valid_dataset = datasets.ImageFolder(root=valid_folder, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=8, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=0.001)

# Обучение
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Оценка
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the validation set: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "mobilenet_transfer_learning.pth")