import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Paths
train_path = "dataset/Training"
test_path = "dataset/Testing"

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Datasets
train_dataset = datasets.ImageFolder(
    root=train_path,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=test_path,
    transform=transform
)

print("Class Mapping:")
print(train_dataset.class_to_idx)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 4)

model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

# Epochs
epochs = 5

# Training loop
for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    print(f"\nEpoch [{epoch+1}/{epochs}]")
    print(f"Training Loss: {running_loss/len(train_loader):.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")

# Save model
os.makedirs("saved_models", exist_ok=True)

torch.save(
    model.state_dict(),
    "saved_models/brain_tumor_model.pth"
)
torch.save(model.state_dict(), "brain_tumor_model.pth")

torch.save(model.state_dict(), "brain_tumor_model.pth")
print("\nModel saved successfully!")