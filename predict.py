import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models

classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)

model.load_state_dict(torch.load("brain_tumor_model.pth"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load image
image = Image.open("test.jpeg").convert("RGB")
image = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print("Prediction:", classes[predicted.item()])