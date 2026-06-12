import streamlit as st
import torch
import os
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
st.title("Brain Tumor Detection App")

st.write("Upload an MRI scan to detect the type of brain tumor using Deep Learning.")

st.sidebar.header("About")

st.sidebar.write(
    "This application uses a Deep Learning model to detect brain tumors from MRI scans."
)
classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)


print("Exists:", os.path.exists("brain_tumor_model.pth"))
print("Is file:", os.path.isfile("brain_tumor_model.pth"))
print("Is directory:", os.path.isdir("brain_tumor_model.pth"))

torch.load(
    r"C:\Users\HP\Desktop\braintumor\brain_tumor_model.pth",
    map_location=torch.device('cpu')
)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# App title


# Upload image
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    

    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)

    image_tensor = image_tensor.to(device)
    model = model.to(device)

    # ===== PREDICTION SECTION STARTS HERE =====
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    _, predicted = torch.max(outputs, 1)

    prediction = classes[predicted.item()]
    confidence = probabilities[predicted.item()].item() * 100
    # ===== PREDICTION SECTION ENDS HERE =====

    st.write(f"Prediction: {prediction}")
    st.write(f"Confidence: {confidence:.2f}%")