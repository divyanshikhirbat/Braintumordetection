# Braintumordetection
It is an AI-powered application that analyzes brain MRI scans and detects tumors using Deep Learning.
📌 **Overview**
Brain Tumor Detection App ek deep learning based web application hai jo brain MRI scans ko analyze karke tumor detect karta hai. User sirf ek MRI image upload karta hai aur model turant prediction de deta hai with confidence score.

✨ **Features**

Upload any brain MRI scan and get instant results
Shows prediction — Tumor or Normal
Displays confidence percentage
Clean and simple web interface
No coding knowledge required to use


🗂️ **Project Structure**
braintumor/
├── app.py              # Streamlit web app (main file)
├── model.py            # Neural network architecture
├── train.py            # Model training script
├── predict.py          # Standalone prediction script
├── brain_tumor_model.pth   # Trained model weights
├── requirements.txt    # Dependencies
└── data/
    ├── tumor/          # Tumor MRI images
    └── normal/         # Healthy MRI images

🚀 **Run Locally**
bashpip install -r requirements.txt
python -m streamlit run app.py
Open browser and go to http://localhost:8501

🏋️ **Train Your Own Model**
Download dataset from Kaggle:

👉 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Place images in data/tumor/ and data/normal/, then run:
bashpython train.py

📊 **Model Details**
PropertyDetailsFrameworkPyTorchInput Size224 × 224 pxOutputTumor / Normal

⚠️ **Disclaimer**
For educational purposes only. Not a substitute for professional medical diagnosis.
