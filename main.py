import os
import io
import torch
import torch.nn as nn
import joblib
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from torchvision import models, transforms
from PIL import Image

# --- 1. SETUP ---
app = FastAPI(title="Medical AI Integration - HUMIC Engineering")
os.makedirs("static/outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. LOAD MODELS ---
def load_dl_model(path):
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    # Berdasarkan error sebelumnya, model Anda memiliki 3 kelas output
    model.classifier = nn.Linear(num_ftrs, 3) 
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

dl_model = load_dl_model("models/model_densenet121_deeplearning.pth")
bundle = joblib.load("models/model_lgbm_machinelearning.joblib")

scaler = bundle['scaler']
anova_selector = bundle['reducer']
lgbm_model = bundle['classifier']
label_encoder = bundle['label_encoder']

print("✅ Semua model (DL & ML Bundle) berhasil dimuat!")

# --- 3. PREPROCESSING (MEDIAN FILTER + CLAHE) ---
def preprocess_image(image_bytes):
    # Decode image ke OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Median Filter (Noise Reduction)
    img_median = cv2.medianBlur(img, 5)

    # 2. CLAHE (Contrast Enhancement)
    img_gray = cv2.cvtColor(img_median, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    
    # Convert back to RGB untuk model DenseNet
    img_final = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_final)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img_pil).unsqueeze(0).to(device)

# --- 4. ENDPOINT PREDICT ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    input_tensor = preprocess_image(contents)

    # Deep Learning Side: Feature Extraction
    with torch.no_grad():
        features = dl_model.features(input_tensor)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        deep_features = torch.flatten(features, 1).cpu().numpy()

    # Machine Learning Side: Bundle Pipeline
    features_scaled = scaler.transform(deep_features)
    features_reduced = anova_selector.transform(features_scaled)
    prediction = lgbm_model.predict(features_reduced)
    probability = lgbm_model.predict_proba(features_reduced)

    class_name = label_encoder.inverse_transform(prediction)[0]

    return {
        "status": "success",
        "patient_data": {"filename": file.filename},
        "ai_results": {
            "classification": class_name,
            "confidence": f"{np.max(probability) * 100:.2f}%",
            "gradcam_output": f"/static/outputs/gradcam_{file.filename}.png" # Placeholder
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)