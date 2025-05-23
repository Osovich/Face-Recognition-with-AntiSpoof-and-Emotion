import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import os
from face_recognition.embnet import EmbNet

# ========== CONFIG ==========
MODEL_PATH = "models/embnet_triplet_15k.pth"
CLASS_DB_PATH = "models/class_db.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== MODEL ==========
class ClassifierFeats(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = nn.Sequential(*list(model.children())[:-1])
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return torch.nn.functional.normalize(x, dim=1)

@st.cache_resource(show_spinner=True)

def load_embnet(model_path, device="cpu"):
    model = EmbNet(dim=256).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(img).unsqueeze(0)

def load_class_db(path):
    if os.path.exists(path):
        return torch.load(path)
    else:
        return {}

def save_class_db(class_db, path):
    torch.save(class_db, path)

# ========== STREAMLIT UI ==========
st.title("Face Registration (Webcam)")

# Load model and class_db (once)
model = load_embnet(MODEL_PATH)
class_db = load_class_db(CLASS_DB_PATH)

# Webcam feed
frame_placeholder = st.empty()
capture_btn = st.button("Capture Face")

name = st.text_input("Enter your name to register")

# Webcam logic
capture = cv2.VideoCapture(0)
ret, frame = capture.read()
if not ret:
    st.warning("Webcam not found!")
    capture.release()
else:
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        if capture_btn:
            img = Image.fromarray(frame_rgb)
            break
    capture.release()

    if capture_btn and name:
        # Preprocess, embed, and register
        image_tensor = preprocess(img)
        emb = model(image_tensor.to(DEVICE)).cpu().flatten()
        class_db[name] = emb
        save_class_db(class_db, CLASS_DB_PATH)
        st.success(f"Registered new user: {name}")

    elif capture_btn and not name:
        st.warning("Please enter your name before registering.")

st.info("Start webcam, enter your name, and click 'Capture Face' to register.")