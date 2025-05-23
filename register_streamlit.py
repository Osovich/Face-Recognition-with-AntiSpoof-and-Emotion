import streamlit as st
import torch
import torch.nn as nn
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

@st.cache_resource(show_spinner=True)
def load_embnet(model_path, device="cpu"):
    model = EmbNet(dim=256).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
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

model = load_embnet(MODEL_PATH)
class_db = load_class_db(CLASS_DB_PATH)

if 'frame' not in st.session_state:
    st.session_state['frame'] = None
if 'captured_frame' not in st.session_state:
    st.session_state['captured_frame'] = None

name = st.text_input("Enter your name to register")

frame_placeholder = st.empty()
capture_btn = st.button("Capture Face")
register_btn = st.button("Register Face")

# Main LIVE webcam loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.warning("Webcam not found!")
else:
    # Stream webcam until user interrupts
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Can't read from webcam!")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", caption="Webcam Live")
        st.session_state['frame'] = frame  # Always keep the latest frame

        # Stop the loop if user interacts (e.g., presses a Streamlit button)
        if capture_btn or register_btn:
            break
    cap.release()

# If capture button is pressed, store the frame
if capture_btn and st.session_state['frame'] is not None:
    st.session_state['captured_frame'] = st.session_state['frame'].copy()
    st.success("Face captured! Now you can register with your name.")

# Show the captured frame and user prompt
if st.session_state['captured_frame'] is not None:
    st.image(st.session_state['captured_frame'][:, :, ::-1], channels="RGB", caption="Captured Face")

if register_btn:
    if name and st.session_state['captured_frame'] is not None:
        img = Image.fromarray(st.session_state['captured_frame'][:, :, ::-1])
        image_tensor = preprocess(img).to(DEVICE)
        with torch.no_grad():
            emb = model(image_tensor).cpu().flatten()
        class_db[name] = emb
        try:
            save_class_db(class_db, CLASS_DB_PATH)
            st.success(f"Registered new user: {name}")
        except Exception as e:
            st.error(f"Failed to save database: {e}")
    elif not name:
        st.warning("Please enter your name before registering.")
    elif st.session_state['captured_frame'] is None:
        st.warning("Please capture a face before registering.")

# Sidebar: show registered users
with st.sidebar:
    st.header("Registered Users")
    if class_db:
        for user in class_db.keys():
            st.write(user)
        st.write(f"Total: {len(class_db)} users")
    else:
        st.write("No users registered yet.")