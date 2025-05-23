import streamlit as st
import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import os
from face_recognition.embnet import EmbNet
from face_recognition.recognition_utils import recognize_face
import torch.nn.functional as F

# ========== CONFIG ==========
MODEL_PATH = "models/embnet_triplet_15k.pth"
CLASS_DB_PATH = "models/class_db.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ========== MODEL & UTILS ==========
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
st.title("Face Registration & Recognition (Webcam)")

model = load_embnet(MODEL_PATH, device=DEVICE)
class_db = load_class_db(CLASS_DB_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

frame_placeholder = st.empty()
capture_btn = st.button("Capture Face")
name = st.text_input("Enter your name to register")

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    st.warning("Webcam not found!")
else:
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(frame_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        # Recognize and annotate faces
        for (x, y, w, h) in faces:
            face_img = frame_rgb[y:y+h, x:x+w]
            pil_face = Image.fromarray(face_img)
            image_tensor = preprocess(pil_face)
            user, sim = recognize_face(image_tensor, model, class_db, threshold=0.7, device=DEVICE)
            label = user if user else "Unknown"
            # Draw rectangle and name
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame_rgb, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        frame_placeholder.image(frame_rgb, channels="RGB", caption="Webcam Live (Recognition)")

        if capture_btn:
            # For registration: use the biggest detected face (if any)
            if len(faces) > 0 and name:
                x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                pil_face = Image.fromarray(frame_rgb[y:y+h, x:x+w])
                image_tensor = preprocess(pil_face)
                emb = model(image_tensor.to(DEVICE)).cpu().flatten()
                emb = F.normalize(emb, dim=0)  # Consistent normalization with recognition_utils
                class_db[name] = emb
                save_class_db(class_db, CLASS_DB_PATH)
                st.success(f"Registered new user: {name}")
            elif len(faces) == 0:
                st.warning("No face detected for registration.")
            elif not name:
                st.warning("Please enter your name before registering.")
            break  # End webcam loop after registration

    capture.release()

st.info("Live recognition: your name will appear above your face if registered.")

# Registered users in sidebar
with st.sidebar:
    st.header("Registered Users")
    if class_db:
        for user in class_db.keys():
            st.write(user)
        st.write(f"Total: {len(class_db)} users")
    else:
        st.write("No users registered yet.")