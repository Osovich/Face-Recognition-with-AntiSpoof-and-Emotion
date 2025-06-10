"""
Face-Recognition GUI  –  PySide6 + OpenCV + PyTorch + dlib blink anti-spoof
+ FER emotion detection (7 classes)
-----------------------------------------------------------------------------
• Camera locked to MSMF:0   • 60 FPS capture loop
• Blink-gated liveness: unknown must blink ≥1 × to enrol, known users are
  tagged FAKE until they blink.
• Emotion (angry, disgust, fear, happy, sad, surprise, neutral) shown next
  to every face in real time.
"""

import sys, time, threading, cv2, torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from PIL import Image
from scipy.spatial import distance                       # EAR maths :contentReference[oaicite:3]{index=3}
import dlib                                              # 68-landmark predictor
from fer import FER                                      # emotion detector :contentReference[oaicite:4]{index=4}
from torchvision import transforms
from face_recognition.embnet import EmbNet
from PySide6.QtCore    import Qt, QThread, Signal, Slot
from PySide6.QtGui     import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout

# ─── CONFIG ─────────────────────────────────────────────────────────────
MODEL_PATH   = "models/embnet_triplet_hard_50k.pth"
CLASS_DB     = Path("models/class_db.pt")
PREDICTOR    = "models/shape_predictor_68_face_landmarks.dat"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 112
THRESH, MARG = 0.80, 0.05
TARGET_FPS   = 60
CAM_INDEX, CAM_BACKEND = 0, cv2.CAP_MSMF
# Blink params
EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES = 0.20, 1
lStart, lEnd, rStart, rEnd = 42, 48, 36, 42 # left/right eye landmarks
# Emotion colours (BGR)
EMO_COL = dict(
    angry=(0,0,255), disgust=(0,128,128), fear=(128,0,128),
    happy=(0,255,0),  sad=(255,0,0),     surprise=(0,255,255),
    neutral=(200,200,200)
)
torch.backends.cudnn.benchmark = True   # optimize for fixed input size

# ─── preprocessing ──────────────────────────────────────────────────────
_tf = transforms.Compose([
    transforms.Resize(128), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),  transforms.Normalize([0.5]*3, [0.5]*3)
])
pre = lambda pil: _tf(pil).unsqueeze(0)

def load_net(path: str):
    net = EmbNet(dim=512).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    net.load_state_dict(ckpt["model_state"]); net.eval()
    with torch.no_grad():
        net(torch.zeros(1,3,IMG_SIZE,IMG_SIZE, device=DEVICE))
    return net                                           # eager (no Triton)

def l2_mean(vecs):
    if isinstance(vecs, torch.Tensor):
        # If shape is (N, 512), average over axis 0
        if len(vecs.shape) == 2:
            return F.normalize(vecs.mean(0), dim=0)
        return F.normalize(vecs, dim=0)
    elif isinstance(vecs, list):
        return F.normalize(torch.stack(vecs).mean(0), dim=0)
    else:
        raise ValueError("Unknown embedding format")

def ear(eye):   
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ─── worker thread ──────────────────────────────────────────────────────
class CamThread(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        # camera
        self.cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        if not self.cap.isOpened() or not self.cap.read()[0]:
            raise SystemExit("❌ Camera MSMF:0 not available.")
        # models
        self.net  = load_net(MODEL_PATH)
        self.casc = cv2.CascadeClassifier(CASCADE_PATH) # Haar cascade
        self.detector = dlib.get_frontal_face_detector()    # dlib detector
        self.predict  = dlib.shape_predictor(PREDICTOR) # 68-landmark predictor
        self.emdet    = FER(mtcnn=False)                # fast CNN :contentReference[oaicite:5]{index=5}
        # DB
        self.db = torch.load(CLASS_DB) if CLASS_DB.exists() else {}
        self.cent = {n: l2_mean(v) for n, v in self.db.items()} # centroids
        # state
        self.eye_counter, self.blinked = 0, False   # blink state
        self.last = None    # last captured frame (for registration)

    def run(self):
        dt = 1/TARGET_FPS
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                time.sleep(dt); continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.last = rgb
            self.frame_ready.emit(self.annotate(rgb))
            time.sleep(dt)

    # ── annotate frame ────────────────────────────────────────────────
    def annotate(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rects = self.detector(gray, 0)
        for rect in rects:
            shape = self.predict(gray, rect)
            pts   = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            leftEAR  = ear(pts[lStart:lEnd]); rightEAR = ear(pts[rStart:rEnd])
            if (leftEAR + rightEAR)/2 < EYE_AR_THRESH:
                self.eye_counter += 1
            else:
                if self.eye_counter >= EYE_AR_CONSEC_FRAMES:
                    self.blinked = True
                self.eye_counter = 0

        faces = self.casc.detectMultiScale(img, 1.1, 5, minSize=(60,60))
        for (x,y,w,h) in faces:
            crop = img[y:y+h, x:x+w]
            with torch.cuda.amp.autocast():
                emb = self.net(pre(Image.fromarray(crop)).to(DEVICE)).cpu().flatten()
            emb = F.normalize(emb, dim=0)

            best, s1, s2 = None, -1, -1
            for name, cent in self.cent.items():
                emb = emb.flatten()
                cent = cent.flatten()
                s = F.cosine_similarity(emb, cent, dim=0).item()
                if s > s1: s2, s1, best = s1, s, name
                elif s > s2: s2 = s

            # emotion detection (CPU)
            emo, prob = self.emdet.top_emotion(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)) or ("neutral", 0)
            if emo is None or prob is None:                     # <-- new guard
                emo, prob = "neutral", 0
            emo_txt = f"{emo} {prob:.0%}"
            colour  = EMO_COL.get(emo, (255, 255, 255))

            # decide final label
            if s1>THRESH and s1 - s2 >= MARG:     # known
                ident = best if self.blinked else "FAKE"
            else:
                ident = "Unknown" if self.blinked else "Blink…"

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, ident, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.putText(img, emo_txt, (x, y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        return img

    # ── register face ────────────────────────────────────────────────
    def register_face(self, name):
        if not self.blinked:
            return False, "Blink first!"
        if self.last is None:
            return False, "Camera not ready"
        faces = self.casc.detectMultiScale(
            cv2.cvtColor(self.last, cv2.COLOR_RGB2BGR), 1.1, 5, minSize=(60,60))
        if not len(faces):
            return False, "No face detected"
        x,y,w,h = max(faces, key=lambda f:f[2]*f[3])
        crop = Image.fromarray(self.last[y:y+h, x:x+w])
        with torch.cuda.amp.autocast():
            emb = self.net(pre(crop).to(DEVICE)).cpu().flatten()
        emb = F.normalize(emb, dim=0)

        if name in self.db:
            self.db[name] = torch.cat([self.db[name], emb.unsqueeze(0)], dim=0)
        else:
            self.db[name] = emb.unsqueeze(0)
        self.cent[name] = F.normalize(self.db[name].mean(0), dim=0)
        threading.Thread(target=torch.save, args=(self.db, CLASS_DB), daemon=True).start()
        self.blinked = False
        return True, f"Registered {name}"

    def clear_db(self):
        self.db.clear(); self.cent.clear()
        if CLASS_DB.exists(): CLASS_DB.unlink()
        self.blinked = False

# ─── GUI ───────────────────────────────────────────────────────────────
class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍 Face-Rec + Blink Liveness + Emotion")
        self.label   = QLabel(alignment=Qt.AlignCenter)
        self.name_in = QLineEdit(placeholderText="type a name")
        self.btn_reg = QPushButton("📸  Capture / Register")
        self.btn_clr = QPushButton("🗑️  Clear DB")

        vbox = QVBoxLayout(self); vbox.addWidget(self.label, 1)
        hbox = QHBoxLayout(); hbox.addWidget(self.name_in)
        hbox.addWidget(self.btn_reg); hbox.addWidget(self.btn_clr)
        vbox.addLayout(hbox)

        self.worker = CamThread()
        self.worker.frame_ready.connect(self.update_img)
        self.worker.start()

        self.btn_reg.clicked.connect(self.do_register)
        self.btn_clr.clicked.connect(self.do_clear)

    @Slot(np.ndarray)
    def update_img(self, rgb):
        h,w,ch = rgb.shape
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def do_register(self):
        ok, msg = self.worker.register_face(self.name_in.text().strip())
        self.setWindowTitle(msg)

    def do_clear(self):
        self.worker.clear_db()
        self.setWindowTitle("Database cleared")

# ─── main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Main(); w.resize(900, 650); w.show()
    sys.exit(app.exec())
