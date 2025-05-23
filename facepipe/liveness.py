# facepipe/liveness.py  (replace previous version)
import cv2, numpy as np, pathlib, torch, onnxruntime as ort

# ------------------------------------------------------------------ paths
_MODELS = pathlib.Path(__file__).parent.parent / "models"
_ONNX   = _MODELS / "model.onnx"                      # anti-spoof-mn3
# ------------------------------------------------------------------ model spec
IMG = 128
MEAN = np.array([151.2405, 119.5950, 107.8395])
STD  = np.array([ 63.0105,  56.4570,  55.0035])

# ---------- face detector (Haar) ------------------------------------
_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _detect_face(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(gray, 1.1, 4,
                                      flags=cv2.CASCADE_SCALE_IMAGE,
                                      minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])   # biggest face
    return bgr[y:y+h, x:x+w]
# ------------------------------------------------------------------
class Liveness:
    def __init__(self, device="cpu"):
        want = ["CUDAExecutionProvider"] if device=="cuda" else ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(_ONNX), providers=want)
        except Exception:
            self.session = ort.InferenceSession(str(_ONNX),
                                               providers=["CPUExecutionProvider"])

    @staticmethod
    def _prep(face):
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG,IMG))
        rgb = (rgb - MEAN)/STD
        return rgb.transpose(2,0,1).astype("float32")[None]

    @torch.no_grad()
    def score(self, bgr):
        face = _detect_face(bgr)
        if face is None:
            return 0.0, None              # no face → treat as spoof
        out = self.session.run(None, {"actual_input_1": self._prep(face)})[0]
        live = float(out[0,1])            # index-1 = LIVE
        return live, face                 # keep API compatible
