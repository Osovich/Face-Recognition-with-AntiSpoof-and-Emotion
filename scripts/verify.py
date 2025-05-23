import json, cv2, pathlib
from facepipe import Liveness, Embedder
import faiss, torch
from PIL import Image

CFG = json.loads(open("config.json").read())     # has THR_LIVE
THR_LIVE = CFG["THR_LIVE"]
THR_SIM  = 0.36                                  # as before

MODELS = pathlib.Path(__file__).parent.parent / "models"
index   = faiss.read_index(str(MODELS / "gallery.index"))
id_map  = json.load(open(MODELS / "id_lookup.json"))

liv  = Liveness("cuda")
embd = Embedder("cuda")

def verify(image_path):
    bgr = cv2.imread(image_path)
    score, crop = liv.score(bgr)
    if score is None:
        return "No face detected"
    if score < THR_LIVE:
        return f"Spoof (live score {score:.2f})"

    vec = embd.embed(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).numpy()
    sim, idx = faiss.IndexFlatIP.search(index, vec[None], k=1)
    if sim[0][0] >= THR_SIM:
        return f"Welcome {id_map[idx[0][0]]} (sim {sim[0][0]:.2f})"
    else:
        return "Unknown person"
