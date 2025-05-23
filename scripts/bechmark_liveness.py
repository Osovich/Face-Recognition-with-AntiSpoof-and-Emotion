import glob, cv2, numpy as np, argparse, json, pathlib
from sklearn.metrics import roc_auc_score, roc_curve, auc
from facepipe import Liveness

parser = argparse.ArgumentParser()
parser.add_argument("--live",  default="test/live/*.jpg")
parser.add_argument("--spoof", default="test/spoof/*.jpg")
args = parser.parse_args()

liv    = Liveness("cpu")            # or "cuda"
scores, labels = [], []

for p in glob.glob(args.live):
    s,_ = liv.score(cv2.imread(p))
    scores.append(s); labels.append(1)
for p in glob.glob(args.spoof):
    s,_ = liv.score(cv2.imread(p))
    scores.append(s); labels.append(0)

auc_val = roc_auc_score(labels, scores)
fpr, tpr, thr = roc_curve(labels, scores)

target_fpr = 0.01                              # 1 % false accept
idx        = np.searchsorted(fpr, target_fpr)
best_thr   = thr[idx]

print(f"Liveness AUC = {auc_val:.4f}")
print(f"Threshold @ {target_fpr:.2%} FPR = {best_thr:.3f}")

# save so verify.py can import it
(pathlib.Path(__file__).parent.parent / "config.json").write_text(
    json.dumps({"THR_LIVE": float(best_thr)}, indent=2))
