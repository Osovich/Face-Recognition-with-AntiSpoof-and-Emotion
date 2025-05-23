# scripts/webcam_liveness_demo.py  – fixed version
import cv2, time
from facepipe import Liveness

liv   = Liveness(device="cuda")           # or "cuda" if ORT-GPU is working
THR   = 0.55
FONT  = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not found")

print("Press [q] to quit")
while True:
    ok, frame = cap.read()
    if not ok:
        break

    prob, _ = liv.score(frame)              # ← single float now
    if prob >= THR:
        label, color = f"LIVE  {prob:.2f}", (0,255,0)
    else:
        label, color = f"SPOOF {prob:.2f}", (0,0,255)

    cv2.putText(frame, label, (10,40), FONT, 1.1, color, 2, cv2.LINE_AA)
    cv2.imshow("liveness demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
