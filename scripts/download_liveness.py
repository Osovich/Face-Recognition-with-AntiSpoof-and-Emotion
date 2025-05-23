from facepipe import Liveness

# first construction triggers insightface to download MiniFASNet
print("Downloading / initialising MiniFASNet anti-spoof model …")
Liveness(device="cpu")          # or "cuda" if ONNX-GPU works for you
print("✅  MiniFASNet ready (stored under models/)")
