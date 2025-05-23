import torch
from torchvision import transforms
from PIL import Image
from face_recognition.recognition_utils import recognize_face

# Load your model (classifier_feat_model) and class_db here...

def preprocess(img_path):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)  # Shape: (1, 3, 112, 112)

if __name__ == "__main__":
    img_path = "path/to/your/image.jpg"
    image_tensor = preprocess(img_path)
    identity, similarity = recognize_face(image_tensor, classifier_feat_model, class_db, threshold=0.7, device="cuda")
    if identity:
        print(f"Recognized as: {identity} (similarity: {similarity:.2f})")
    else:
        print("Unknown face")