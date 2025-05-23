import torch
import torch.nn.functional as F

def recognize_face(image_tensor, classifier_feat_model, class_db, threshold=0.7, device="cpu"):
    """
    Recognize a face using classifier embeddings.
    - image_tensor: (1, 3, 112, 112) preprocessed image.
    - classifier_feat_model: model to extract features (should output a 1D or 2D tensor).
    - class_db: dict {identity: embedding tensor}
    - threshold: similarity threshold for a match.
    - device: "cuda" or "cpu"
    Returns: (identity, similarity) or (None, max_sim).
    """
    feat = classifier_feat_model(image_tensor.to(device)).cpu().flatten()
    max_sim = -1
    best_id = None
    for identity, emb in class_db.items():
        sim = F.cosine_similarity(feat, emb, dim=0).item()
        if sim > max_sim:
            max_sim = sim
            best_id = identity
    if max_sim > threshold:
        return best_id, max_sim
    else:
        return None, max_sim