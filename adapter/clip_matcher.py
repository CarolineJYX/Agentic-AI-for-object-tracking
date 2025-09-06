from typing import List, Tuple, Optional
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Dict, Any
import torch
# adapter/clip_matcher.py
from typing import List, Dict, Any
import numpy as np
import open_clip

def load_clip_model(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load an OpenCLIP model and its preprocess function.
    Returns: (model, preprocess, tokenizer, device)
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.to(device).eval()
    return model, preprocess, tokenizer, device

def clip_to_rate(
    frame: np.ndarray,
    detections: List[Dict],
    prompt: str,
    clip_model,
    preprocess,
    tokenizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    clip_threshold: float = 0.25,
) -> List[Dict]:
    """
    For each YOLO detection, compute CLIP text-image similarity against the user prompt.
    Adds 'clip_score' to each detection and filters out low-scoring ones.

    Args:
        frame: BGR image (H x W x 3)
        detections: list of dicts with at least {'bbox': [x1,y1,x2,y2], ...}
        prompt: natural language description
        clip_model, preprocess, tokenizer: OpenCLIP components
        device: 'cuda' or 'cpu'
        clip_threshold: minimum similarity to keep the detection

    Returns:
        List[Dict]: detections augmented with 'clip_score' (float), filtered by threshold
    """
    if frame is None or frame.size == 0 or not detections:
        return []

    H, W = frame.shape[:2]

    # Encode text once
    with torch.no_grad():
        text_tokens = tokenizer([prompt]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    kept: List[Dict] = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])

        # Clamp to image bounds
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))

        if x2 <= x1 or y2 <= y1:
            det["clip_score"] = 0.0
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            det["clip_score"] = 0.0
            continue

        # Preprocess crop and encode image
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            img_features = clip_model.encode_image(image_input)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            similarity = float((img_features @ text_features.T).item())

        det["clip_score"] = similarity  # add CLIP similarity; do not overwrite YOLO confidence

        if similarity >= clip_threshold:
            kept.append(det)

    return kept

@torch.no_grad()
def clip_to_rate_fast(
    frame: np.ndarray,
    detections: List[Dict],
    text_features: torch.Tensor,   # precomputed and normalized, shape [1, D]
    clip_model,
    preprocess,
    device: str,
    clip_threshold: float = 0.25,
) -> List[Dict]:
    """
    Faster version of clip_to_rate: expects precomputed text_features.
    """
    if frame is None or frame.size == 0 or not detections:
        return []

    H, W = frame.shape[:2]
    kept: List[Dict] = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1)); y2 = max(0, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            det["clip_score"] = 0.0
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            det["clip_score"] = 0.0
            continue

        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image_input = preprocess(image).unsqueeze(0).to(device)
        img_features = clip_model.encode_image(image_input)
        img_features = img_features / (img_features.norm(dim=-1, keepdim=True) + 1e-6)
        similarity = float((img_features @ text_features.T).item())

        det["clip_score"] = similarity
        if similarity >= clip_threshold:
            kept.append(det)

    return kept

@torch.no_grad()
def compute_clip_sim_and_feat_batch(
    frame: np.ndarray,
    bboxes_xyxy: List[List[float]],
    clip_model,
    preprocess,
    text_features: torch.Tensor,   # shape: [1, D], already normalized
    device: str
) -> List[Tuple[float, Optional[torch.Tensor]]]:
    """
    Batched CLIP: encode all boxes from a single frame in one forward pass.
    Returns a list aligned with input bboxes: [(similarity, image_feature or None), ...]
    """
    H, W = frame.shape[:2]
    crops = []
    valid_idx = []

    for i, (x1, y1, x2, y2) in enumerate(bboxes_xyxy):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1)); y2 = max(0, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crops.append(preprocess(image))
        valid_idx.append(i)

    results: List[Tuple[float, Optional[torch.Tensor]]] = [(0.0, None)] * len(bboxes_xyxy)
    if not crops:
        return results

    batch = torch.stack(crops, dim=0).to(device)
    img_feats = clip_model.encode_image(batch)
    img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-6)   # L2 normalize
    sims = (img_feats @ text_features.T).squeeze(1).tolist()

    for k, i in enumerate(valid_idx):
        results[i] = (float(sims[k]), img_feats[k:k+1])  # keep feature as [1, D]

    return results


# Step 3: Load CLIP model and assign similarity scores to ByteTrack-selected objects
def clip_stats(detections: List[Dict[str, Any]], prompt: str, frame=None, 
               model=None, preprocess=None, tokenizer=None, device=None) -> Dict[str, float]:
    """
    直接复用你提供的 CLIP 实现：
      - 若给了 frame：用 CLIP 计算每个框的相似度并写回 det["clip_score"]
      - 若没给 frame：回退到 detections 自带的 confidence/score
    返回 {"avg_conf", "top10_conf", "margin"}
    """
    n = len(detections)
    if n == 0:
        return {"avg_conf": 0.0, "top10_conf": 0.0, "margin": 0.0}

    # 有帧 → 用 CLIP 算相似度
    scores: List[float]
    if frame is not None:
        # ★ 优化：使用预加载的模型
        if model is None or preprocess is None or tokenizer is None or device is None:
            model, preprocess, tokenizer, device = load_clip_model()
        
        with torch.no_grad():
            text_tokens = tokenizer([prompt]).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

        boxes = [d.get("bbox", [0, 0, 0, 0]) for d in detections]
        sims_feats = compute_clip_sim_and_feat_batch(
            frame, boxes, model, preprocess, text_features, device
        )
        sims = [float(sf[0]) for sf in sims_feats]
        feats = [sf[1] for sf in sims_feats]
        for d, s, f in zip(detections, sims, feats):
            d["clip_score"] = s
            d["img_feat"] = f  # ★ 保存CLIP特征用于Global ID分配
        scores = sims
    else:
        # 无帧 → 回退到检测置信度（若已有 clip_score 也会优先用）
        scores = [
            float(d.get("clip_score", d.get("confidence", d.get("score", 0.0))))
            for d in detections
        ]

    arr = np.asarray(scores, dtype=np.float32)
    avg_conf = float(arr.mean())
    k = min(10, len(arr))
    top10_conf = float(np.sort(arr)[-k:].mean()) if k > 0 else 0.0
    rest_mean = float(np.sort(arr)[:-k].mean()) if len(arr) > k else 0.0
    margin = float(top10_conf - rest_mean)

    return {"avg_conf": avg_conf, "top10_conf": top10_conf, "margin": margin}