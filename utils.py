import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamPredictor

def load_all_models(yolo_name, clip_name, sam_type, sam_checkpoint_path, device):
    """Tải tất cả các model cần thiết và đưa lên device."""
    print("--- Đang tải các model AI ---")
    
    # Tải YOLO
    print(f"YOLO: {yolo_name}")
    yolo_model = YOLO(yolo_name).to(device)
    
    # Tải CLIP
    print(f"CLIP: {clip_name}")
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name)
    
    # Tải SAM
    print(f"SAM: {sam_type}")
    if not sam_checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy SAM checkpoint tại: {sam_checkpoint_path}")
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint_path).to(device)
    sam_predictor = SamPredictor(sam)
    
    print("--- Tải tất cả model thành công. ---")
    return yolo_model, clip_model, clip_processor, sam_predictor

def get_clip_text_features(text_list, clip_model, clip_processor, device):
    """Tạo vector đặc trưng cho một danh sách văn bản."""
    with torch.no_grad():
        inputs = clip_processor(
            text=text_list, return_tensors="pt", padding=True
        ).to(device)
        text_features = clip_model.get_text_features(**inputs)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu() # Giữ trên CPU để dễ dàng gán

def get_clip_image_features(image_list, clip_model, clip_processor, device, batch_size=64):
    """Tạo vector đặc trưng cho một danh sách hình ảnh theo batch."""
    all_features = []
    with torch.no_grad():
        for i in range(0, len(image_list), batch_size):
            batch = image_list[i : i + batch_size]
            # Xử lý trường hợp ảnh trống
            valid_images = [img for img in batch if img is not None and img.size > 0]
            if not valid_images: continue
            
            inputs = clip_processor(
                images=valid_images, return_tensors="pt", padding=True
            ).to(device)
            image_features = clip_model.get_image_features(**inputs)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            all_features.append(image_features.cpu())

    if not all_features:
        return np.array([])
    return torch.cat(all_features).numpy()


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Áp dụng mask nhị phân lên ảnh, giữ lại vùng object và làm đen phần nền."""
    masked_image = np.zeros_like(image)
    # Mask từ SAM là (H, W), cần mở rộng thành (H, W, 3)
    masked_image[mask] = image[mask]
    return masked_image

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Tính độ tương đồng cosine giữa hai vector."""
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def calculate_iou(boxA, boxB):
    """
    Tính Intersection over Union (IoU) giữa hai bounding box.
    Định dạng box: [x1, y1, x2, y2]
    """
    # Xác định tọa độ của vùng giao nhau
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Tính diện tích vùng giao nhau
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    # Tính diện tích của mỗi box
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Tính IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Tính độ tương đồng cosine giữa hai vector NumPy.
    Đảm bảo vector đã được chuẩn hóa (norm L2 = 1).
    """
    # Vì các vector feature của CLIP đã được chuẩn hóa,
    # phép nhân vô hướng (dot product) chính là cosine similarity.
    sim = np.dot(vec_a, vec_b)
    # Giới hạn giá trị trong khoảng [-1, 1] để tránh lỗi làm tròn số
    return np.clip(sim, -1.0, 1.0)