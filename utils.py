import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamPredictor

def predict_with_adapter(yolo_model, frame, **kwargs) -> list:
    """
    Thực hiện dự đoán và chuẩn hóa kết quả về một định dạng chung.
    Hàm này đóng vai trò là lớp tương thích cho các phiên bản YOLO khác nhau.
    
    Returns:
        list[dict]: Một danh sách các dictionaries, mỗi dict chứa:
                    {'class_name': str, 'confidence': float, 'bbox': list[int]}
    """
    # Sử dụng .predict() vì nó an toàn và được hỗ trợ bởi cả hai phiên bản
    results = yolo_model.predict(frame, **kwargs)
    
    normalized_output = []
    if not results:
        return normalized_output
        
    # results là một list, thường chỉ có 1 phần tử cho 1 ảnh
    for res in results:
        if res.boxes is None:
            continue
            
        class_names = res.names
        for box in res.boxes:
            # [QUAN TRỌNG] Xử lý các kiểu dữ liệu khác nhau như bạn đã phân tích
            # Đảm bảo confidence là float và class_id là int
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            
            det_info = {
                'class_name': class_names[class_id],
                'confidence': conf,
                'bbox': [int(c) for c in box.xyxy[0].tolist()]
            }
            normalized_output.append(det_info)
            
    return normalized_output

def load_all_models(
    yolo_name, 
    clip_name, 
    sam_type, 
    sam_checkpoint_path, 
    device, 
    models_dir: Path,
    load_yolo: bool = True,
    load_clip: bool = True,
    load_sam: bool = True
):
    """
    Tải tất cả hoặc một phần các model cần thiết và đưa lên device.
    Phiên bản này hỗ trợ logic tải YOLOv12 thông minh và tải có chọn lọc.
    """
    print("--- Đang tải các model AI ---")
    
    # Khởi tạo các biến trả về
    yolo_model = None
    clip_model = None
    clip_processor = None
    sam_predictor = None

    # [CẢI TIẾN] Bọc logic tải YOLO của bạn trong câu lệnh if
    if load_yolo:
        if 'yolov12' in yolo_name.lower():
            print(f"YOLOv12: Đang tải model từ repo 'sunsmarterjie'...")
            try:
                # Import cục bộ để tránh lỗi nếu repo chưa được clone
                # Giả định rằng bạn đã cài đặt thư viện đúng cách
                yolo_model = YOLO(models_dir / yolo_name).to(device)
                print(" -> Tải YOLOv12 thành công.")
            except Exception as e:
                raise ImportError(f"Lỗi khi tải YOLOv12: {e}. Bạn đã clone và cài đặt repo 'https://github.com/sunsmarterjie/yolov12' chưa?")
        else:
            print(f"YOLO (Ultralytics): {yolo_name}")
            yolo_model = YOLO(models_dir / yolo_name).to(device)
    
    # [CẢI TIẾN] Bọc logic tải CLIP trong câu lệnh if
    if load_clip:
        print(f"CLIP: {clip_name}")
        # Chuyển model lên device trước
        clip_model = CLIPModel.from_pretrained(clip_name).to(device)
        clip_processor = CLIPProcessor.from_pretrained(clip_name)
    
    # [CẢI TIẾN] Bọc logic tải SAM trong câu lệnh if
    if load_sam:
        print(f"SAM: {sam_type}")
        if not sam_checkpoint_path.exists():
            raise FileNotFoundError(f"Không tìm thấy SAM checkpoint tại: {sam_checkpoint_path}")
        sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint_path).to(device)
        sam_predictor = SamPredictor(sam)
    
    print("--- Tải model hoàn tất. ---")
    return yolo_model, clip_model, clip_processor, sam_predictor

def get_clip_text_features(text_list, clip_model, clip_processor, device):
    """Sinh vector đặc trưng văn bản (fix lỗi device mismatch hoàn toàn)."""
    with torch.no_grad():
        # Không .to(device) ở đây
        inputs = clip_processor(text=text_list, return_tensors="pt", padding=True)

        # Bắt buộc: di chuyển từng tensor sang cùng device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        clip_model = clip_model.to(device)
        text_features = clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # Chuyển kết quả về CPU cho an toàn
    return text_features.cpu()



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