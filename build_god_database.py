# build_god_database.py
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import time
import gc

# Import các hàm tiện ích
from utils import (
    load_all_models,
    get_clip_text_features,
    get_clip_image_features,
    apply_mask,
    predict_with_adapter,
)

# --- 1. CONFIGURATION ---
VIDEO_SOURCE_DIR = Path('./Video_vong_loai/')
MODELS_DIR = Path('./models/')
OUTPUT_DB_PATH = Path('./god_database_final.parquet') # Lưu dưới dạng Parquet

# Cấu hình Model
# YOLO_MODEL_NAME = 'yolov8x.pt' 
YOLO_MODEL_NAME = 'yolov12n.pt' 

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"

# Cấu hình Xử lý & Thẩm định
RAW_CONFIDENCE_THRESHOLD = 0.08
HIGH_CONF_THRESHOLD = 0.6
SAM_IOU_THRESHOLD = 0.90
CLIP_SIMILARITY_THRESHOLD = 0.22 

# Các lớp quan tâm (Đảm bảo ID khớp với model YOLO của bạn)
TARGET_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
TARGET_CLASS_IDS = list(TARGET_CLASSES.keys())
CLIP_PROMPTS = {name: f"a photo of a {name}" for name in TARGET_CLASSES.values()}

# [CẢI TIẾN] Cấu hình cho việc trích xuất màu sắc
COLOR_PROMPTS = {
    'red': 'a photo of a red object', 'blue': 'a photo of a blue object',
    'white': 'a photo of a white object', 'black': 'a photo of a black object',
    'gray': 'a photo of a gray object', 'yellow': 'a photo of a yellow object',
    'green': 'a photo of a green object', 'silver': 'a photo of a silver object',
}

# Cấu hình cho tracking
IOU_THRESHOLD = 0.55 
MAX_FRAMES_TO_LOSE_TRACK = 250

# --- 2. HÀM TIỆN ÍCH ---

def calculate_iou(boxA, boxB):
    """Tính IoU giữa hai bounding box."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def get_dominant_color(clip_feature_vector: np.ndarray, color_features_tensor: torch.Tensor, color_names: list, device: str) -> str:
    """So sánh feature của ảnh với các feature màu và trả về màu có độ tương đồng cao nhất."""
    if clip_feature_vector is None or clip_feature_vector.size == 0:
        return "unknown"
    image_tensor = torch.from_numpy(clip_feature_vector).to(device).unsqueeze(0)
    similarities = F.cosine_similarity(image_tensor, color_features_tensor)
    best_match_index = torch.argmax(similarities).item()
    return color_names[best_match_index]

# --- 3. LOGIC CHÍNH ---

def build_god_database():
    start_time = time.time()
    print("--- BẮT ĐẦU XÂY DỰNG 'SIÊU CSDL' (GĐ 0) ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")
    
    # --- Tải Models ---
    yolo, clip, clip_proc, sam_pred = load_all_models(YOLO_MODEL_NAME, CLIP_MODEL_NAME, SAM_MODEL_TYPE, 
                                                       MODELS_DIR / SAM_CHECKPOINT_NAME, device, models_dir=MODELS_DIR)
    
    # --- Tiền xử lý các Text Prompts ---
    print("Đang tiền tính toán các vector đặc trưng cho text...")
    text_features_map = {name: get_clip_text_features([prompt], clip, clip_proc, device)[0] 
                         for name, prompt in CLIP_PROMPTS.items()}
    color_names = list(COLOR_PROMPTS.keys())
    color_prompts_list = list(COLOR_PROMPTS.values())
    color_features_tensor = get_clip_text_features(color_prompts_list, clip, clip_proc, device)

    # === GIAI ĐOẠN 1: THU THẬP BẰNG CHỨNG TỪ VIDEO ===
    video_files = sorted(list(VIDEO_SOURCE_DIR.glob('*.mp4')))
    all_detections_raw = []

    for video_path in video_files:
        video_name = video_path.name
        print(f"\n--- [1/5] Đang thu thập bằng chứng từ video: {video_name} ---")
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc="  -> Frames") as pbar:
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # 1. Thu thập và Phân loại bằng YOLO
                # results = yolo.predict(frame, conf=RAW_CONFIDENCE_THRESHOLD, classes=TARGET_CLASS_IDS, device=device, verbose=False)
                all_yolo_dets = predict_with_adapter(yolo, frame, 
                                     conf=RAW_CONFIDENCE_THRESHOLD, 
                                     classes=TARGET_CLASS_IDS, 
                                     device=device, 
                                     verbose=False)

                high_conf_dets = []
                gray_zone_candidates = []

                # Bây giờ, chúng ta chỉ cần lặp qua một list đã được chuẩn hóa
                for det_info in all_yolo_dets:
                    if det_info['confidence'] >= HIGH_CONF_THRESHOLD:
                        high_conf_dets.append(det_info)
                    else:
                        gray_zone_candidates.append(det_info)

                # for res in results:
                #     for box in res.boxes:
                #         conf = float(box.conf[0])
                #         det_info = {
                #             'class_name': yolo.names[int(box.cls[0])],
                #             'confidence': conf,
                #             'bbox': [int(c) for c in box.xyxy[0].tolist()]
                #         }
                #         if conf >= HIGH_CONF_THRESHOLD: high_conf_dets.append(det_info)
                #         else: gray_zone_candidates.append(det_info)

                detections_for_frame = []

                # 2. Xử lý Nhóm Tin cậy cao
                for det in high_conf_dets:
                    det['validation_method'] = 'yolo_high_conf'
                    detections_for_frame.append(det)

                # 3. "Hội đồng Thẩm định" xử lý Vùng xám
                if gray_zone_candidates:
                    sam_pred.set_image(frame)
                    
                    for candidate in gray_zone_candidates:
                        input_box = np.array(candidate['bbox'])
                        masks, scores, _ = sam_pred.predict(box=input_box, multimask_output=False)
                        
                        if scores[0] < SAM_IOU_THRESHOLD: continue
                        
                        cropped_by_mask = apply_mask(frame, masks[0])
                        if cropped_by_mask.size == 0: continue
                        
                        img_feat = get_clip_image_features([cropped_by_mask], clip, clip_proc, device)
                        if img_feat.size == 0: continue
                        
                        text_feat_tensor = text_features_map[candidate['class_name']]
                        similarity = F.cosine_similarity(torch.from_numpy(img_feat), text_feat_tensor).item()
                        
                        if similarity < CLIP_SIMILARITY_THRESHOLD: continue
                        
                        candidate['validation_method'] = 'sam_clip_validated'
                        detections_for_frame.append(candidate)

                # 4. Trích xuất Feature CLIP cho tất cả các phát hiện hợp lệ
                if detections_for_frame:
                    cropped_images = [frame[d['bbox'][1]:d['bbox'][3], d['bbox'][0]:d['bbox'][2]] for d in detections_for_frame]
                    clip_features = get_clip_image_features(cropped_images, clip, clip_proc, device)

                    if len(clip_features) == len(detections_for_frame):
                        for i, det in enumerate(detections_for_frame):
                            det['clip_feature'] = clip_features[i]
                            det['video_name'] = video_name
                            det['frame_id'] = frame_id
                        all_detections_raw.extend(detections_for_frame)

                frame_id += 1
                pbar.update(1)
            
            cap.release()
        
        gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()

    if not all_detections_raw:
        print("\n LỖI: Không có phát hiện nào được tìm thấy. Dừng chương trình.")
        return

    print("\n--- [2/5] Đang tạo DataFrame và làm giàu dữ liệu ---")
    df = pd.DataFrame(all_detections_raw)
    
    # === GIAI ĐOẠN 2: LÀM GIÀU DỮ LIỆU CẤP ĐỐI TƯỢNG ===
    print("  -> Đang tính toán màu sắc chủ đạo cho mỗi đối tượng...")
    df['dominant_color'] = df['clip_feature'].apply(
        lambda x: get_dominant_color(x, color_features_tensor, color_names, device)
    )
    # Xóa cột clip_feature sau khi đã dùng xong để tiết kiệm bộ nhớ
    df.drop(columns=['clip_feature'], inplace=True)
    gc.collect()
    print("  -> Tính toán màu sắc hoàn tất.")

    # === GIAI ĐOẠN 3: XÂY DỰNG TRACKS ===
    print("\n--- [3/5] Đang xây dựng các đường đi (Tracking) ---")
    df['track_id'] = -1
    next_track_id = 0
    df.sort_values(['video_name', 'frame_id'], inplace=True)
    
    for video_name, video_group in tqdm(df.groupby('video_name'), desc="  -> Xử lý video"):
        live_tracks = {}
        
        for frame_id, frame_group in video_group.groupby('frame_id'):
            current_indices = list(frame_group.index)
            matched_indices = set()
            tracks_to_delete = []

            # Bước 1: Cố gắng khớp các phát hiện hiện tại với các track đang hoạt động
            for track_id, last_det_info in live_tracks.items():
                if frame_id - last_det_info['frame_id'] > MAX_FRAMES_TO_LOSE_TRACK:
                    tracks_to_delete.append(track_id)
                    continue

                best_match_idx = -1
                best_iou = IOU_THRESHOLD
                
                for idx in current_indices:
                    if idx in matched_indices: continue
                    iou = calculate_iou(last_det_info['bbox'], df.loc[idx, 'bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = idx
                
                if best_match_idx != -1:
                    df.loc[best_match_idx, 'track_id'] = track_id
                    live_tracks[track_id] = {'frame_id': frame_id, 'bbox': df.loc[best_match_idx, 'bbox']}
                    matched_indices.add(best_match_idx)
            
            for track_id in tracks_to_delete:
                del live_tracks[track_id]

            # Bước 2: Tạo track mới cho các phát hiện chưa được khớp
            unmatched_indices = set(current_indices) - matched_indices
            for idx in unmatched_indices:
                df.loc[idx, 'track_id'] = next_track_id
                live_tracks[next_track_id] = {'frame_id': frame_id, 'bbox': df.loc[idx, 'bbox']}
                next_track_id += 1

    # === GIAI ĐOẠN 4: SUY LUẬN CẤP TRACK & FRAME ===
    print("\n--- [4/5] Đang hoàn thiện CSDL với suy luận cấp track ---")
    
    print("  -> Đang thực hiện 'Bầu cử danh tính'...")
    track_class_map = df.groupby('track_id')['class_name'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'unknown')
    df['consistent_class_name'] = df['track_id'].map(track_class_map)

    print("  -> Đang tiền tính toán số lượng đối tượng mỗi frame...")
    df['class_count_in_frame'] = df.groupby(['video_name', 'frame_id', 'consistent_class_name'])['track_id'].transform('nunique')

    # === GIAI ĐOẠN 5: LƯU KẾT QUẢ ===
    print("\n--- [5/5] HOÀN TẤT LÀM GIÀU DỮ LIỆU ---")
    print(f"Tổng số {len(df):,} bản ghi được tạo ra.")
    print("Ví dụ 5 dòng đầu của 'Siêu CSDL':")
    # Chọn các cột quan trọng để hiển thị
    preview_cols = ['video_name', 'frame_id', 'track_id', 'consistent_class_name', 'dominant_color', 'class_count_in_frame', 'confidence']
    print(df[preview_cols].head())
    
    print(f"\nĐang lưu 'Siêu CSDL' vào file: {OUTPUT_DB_PATH}")
    # Chuyển đổi các cột có thể gây lỗi sang kiểu dữ liệu tương thích
    df['bbox'] = df['bbox'].astype(str)
    df.reset_index(drop=True).to_parquet(OUTPUT_DB_PATH)
    print("Lưu file thành công!")
    
    end_time = time.time()
    print(f"\n--- GIAI ĐOẠN 0 HOÀN TẤT. Tổng thời gian: {(end_time - start_time) / 60:.2f} phút. ---")


if __name__ == "__main__":
    # Đảm bảo bạn đã có các file model trong thư mục ./models/
    # và các file video trong thư mục ./Video_vong_loai/
    build_god_database()