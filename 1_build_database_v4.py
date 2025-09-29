
import torch
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import time
import gc

# Import các hàm tiện ích từ file utils.py
from utils import (
    load_all_models,
    get_clip_text_features,
    get_clip_image_features,
    apply_mask,
    cosine_similarity,
)

# --- 1. CONFIGURATION ---
# --- Đường dẫn ---
VIDEO_SOURCE_DIR = Path('./Video_vong_loai/')
MODELS_DIR = Path('./models/')
OUTPUT_DB_PATH = Path('./evidence_database_v4_high_quality.feather')

# --- Cấu hình Model ---
YOLO_MODEL_NAME = 'yolov8x.pt'
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"

# --- Cấu hình Xử lý & Thẩm định ---
RAW_CONFIDENCE_THRESHOLD = 0.1
HIGH_CONF_THRESHOLD = 0.6
GRAY_ZONE_UPPER_BOUND = 0.6 # Mở rộng vùng xám

# Ngưỡng của Hội đồng Thẩm định
SAM_IOU_THRESHOLD = 0.90
CLIP_SIMILARITY_THRESHOLD = 0.22 

# Các lớp quan tâm và text prompts tương ứng
TARGET_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
}
TARGET_CLASS_IDS = list(TARGET_CLASSES.keys())
CLIP_PROMPTS = {name: f"a photo of a {name}" for name in TARGET_CLASSES.values()}

# --- 2. MAIN LOGIC ---

def build_high_quality_database():
    start_time = time.time()
    print("--- BẮT ĐẦU XÂY DỰNG CSDL V4 (Hội đồng Thẩm định) ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    yolo, clip, clip_proc, sam_pred = load_all_models(
        YOLO_MODEL_NAME, CLIP_MODEL_NAME, SAM_MODEL_TYPE, 
        MODELS_DIR / SAM_CHECKPOINT_NAME, device
    )

    print("Chuẩn bị text features cho CLIP...")
    text_features_map = {
        name: get_clip_text_features([prompt], clip, clip_proc, device)[0]
        for name, prompt in CLIP_PROMPTS.items()
    }
    
    video_files = list(VIDEO_SOURCE_DIR.glob('*.mp4'))
    all_final_detections = []

    for video_path in video_files:
        video_name = video_path.name
        print(f"\n--- Đang xử lý video: {video_name} ---")
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc="  -> Frames") as pbar:
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # --- Thu thập và Phân loại ---
                results = yolo.predict(
                    frame, conf=RAW_CONFIDENCE_THRESHOLD, classes=TARGET_CLASS_IDS,
                    device=device, verbose=False
                )
                
                high_conf_dets = []
                gray_zone_candidates = []

                for res in results:
                    for box in res.boxes:
                        conf = float(box.conf[0])
                        det_info = {
                            'class_name': yolo.names[int(box.cls[0])],
                            'confidence': conf,
                            'bbox': [int(c) for c in box.xyxy[0].tolist()]
                        }
                        if conf >= HIGH_CONF_THRESHOLD:
                            high_conf_dets.append(det_info)
                        else: # Mọi thứ dưới high_conf đều là ứng viên
                            gray_zone_candidates.append(det_info)
                
                final_detections_for_frame = []
                
                # --- Xử lý Nhóm Tin cậy cao ---
                for det in high_conf_dets:
                    det['validation_method'] = 'yolo_high_conf'
                    final_detections_for_frame.append(det)

                # --- "Hội đồng Thẩm định" xử lý Vùng xám ---
                if gray_zone_candidates:
                    sam_pred.set_image(frame)
                    
                    for candidate in gray_zone_candidates:
                        input_box = np.array(candidate['bbox'])
                        masks, scores, _ = sam_pred.predict(box=input_box, multimask_output=False)
                        
                        # Vòng 1: SAM
                        if scores[0] < SAM_IOU_THRESHOLD: continue
                        
                        # Vòng 2: CLIP
                        cropped_by_mask = apply_mask(frame, masks[0])
                        if cropped_by_mask.size == 0: continue
                        
                        img_feat = get_clip_image_features([cropped_by_mask], clip, clip_proc, device)
                        if img_feat.size == 0: continue
                        
                        text_feat = text_features_map[candidate['class_name']]
                        similarity = cosine_similarity(img_feat[0], text_feat.numpy())
                        
                        if similarity < CLIP_SIMILARITY_THRESHOLD: continue
                        
                        # Thẩm định thành công!
                        candidate['validation_method'] = 'sam_clip_validated'
                        candidate['sam_iou_score'] = scores[0]
                        candidate['clip_similarity'] = similarity
                        final_detections_for_frame.append(candidate)
                
                # --- Trích xuất Feature CLIP cho tất cả các phát hiện hợp lệ ---
                if final_detections_for_frame:
                    cropped_images = [frame[d['bbox'][1]:d['bbox'][3], d['bbox'][0]:d['bbox'][2]] for d in final_detections_for_frame]
                    clip_features = get_clip_image_features(cropped_images, clip, clip_proc, device)

                    if len(clip_features) == len(final_detections_for_frame):
                        for i, det in enumerate(final_detections_for_frame):
                            det['clip_feature'] = clip_features[i]
                            det['video_name'] = video_name
                            det['frame_id'] = frame_id
                        all_final_detections.extend(final_detections_for_frame)

                frame_id += 1
                pbar.update(1)
            
            cap.release()
        
        gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()

    # --- Tạo DataFrame và Lưu ---
    print("\n--- Hoàn thành. Đang tạo DataFrame cuối cùng. ---")
    df = pd.DataFrame(all_final_detections)
    
    print(f"Tạo DataFrame thành công với {len(df):,} phát hiện chất lượng cao.")
    if not df.empty:
        print("Phân bố phương thức xác thực:")
        print(df['validation_method'].value_counts())

    df.reset_index(drop=True).to_feather(OUTPUT_DB_PATH)
    print(f"Lưu CSDL V4 thành công tại: {OUTPUT_DB_PATH}")
    
    end_time = time.time()
    print(f"\n--- GIAI ĐOẠN 1 (V4) HOÀN TẤT. Tổng thời gian: {(end_time - start_time) / 60:.2f} phút. ---")

if __name__ == "__main__":
    build_high_quality_database()