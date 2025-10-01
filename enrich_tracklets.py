import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import time
import pickle
import gc
import torch
from collections import defaultdict, Counter
import random

# Giả sử bạn đã có file utils.py từ trước
from utils import (
    get_clip_text_features,
    get_clip_image_features,
    apply_mask,
    cosine_similarity,
)
from segment_anything import sam_model_registry, SamPredictor
from transformers import CLIPProcessor, CLIPModel

# --- 1. CONFIGURATION ---

# --- Đường dẫn ---
RAW_TRACKLETS_DB_PATH = Path('./raw_tracklets.pkl')
ENRICHED_TRACKLETS_DB_PATH = Path('./enriched_tracklets.feather')
VIDEO_SOURCE_DIR = Path('./Video_vong_loai/')
MODELS_DIR = Path('./models/')

# --- Cấu hình Model ---
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"

# --- Cấu hình Xử lý & Thẩm định ---
NUM_KEYFRAMES_PER_TRACKLET = 7
CLIP_SIMILARITY_THRESHOLD = 0.25
SAM_IOU_THRESHOLD = 0.88

# Các lớp và text prompts
TARGET_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
}
CLIP_PROMPTS = {name: f"a photo of a {name}" for name in TARGET_CLASSES.values()}

# --- 2. HELPER FUNCTIONS ---

video_path_cache = {}
def find_video_path(video_name: str) -> Path | None:
    if video_name in video_path_cache: return video_path_cache[video_name]
    try:
        path = next(VIDEO_SOURCE_DIR.rglob(f"**/{video_name}"))
        video_path_cache[video_name] = path
        return path
    except StopIteration: return None

def sample_keyframes_from_tracklet(tracklet: dict, num_samples: int) -> list:
    frames, bboxes, confs = tracklet['frames'], tracklet['bboxes'], tracklet['confidences']
    if len(frames) <= num_samples: return list(zip(frames, bboxes))
    max_conf_idx = np.argmax(confs)
    indices = {0, len(frames) - 1, max_conf_idx}
    rem_indices = list(set(range(len(frames))) - indices)
    num_random = num_samples - len(indices)
    if num_random > 0 and rem_indices:
        indices.update(random.sample(rem_indices, min(num_random, len(rem_indices))))
    return [(frames[i], bboxes[i]) for i in sorted(list(indices))]

# --- 3. MAIN SCRIPT ---

def enrich_tracklets_batch_processing():
    start_time = time.time()
    print("--- GĐ 2 (TỐI ƯU BATCH): BẮT ĐẦU LÀM GIÀU TRACKLETS ---")

    # --- Tải Dữ liệu và Model ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Đang tải model CLIP và SAM...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    sam_checkpoint_path = MODELS_DIR / SAM_CHECKPOINT_NAME
    if not sam_checkpoint_path.exists(): raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint_path}")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=sam_checkpoint_path).to(device)
    sam_predictor = SamPredictor(sam)

    print("Chuẩn bị text features cho CLIP...")
    text_features_map = {
        name: get_clip_text_features([prompt], clip_model, clip_processor, device)[0]
        for name, prompt in CLIP_PROMPTS.items()
    }

    if not RAW_TRACKLETS_DB_PATH.exists(): raise FileNotFoundError(f"Raw tracklets DB not found: {RAW_TRACKLETS_DB_PATH}")
    print("Đang tải CSDL tracklet thô...")
    with open(RAW_TRACKLETS_DB_PATH, 'rb') as f:
        raw_tracklets = pickle.load(f)
    print(f"Tải thành công {len(raw_tracklets)} tracklets.")
    
    # --- BƯỚC 1: GOM TẤT CẢ CÁC CÔNG VIỆC LẠI ---
    print("Bước 1: Gom và sắp xếp tất cả các keyframe cần xử lý...")
    tasks_by_video = defaultdict(list)
    for tracklet in raw_tracklets:
        keyframes = sample_keyframes_from_tracklet(tracklet, NUM_KEYFRAMES_PER_TRACKLET)
        for frame_id, bbox in keyframes:
            tasks_by_video[tracklet['video_name']].append({
                'track_id': tracklet['track_id'],
                'frame_id': frame_id,
                'bbox': bbox
            })

    # --- BƯỚC 2: XỬ LÝ THEO TỪNG VIDEO ---
    validation_results = defaultdict(list) # {track_id: [class_name_1, class_name_2, ...]}

    for video_name, tasks in tqdm(tasks_by_video.items(), desc="Processing Videos in Batches"):
        video_path = find_video_path(video_name)
        if not video_path: continue
        
        sorted_tasks = sorted(tasks, key=lambda x: x['frame_id'])
        
        cap = cv2.VideoCapture(str(video_path))
        task_pointer = 0
        frame_id = 0
        
        while cap.isOpened() and task_pointer < len(sorted_tasks):
            ret, frame = cap.read()
            if not ret: break
            
            target_frame_id = sorted_tasks[task_pointer]['frame_id']
            if frame_id == target_frame_id:
                tasks_in_batch = []
                while task_pointer < len(sorted_tasks) and sorted_tasks[task_pointer]['frame_id'] == frame_id:
                    tasks_in_batch.append(sorted_tasks[task_pointer])
                    task_pointer += 1
                
                # --- BƯỚC 3: XỬ LÝ AI THEO BATCH ---
                if tasks_in_batch:
                    sam_predictor.set_image(frame)
                    input_boxes = torch.tensor([t['bbox'] for t in tasks_in_batch], device=sam_predictor.device)
                    transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
                    
                    with torch.no_grad():
                        masks, scores, _ = sam_predictor.predict_torch(
                            point_coords=None, point_labels=None,
                            boxes=transformed_boxes, multimask_output=False
                        )
                    
                    valid_masks, valid_tasks = [], []
                    for i, score in enumerate(scores):
                        if score >= SAM_IOU_THRESHOLD:
                            valid_masks.append(masks[i])
                            valid_tasks.append(tasks_in_batch[i])

                    if valid_tasks:
                        cropped_images = [apply_mask(frame, m.cpu().numpy()) for m in valid_masks]
                        img_feats = get_clip_image_features(cropped_images, clip_model, clip_processor, device)
                        
                        if img_feats.shape[0] == len(valid_tasks):
                            for i, task in enumerate(valid_tasks):
                                best_class, max_sim = None, -1
                                for class_name, text_feat in text_features_map.items():
                                    sim = cosine_similarity(img_feats[i], text_feat.numpy())
                                    if sim > max_sim:
                                        max_sim, best_class = sim, class_name
                                if max_sim > CLIP_SIMILARITY_THRESHOLD:
                                    validation_results[task['track_id']].append(best_class)
            frame_id += 1
        cap.release()
        gc.collect()

    # --- BƯỚC 4: TỔNG HỢP KẾT QUẢ CUỐI CÙNG ---
    print("\nBước 4: Tổng hợp và tạo CSDL đã làm giàu...")
    enriched_tracklets = []
    for tracklet in raw_tracklets:
        track_id = tracklet['track_id']
        if track_id in validation_results:
            results = validation_results[track_id]
            if not results: continue
            
            final_class_name = Counter(results).most_common(1)[0][0]
            
            enriched_tracklet = tracklet.copy()
            enriched_tracklet['final_class_name'] = final_class_name
            enriched_tracklet['validation_count'] = len(results)
            enriched_tracklets.append(enriched_tracklet)
            
    # --- Lưu kết quả ---
    print(f"\n--- Hoàn thành. Tổng cộng có {len(enriched_tracklets)} tracklets đã được xác thực. ---")
    df = pd.DataFrame(enriched_tracklets)
    if not df.empty:
        print("Phân bố các lớp đối tượng cuối cùng:")
        print(df['final_class_name'].value_counts())
    df.reset_index(drop=True).to_feather(ENRICHED_TRACKLETS_DB_PATH)
    print(f"Lưu CSDL đã làm giàu thành công tại: {ENRICHED_TRACKLETS_DB_PATH}")
    
    end_time = time.time()
    print(f"\n--- GIAI ĐOẠN 2 (TỐI ƯU BATCH) HOÀN TẤT. Tổng thời gian: {(end_time - start_time) / 60:.2f} phút. ---")


if __name__ == "__main__":
    enrich_tracklets_batch_processing()