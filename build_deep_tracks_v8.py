import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment # Thuật toán Hungarian
import time

# Import các hàm tiện ích
from utils import calculate_iou, cosine_similarity

# --- 1. CONFIGURATION ---
DB_PATH = Path('./evidence_database_v5_expert.feather')
OUTPUT_DB_PATH = Path('./deep_tracked_database_v8.feather')

# --- Tham số của Thuật toán Tracking ---
W_SPATIAL = 0.2      # Giảm trọng số không gian
W_APPEARANCE = 0.8   # Tăng trọng số diện mạo
COST_THRESHOLD = 0.5 # Siết chặt ngưỡng chi phí tổng
MAX_AGE = 100        # Số frame tối đa một track có thể bị mất dấu

# --- THAY ĐỔI 1: THÊM SIÊU THAM SỐ MỚI ---
CLASS_MISMATCH_PENALTY = 0.5 # Mức phạt khi gán ghép một track với một detection khác lớp

# --- 2. LỚP CẤU TRÚC DỮ LIỆU TRACK ---
class Track:
    def __init__(self, detection_row, track_id: int):
        self.id = track_id
        # --- THAY ĐỔI 2: SỬA LẠI CÁCH LẤY INDEX (Quan trọng!) ---
        # `detection_row` là một itertuple, index của nó là `detection_row.Index`
        self.detection_indices = {detection_row.Index} 
        
        self.last_update_frame = detection_row.frame_id
        self.last_bbox = detection_row.bbox
        self.age = 0
        self.avg_feature = np.array(detection_row.clip_feature, copy=True)
        self.class_histogram = {detection_row.class_name: 1}

    def update(self, detection_row):
        self.detection_indices.add(detection_row.Index)
        
        self.last_update_frame = detection_row.frame_id
        self.last_bbox = detection_row.bbox
        self.age = 0
        
        alpha = 0.2
        new_feature = np.array(detection_row.clip_feature)
        self.avg_feature = (1 - alpha) * self.avg_feature + alpha * new_feature
        self.avg_feature /= np.linalg.norm(self.avg_feature)
        
        cls_name = detection_row.class_name
        self.class_histogram[cls_name] = self.class_histogram.get(cls_name, 0) + 1
        
    def increment_age(self):
        """Tăng tuổi của track lên 1."""
        self.age += 1

# --- 3. MAIN SCRIPT ---
def build_deep_tracks():
    start_time = time.time()
    print("--- BẮT ĐẦU XÂY DỰNG DEEP TRACKS (V8 - Nâng cấp Phạt Chéo Lớp) ---")

    if not DB_PATH.exists():
        print(f"LỖI: Không tìm thấy CSDL tại '{DB_PATH}'"); return
        
    print(f"Đang tải CSDL từ: {DB_PATH}")
    df = pd.read_feather(DB_PATH)
    
    df['track_id'] = -1
    df['clip_feature'] = df['clip_feature'].apply(lambda x: x.copy())

    df = df.sort_values(['video_name', 'frame_id']).reset_index(drop=True)
    
    next_global_track_id = 0

    for video_name, video_df in df.groupby('video_name'):
        print(f"\n--- Đang xử lý video: {video_name} ---")
        live_tracks = []

        for frame_id, frame_group in tqdm(video_df.groupby('frame_id'), desc=" -> Tracking Frames", leave=False):
            
            for track in live_tracks:
                track.increment_age()
            live_tracks = [track for track in live_tracks if track.age <= MAX_AGE]

            detections = list(frame_group.itertuples())
            
            if not detections: continue
            if not live_tracks:
                for det_row in detections:
                    new_track = Track(det_row, next_global_track_id)
                    live_tracks.append(new_track)
                    next_global_track_id += 1
                continue

            num_tracks = len(live_tracks)
            num_dets = len(detections)
            cost_matrix = np.full((num_tracks, num_dets), 1e5, dtype=np.float32)

            track_features = np.array([track.avg_feature for track in live_tracks])
            det_features = np.array([det.clip_feature for det in detections])
            sim_matrix = cosine_similarity_batch(track_features, det_features)
            
            for i, track in enumerate(live_tracks):
                # Xác định lớp chiếm ưu thế của track dựa trên lịch sử của nó
                dominant_class_in_track = max(track.class_histogram, key=track.class_histogram.get)
                
                for j, det in enumerate(detections):
                    iou = calculate_iou(track.last_bbox, det.bbox)
                    sim = sim_matrix[i, j]
                    
                    cost = W_SPATIAL * (1 - iou) + W_APPEARANCE * (1 - sim)
                    
                    # --- NÂNG CẤP CHÍNH: ÁP DỤNG PHẠT CHÉO LỚP ---
                    if dominant_class_in_track != det.class_name:
                        cost += CLASS_MISMATCH_PENALTY
                    
                    cost_matrix[i, j] = cost

            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            matched_det_indices = set()
            for track_idx, det_idx in zip(track_indices, det_indices):
                cost = cost_matrix[track_idx, det_idx]
                if cost < COST_THRESHOLD:
                    track_to_update = live_tracks[track_idx]
                    detection_to_assign = detections[det_idx]
                    track_to_update.update(detection_to_assign)
                    matched_det_indices.add(det_idx)

            for j, det in enumerate(detections):
                if j not in matched_det_indices:
                    new_track = Track(det, next_global_track_id)
                    live_tracks.append(new_track)
                    next_global_track_id += 1
        
        print(f"  -> Gán {len(live_tracks)} track_id vào DataFrame...")
        all_indices = []
        all_track_ids = []
        for track in live_tracks:
            # Thu thập tất cả index và id để cập nhật một lần, nhanh hơn
            indices_to_update = list(track.detection_indices)
            all_indices.extend(indices_to_update)
            all_track_ids.extend([track.id] * len(indices_to_update))
        
        # Cập nhật DataFrame bằng .loc với danh sách, hiệu quả hơn nhiều
        df.loc[all_indices, 'track_id'] = all_track_ids


    print("\n--- Hoàn thành. Đang lưu CSDL đã được tracking (V8-Upgraded). ---")
    untracked_count = (df['track_id'] == -1).sum()
    print(f"Số phát hiện chưa được gán track (nếu có): {untracked_count}")
    
    final_df = df[df['track_id'] != -1].copy()

    final_df.to_feather(OUTPUT_DB_PATH)
    print(f"Lưu file thành công tại: {OUTPUT_DB_PATH}")

    end_time = time.time()
    print(f"\n--- GIAI ĐOẠN 1.5 (V8-Upgraded) HOÀN TẤT. Tổng thời gian: {(time.time() - start_time):.2f} giây. ---")

if __name__ == "__main__":
    def cosine_similarity_batch(matrix, vectors):
        dot = matrix @ vectors.T
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return dot / (matrix_norms @ vector_norms.T)
        
    build_deep_tracks()