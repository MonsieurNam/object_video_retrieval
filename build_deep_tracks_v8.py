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
W_SPATIAL = 0.3      # Trọng số cho chi phí vị trí (1 - IoU)
W_APPEARANCE = 0.7   # Trọng số cho chi phí diện mạo (1 - Cosine Sim)
COST_THRESHOLD = 0.6 # Ngưỡng chi phí tối đa để coi là một cặp khớp
MAX_AGE = 15         # Số frame tối đa một track có thể bị mất dấu trước khi bị xóa

# --- 2. LỚP CẤU TRÚC DỮ LIỆU TRACK ---
class Track:
    """Lớp để lưu trữ và quản lý thông tin của một đường đi (track)."""
    def __init__(self, detection_row, track_id: int):
        self.id = track_id
        self.detection_indices = {detection_row.name} # Dùng set để lưu index
        self.last_update_frame = detection_row['frame_id']
        self.last_bbox = detection_row['bbox']
        self.age = 0
        self.avg_feature = detection_row['clip_feature'].copy() # Tạo bản sao
        self.class_histogram = {detection_row['class_name']: 1}

    def update(self, detection_row):
        """Cập nhật track với một phát hiện mới."""
        self.detection_indices.add(detection_row.name)
        self.last_update_frame = detection_row['frame_id']
        self.last_bbox = detection_row['bbox']
        self.age = 0
        
        alpha = 0.2 # Trọng số cho feature mới
        self.avg_feature = (1 - alpha) * self.avg_feature + alpha * detection_row['clip_feature']
        self.avg_feature /= np.linalg.norm(self.avg_feature)
        
        cls_name = detection_row['class_name']
        self.class_histogram[cls_name] = self.class_histogram.get(cls_name, 0) + 1

    def increment_age(self):
        self.age += 1

# --- 3. MAIN SCRIPT ---
def build_deep_tracks():
    start_time = time.time()
    print("--- BẮT ĐẦU XÂY DỰNG DEEP TRACKS (V8) ---")

    if not DB_PATH.exists():
        print(f"LỖI: Không tìm thấy CSDL tại '{DB_PATH}'"); return
        
    print(f"Đang tải CSDL từ: {DB_PATH}")
    df = pd.read_feather(DB_PATH)
    
    # Chuẩn bị cột mới và đảm bảo vector là writable
    df['track_id'] = -1
    df['clip_feature'] = df['clip_feature'].apply(lambda x: x.copy())

    df = df.sort_values(['video_name', 'frame_id']).reset_index(drop=True)
    
    next_global_track_id = 0

    # --- Vòng lặp chính: Xử lý từng video ---
    for video_name, video_df in df.groupby('video_name'):
        print(f"\n--- Đang xử lý video: {video_name} ---")
        live_tracks = [] # Danh sách các đối tượng Track đang hoạt động

        for frame_id, frame_group in tqdm(video_df.groupby('frame_id'), desc=" -> Tracking Frames", leave=False):
            
            # --- Tăng tuổi và xóa các track cũ ---
            for track in live_tracks:
                track.increment_age()
            live_tracks = [track for track in live_tracks if track.age <= MAX_AGE]

            detections = list(frame_group.itertuples()) # itertuples nhanh hơn iterrows
            
            if not detections: continue
            if not live_tracks:
                # Nếu không có track nào, tạo track mới cho tất cả phát hiện
                for det_row in detections:
                    new_track = Track(det_row, next_global_track_id)
                    live_tracks.append(new_track)
                    next_global_track_id += 1
                continue

            # --- Xây dựng Ma trận Chi phí ---
            num_tracks = len(live_tracks)
            num_dets = len(detections)
            cost_matrix = np.full((num_tracks, num_dets), 1e5, dtype=np.float32)

            track_features = np.array([track.avg_feature for track in live_tracks])
            det_features = np.array([det.clip_feature for det in detections])
            # Tính cosine similarity cho toàn bộ batch
            sim_matrix = cosine_similarity_batch(track_features, det_features)
            
            for i, track in enumerate(live_tracks):
                for j, det in enumerate(detections):
                    iou = calculate_iou(track.last_bbox, det.bbox)
                    sim = sim_matrix[i, j]
                    
                    # Chi phí càng thấp càng tốt
                    cost = W_SPATIAL * (1 - iou) + W_APPEARANCE * (1 - sim)
                    cost_matrix[i, j] = cost

            # --- Giải Bài toán Gán ghép ---
            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            # --- Cập nhật, Tạo mới, và Quản lý ---
            matched_det_indices = set()
            for track_idx, det_idx in zip(track_indices, det_indices):
                cost = cost_matrix[track_idx, det_idx]
                if cost < COST_THRESHOLD:
                    track_to_update = live_tracks[track_idx]
                    detection_to_assign = detections[det_idx]
                    track_to_update.update(detection_to_assign)
                    matched_det_indices.add(det_idx)

            # Tạo track mới cho các phát hiện không được gán ghép
            for j, det in enumerate(detections):
                if j not in matched_det_indices:
                    new_track = Track(det, next_global_track_id)
                    live_tracks.append(new_track)
                    next_global_track_id += 1
        
        # --- Gán track_id vào DataFrame sau khi xử lý xong video ---
        print(f"  -> Gán {len(live_tracks)} track_id vào DataFrame...")
        for track in live_tracks:
            # Lấy các index đã lưu trong track
            indices_to_update = list(track.detection_indices)
            df.loc[indices_to_update, 'track_id'] = track.id

    # --- Lưu Kết quả ---
    print("\n--- Hoàn thành. Đang lưu CSDL đã được tracking (V8). ---")
    untracked_count = (df['track_id'] == -1).sum()
    print(f"Số phát hiện chưa được gán track (nếu có): {untracked_count}")
    
    # Xóa các dòng không được gán track (thường là do lỗi, không nên có nhiều)
    final_df = df[df['track_id'] != -1].copy()

    final_df.to_feather(OUTPUT_DB_PATH)
    print(f"Lưu file thành công tại: {OUTPUT_DB_PATH}")

    end_time = time.time()
    print(f"\n--- GIAI ĐOẠN 1.5 (V8) HOÀN TẤT. Tổng thời gian: {(end_time - start_time):.2f} giây. ---")

if __name__ == "__main__":
    # Cần định nghĩa lại hàm cosine_similarity_batch nếu utils.py không được import
    def cosine_similarity_batch(matrix, vectors):
        dot = matrix @ vectors.T
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return dot / (matrix_norms @ vector_norms.T)
        
    build_deep_tracks()