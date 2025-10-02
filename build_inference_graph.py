import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment # Hungarian algorithm for optimal matching
from collections import defaultdict

# --- 1. CONFIGURATION ---
DB_PATH = Path('./evidence_database_v5_expert.feather') # Hoặc CSDL V3 nếu muốn nhiều "nút" hơn
OUTPUT_DB_PATH = Path('./inference_graph_database_v7_new.feather')

# --- Tham số Tracking ---
IOU_THRESHOLD = 0.4              # Ngưỡng IoU để xem xét một kết nối
CLIP_SIMILARITY_THRESHOLD = 0.85 # Ngưỡng tương đồng hình ảnh rất cao
MAX_FRAMES_TO_DISAPPEAR = 10     # Số frame tối đa một track có thể "biến mất" trước khi bị xóa
MIN_TRACK_LENGTH = 5             # Một track phải có ít nhất 5 phát hiện mới được giữ lại

# --- 2. HÀM TIỆN ÍCH ---

def calculate_iou(boxA, boxB):
    # ... (Hàm tính IoU giữ nguyên như V6) ...
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def cosine_similarity_batch(matrix_a, matrix_b):
    """Tính cosine similarity giữa hai ma trận các vector."""
    norm_a = np.linalg.norm(matrix_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(matrix_b, axis=1, keepdims=True)
    return (matrix_a @ matrix_b.T) / (norm_a @ norm_b.T)

# --- 3. SCRIPT CHÍNH ---

def build_inference_graph():
    print("--- BẮT ĐẦU XÂY DỰNG VÀ SUY LUẬN TRÊN ĐỒ THỊ (V7) ---")
    df = pd.read_feather(DB_PATH)
    print(f"Đã tải {len(df):,} phát hiện thô.")

    df['track_id'] = -1
    all_tracked_dfs = []
    
    video_groups = df.groupby('video_name')
    
    for video_name, video_df in tqdm(video_groups, desc="Processing Videos"):
        
        live_tracks = {} # { track_id: {'last_det_idx': index, 'age': 0, 'hits': 1} }
        next_track_id = 0
        
        # Sắp xếp theo frame
        video_df = video_df.sort_values('frame_id').reset_index()

        # Tạo ma trận clip features cho toàn bộ video một lần
        all_clip_features = np.vstack(video_df['clip_feature'].values)

        frame_groups = video_df.groupby('frame_id')
        sorted_frames = sorted(frame_groups.groups.keys())

        for i in range(len(sorted_frames)):
            frame_id = sorted_frames[i]
            
            # Lấy các phát hiện của frame hiện tại
            current_det_indices = frame_groups.get_group(frame_id).index
            
            if not live_tracks:
                # Bắt đầu các track mới
                for idx in current_det_indices:
                    video_df.at[idx, 'track_id'] = next_track_id
                    live_tracks[next_track_id] = {'last_det_idx': idx, 'age': 1, 'hits': 1}
                    next_track_id += 1
                continue

            # --- Xây dựng ma trận chi phí để khớp cặp ---
            live_track_ids = list(live_tracks.keys())
            prev_det_indices = [live_tracks[tid]['last_det_idx'] for tid in live_track_ids]
            
            # Tính ma trận IoU và Visual Similarity
            iou_matrix = np.zeros((len(prev_det_indices), len(current_det_indices)), dtype=np.float32)
            sim_matrix = cosine_similarity_batch(
                all_clip_features[prev_det_indices],
                all_clip_features[current_det_indices]
            )

            for r, prev_idx in enumerate(prev_det_indices):
                for c, curr_idx in enumerate(current_det_indices):
                    iou_matrix[r, c] = calculate_iou(video_df.loc[prev_idx, 'bbox'], video_df.loc[curr_idx, 'bbox'])
            
            # Kết hợp hai ma trận thành một ma trận chi phí (cost matrix)
            # Cost = 1 - score. Thuật toán sẽ tìm cách tối thiểu hóa cost.
            cost_matrix = 1 - (0.6 * iou_matrix + 0.4 * sim_matrix) # Trọng số có thể tinh chỉnh

            # --- Sử dụng thuật toán Hungarian để tìm cặp khớp tối ưu ---
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_indices_current = set()
            
            # Xử lý các cặp đã khớp
            for r, c in zip(row_ind, col_ind):
                track_id = live_track_ids[r]
                current_idx = current_det_indices[c]
                
                # Chỉ chấp nhận nếu điểm khớp đủ tốt
                combined_score = 0.6 * iou_matrix[r, c] + 0.4 * sim_matrix[r, c]
                if combined_score > IOU_THRESHOLD:
                    video_df.at[current_idx, 'track_id'] = track_id
                    live_tracks[track_id]['last_det_idx'] = current_idx
                    live_tracks[track_id]['age'] = 0 # Reset age
                    live_tracks[track_id]['hits'] += 1
                    matched_indices_current.add(current_idx)

            # Tăng tuổi cho các track không được khớp
            unmatched_track_ids = set(live_track_ids) - {live_track_ids[r] for r, c in zip(row_ind, col_ind) if (0.6 * iou_matrix[r, c] + 0.4 * sim_matrix[r, c]) > IOU_THRESHOLD}
            for track_id in unmatched_track_ids:
                live_tracks[track_id]['age'] += 1
                if live_tracks[track_id]['age'] > MAX_FRAMES_TO_DISAPPEAR:
                    del live_tracks[track_id]

            # Tạo track mới cho các phát hiện không khớp
            unmatched_indices_current = set(current_det_indices) - matched_indices_current
            for idx in unmatched_indices_current:
                video_df.at[idx, 'track_id'] = next_track_id
                live_tracks[next_track_id] = {'last_det_idx': idx, 'age': 0, 'hits': 1}
                next_track_id += 1
        
        # Lọc bỏ các track quá ngắn
        track_lengths = video_df['track_id'].value_counts()
        valid_tracks = track_lengths[track_lengths >= MIN_TRACK_LENGTH].index
        video_df = video_df[video_df['track_id'].isin(valid_tracks)]
        
        all_tracked_dfs.append(video_df)

    # --- Bước 2b: "Bầu cử" và Sửa lỗi ---
    print("\nBắt đầu quá trình bầu cử và sửa lỗi danh tính...")
    final_df = pd.concat(all_tracked_dfs, ignore_index=True)
    
    # Tìm class_name phổ biến nhất cho mỗi track
    track_class_map = final_df.groupby('track_id')['class_name'].agg(lambda x: x.mode()[0]).to_dict()
    
    # Gán lại class_name nhất quán
    final_df['consistent_class_name'] = final_df['track_id'].map(track_class_map)
    
    # --- Bước 2c: Điền vào Khoảng trống (Inference-based Gap Filling) ---
    print("Bắt đầu quá trình điền vào khoảng trống...")
    
    # Danh sách chỉ chứa các dòng "ma" được tạo ra
    ghost_rows_list = []
    
    for track_id, track_df in tqdm(final_df.groupby('track_id'), desc="Filling Gaps"):
        # Lấy thông tin đã được thống nhất của track
        if track_df.empty: continue
        
        sorted_track = track_df.sort_values('frame_id')
        
        frames = sorted_track['frame_id'].values
        if len(frames) < 2: continue

        for i in range(len(frames) - 1):
            gap = frames[i+1] - frames[i]
            
            # Chỉ điền các khoảng trống nhỏ
            if 1 < gap <= MAX_FRAMES_TO_DISAPPEAR + 1:
                start_row = sorted_track.iloc[i]
                end_row = sorted_track.iloc[i+1]
                
                start_feat = start_row['clip_feature']
                end_feat = end_row['clip_feature']
                
                for j in range(1, gap):
                    interp_frame_id = frames[i] + j
                    interp_ratio = j / float(gap)
                    
                    # Nội suy bbox
                    interp_bbox = [
                        start_row['bbox'][k] + interp_ratio * (end_row['bbox'][k] - start_row['bbox'][k])
                        for k in range(4)
                    ]
                    
                    # Nội suy clip_feature nếu có thể
                    interp_feat = None
                    if start_feat is not None and end_feat is not None and isinstance(start_feat, np.ndarray) and isinstance(end_feat, np.ndarray):
                        interp_feat = start_feat + interp_ratio * (end_feat - start_feat)

                    # Tạo một dictionary mới cho dòng "ma"
                    ghost_row_dict = {
                        # Sao chép các thông tin nhất quán từ dòng bắt đầu
                        'video_name': start_row['video_name'],
                        'class_name': start_row['consistent_class_name'], # Dùng class đã bầu cử
                        'consistent_class_name': start_row['consistent_class_name'],
                        'track_id': start_row['track_id'],
                        
                        # Gán các giá trị đã được nội suy/mặc định
                        'frame_id': interp_frame_id,
                        'bbox': interp_bbox,
                        'clip_feature': interp_feat,
                        'confidence': 0.0, # Đánh dấu là đã được suy luận
                        'validation_method': 'interpolated',
                        'sam_iou_score': 0.0,
                        'clip_similarity': 0.0,
                    }
                    ghost_rows_list.append(ghost_row_dict)

    print(f"Đã tạo ra {len(ghost_rows_list):,} phát hiện 'ma' bằng phương pháp nội suy.")

    # Ghép các dòng "ma" với DataFrame "thật" ban đầu
    if ghost_rows_list:
        ghost_df = pd.DataFrame(ghost_rows_list)
        inference_df = pd.concat([final_df, ghost_df], ignore_index=True)
    else:
        inference_df = final_df

    print("Sắp xếp lại CSDL cuối cùng...")
    inference_df = inference_df.sort_values(['video_name', 'track_id', 'frame_id']).reset_index(drop=True)
    
    # Đổi tên cột class_name thành consistent_class_name để các truy vấn sau dùng
    inference_df['class_name'] = inference_df['consistent_class_name']
    
    # Lưu kết quả cuối cùng
    columns_to_drop = ['level_0', 'index', 'consistent_class_name']
    inference_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    
    inference_df.to_feather(OUTPUT_DB_PATH)
    print(f"\n--- GIAI ĐOẠN 2 (V7) HOÀN TẤT. Đã lưu CSDL đã qua suy luận tại {OUTPUT_DB_PATH} ---")

if __name__ == "__main__":
    build_inference_graph()