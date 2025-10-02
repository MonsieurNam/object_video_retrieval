import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import collections
import time

# --- 1. CONFIGURATION ---
# Dùng CSDL "thô" nhưng đầy đủ nhất từ V3 hoặc V5 đều được. V3 có nhiều nút hơn.
# Nếu V5 đã đủ tốt, dùng nó sẽ nhanh hơn. Ở đây ta giả định dùng V5.
DB_PATH = Path('./evidence_database_v5_expert.feather')
OUTPUT_DB_PATH = Path('./inference_graph_database_v7.feather')

# --- Tham số cho việc xây dựng đồ thị và suy luận ---
IOU_THRESHOLD = 0.4            # Ngưỡng IoU để tạo "Cạnh Thời gian"
MAX_FRAME_GAP = 5              # Số frame tối đa một track có thể bị "mất dấu"
SIMILARITY_THRESHOLD = 0.92    # Ngưỡng Cosine Similarity để tạo "Cạnh Hình ảnh"
ANCHOR_CONF_THRESHOLD = 0.90   # Ngưỡng confidence để một phát hiện trở thành "nút mỏ neo"

# --- 2. HELPER FUNCTIONS ---

def calculate_iou(boxA, boxB):
    """Tính Intersection over Union (IoU) giữa hai bounding box."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def cosine_similarity_batch(matrix, vectors):
    """Tính cosine similarity giữa một ma trận và một tập các vector."""
    dot_product = np.dot(matrix, vectors.T)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norms = np.linalg.norm(vectors, axis=1)
    return dot_product / np.outer(matrix_norms, vector_norms)

# --- 3. MAIN SCRIPT ---

def build_and_run_inference():
    start_time = time.time()
    print("--- BẮT ĐẦU XÂY DỰNG ĐỒ THỊ SUY LUẬN (V7) ---")
    
    if not DB_PATH.exists():
        print(f"LỖI: Không tìm thấy CSDL tại '{DB_PATH}'")
        return
        
    print(f"Đang tải CSDL từ: {DB_PATH}")
    df = pd.read_feather(DB_PATH)
    
    # Chuẩn bị cột mới
    df['track_id'] = -1
    df['inferred_class_name'] = df['class_name'] # Bắt đầu bằng nhãn gốc
    df['inferred_method'] = 'original'
    
    # Sắp xếp để xử lý tuần tự
    df = df.sort_values(['video_name', 'frame_id']).reset_index(drop=True)

    all_video_dfs = []
    
    # --- BƯỚC 1: XÂY DỰNG ĐỒ THỊ VÀ TRACKING CHO TỪNG VIDEO ---
    for video_name, video_df in tqdm(df.groupby('video_name'), desc="Processing Videos"):
        print(f"\n[DEBUG] Bắt đầu xử lý video: {video_name}, có {len(video_df)} phát hiện.")
        
        live_tracks = {}  # { track_id: last_detection_index }
        next_track_id = 0
        
        # --- 1a. Tracking dựa trên IoU ---
        for frame_id, frame_group in tqdm(video_df.groupby('frame_id'), desc=" -> Tracking Frames", leave=False):
            current_det_indices = list(frame_group.index)
            matched_indices = set()

            # Cố gắng khớp với các track đang tồn tại
            for track_id, last_idx in list(live_tracks.items()):
                last_det = df.loc[last_idx]
                
                # Chỉ khớp nếu khoảng cách frame đủ nhỏ
                if frame_id - last_det['frame_id'] > MAX_FRAME_GAP:
                    del live_tracks[track_id] # Xóa track đã mất dấu quá lâu
                    continue

                best_match_idx = -1
                best_iou = IOU_THRESHOLD
                
                for idx in current_det_indices:
                    if idx in matched_indices: continue
                    iou = calculate_iou(last_det['bbox'], df.loc[idx]['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = idx
                
                if best_match_idx != -1:
                    df.at[best_match_idx, 'track_id'] = track_id
                    live_tracks[track_id] = best_match_idx
                    matched_indices.add(best_match_idx)

            # Tạo track mới cho các phát hiện không khớp
            unmatched_indices = set(current_det_indices) - matched_indices
            for idx in unmatched_indices:
                df.at[idx, 'track_id'] = next_track_id
                live_tracks[next_track_id] = idx
                next_track_id += 1
        
        print(f"[DEBUG] Tracking hoàn tất. Tổng số track được tạo: {next_track_id}")

        # --- 1b. Gán track_id cho DataFrame của video hiện tại ---
        # Lấy lại phần dataframe của video này (giờ đã có track_id)
        current_video_df = df[df['video_name'] == video_name].copy()

        # --- BƯỚC 2: SUY LUẬN VÀ SỬA LỖI TRÊN CÁC TRACK ---
        print("[DEBUG] Bắt đầu suy luận và sửa lỗi trên các track...")
        
        # --- 2a. Gieo mầm (Seeding) ---
        anchor_nodes = current_video_df[current_video_df['confidence'] >= ANCHOR_CONF_THRESHOLD]
        track_votes = collections.defaultdict(lambda: collections.defaultdict(int))
        
        for _, anchor in anchor_nodes.iterrows():
            track_votes[anchor['track_id']][anchor['class_name']] += 1
            
        print(f"[DEBUG] Tìm thấy {len(anchor_nodes)} nút mỏ neo (anchor nodes).")

        # --- 2b. Bầu cử đa số (Majority Voting) ---
        inferred_class_map = {}
        for track_id, votes in track_votes.items():
            if votes:
                # Bầu cử class name dựa trên các anchor node
                winner_class = max(votes, key=votes.get)
                inferred_class_map[track_id] = winner_class

        # --- 2c. Áp dụng kết quả suy luận ---
        # Chỉ cập nhật nhãn cho những track có anchor node
        tracks_to_update = current_video_df['track_id'].isin(inferred_class_map.keys())
        
        # Lưu lại nhãn gốc trước khi ghi đè
        original_classes = current_video_df.loc[tracks_to_update, 'class_name']
        
        # Ghi đè class_name bằng kết quả suy luận
        inferred_classes = current_video_df.loc[tracks_to_update, 'track_id'].map(inferred_class_map)
        current_video_df.loc[tracks_to_update, 'inferred_class_name'] = inferred_classes
        
        # Đánh dấu phương thức suy luận
        current_video_df.loc[tracks_to_update, 'inferred_method'] = 'propagated'
        
        # Debug: In ra số lượng nhãn đã bị thay đổi
        changed_labels = (original_classes != inferred_classes).sum()
        print(f"[DEBUG] Lan truyền nhãn hoàn tất. {changed_labels} nhãn đã được sửa lỗi.")

        all_video_dfs.append(current_video_df)
        
    # --- BƯỚC 3: TỔNG HỢP VÀ LƯU KẾT QUẢ ---
    if not all_video_dfs:
        print("LỖI: Không có dữ liệu sau khi xử lý.")
        return

    print("\n--- Hoàn thành xử lý tất cả video. Đang tổng hợp kết quả. ---")
    final_df = pd.concat(all_video_dfs, ignore_index=True)
    
    print("Thống kê kết quả suy luận cuối cùng:")
    print(final_df['inferred_method'].value_counts())
    
    # In ra so sánh trước và sau
    if 'propagated' in final_df['inferred_method'].unique():
        print("\nSo sánh chéo giữa nhãn gốc và nhãn suy luận:")
        print(pd.crosstab(final_df[final_df['inferred_method']=='propagated']['class_name'], 
                          final_df[final_df['inferred_method']=='propagated']['inferred_class_name']))

    print(f"\nĐang lưu CSDL V7 vào: {OUTPUT_DB_PATH}")
    final_df.to_feather(OUTPUT_DB_PATH)
    
    end_time = time.time()
    print(f"\n--- GIAI ĐOẠN 2 (V7) HOÀN TẤT. Tổng thời gian: {(end_time - start_time):.2f} giây. ---")

if __name__ == "__main__":
    build_and_run_inference()