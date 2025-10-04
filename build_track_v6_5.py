import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tracker import DeepSORTTracker # <<< IMPORT MỚI

# --- 1. CONFIGURATION ---
INPUT_DB_PATH = Path('./evidence_database_v5_expert.feather')
OUTPUT_DB_PATH = Path('./tracked_database_v6_5.feather') # <<< Tên file output mới

# --- Tham số của Thuật toán Tracking Mới (DeepSORT-like) ---
MAX_AGE = 200       # Số frame tối đa một track có thể bị "mất dấu" (tương tự MAX_FRAMES_TO_LOSE_TRACK)
NN_BUDGET = 50      # Giữ lại 50 đặc trưng ngoại hình gần nhất cho mỗi track để so sánh
LAMBDA_VAL = 0.3    # Trọng số kết hợp: 0.3 cho chuyển động, 0.7 cho ngoại hình. Tinh chỉnh sau.

# --- 2. SCRIPT CHÍNH ---
def build_tracks_v6_5():
    """Hàm chính để xây dựng các đường đi với thuật toán nâng cao."""
    print("--- GIAI ĐOẠN 1.7: BẮT ĐẦU XÂY DỰNG TRACKS (DeepSORT-like) ---")

    if not INPUT_DB_PATH.exists():
        print(f"LỖI: Không tìm thấy file CSDL V5 tại '{INPUT_DB_PATH}'.")
        return

    print("Đang tải CSDL V5...")
    df = pd.read_feather(INPUT_DB_PATH)
    print(f"Tải thành công. Tổng số {len(df):,} phát hiện.")

    # Khởi tạo cột track_id
    df['track_id'] = -1

    # Sắp xếp để đảm bảo xử lý tuần tự theo thời gian
    print("Đang sắp xếp dữ liệu theo video và frame...")
    df.sort_values(['video_name', 'frame_id'], inplace=True)
    
    # Lặp qua từng video trong DataFrame
    for video_name, video_group in tqdm(df.groupby('video_name'), desc="Đang xử lý các video"):
        
        # Khởi tạo một tracker mới cho mỗi video
        tracker = DeepSORTTracker(max_age=MAX_AGE, nn_budget=NN_BUDGET, lambda_val=LAMBDA_VAL)
        
        # Lặp qua từng frame trong video đó
        for frame_id, frame_group in video_group.groupby('frame_id'):
            if frame_group.empty:
                continue

            # >>> LOGIC MỚI SẼ ĐƯỢC TRIỂN KHAI Ở ĐÂY (GIAI ĐOẠN 2) <<<
            # 1. Định dạng lại các phát hiện của frame hiện tại cho tracker.
            # Chuyển đổi dataframe của frame thành một list các dictionary
            detections_for_frame = []
            for idx, row in frame_group.iterrows():
                detections_for_frame.append({
                    'bbox': row['bbox'],
                    'confidence': row['confidence'],
                    'clip_feature': row['clip_feature'],
                    'original_index': idx # <<< Rất quan trọng để gán lại track_id
                })

            # 2. Gọi hàm update của tracker
            tracker.update(detections_for_frame)

            # 3. Gán track_id vào dataframe gốc
            for track in tracker.tracks:
                # Chỉ gán ID cho các track đang "sống" và đã được "xác nhận"
                if not track.is_confirmed(): # and track.time_since_update > 0: #track cần được cập nhật
                    continue

                # Lấy bbox và index gốc từ lần cập nhật gần nhất của track
                #last_detection_bbox = track.bbox
                last_detection = track.last_matched_detection
                original_index = last_detection['original_index'] # Quan trọng
                df.at[original_index, 'track_id'] = track.track_id # Gán track_id

    print("\n--- Hoàn thành quá trình tracking. ---")
    
    # --- Phân tích kết quả ---
    # Chú ý: Cần lọc ra các track_id != -1 trước khi phân tích
    tracked_df = df[df['track_id'] != -1]
    if not tracked_df.empty:
        num_tracks = tracked_df['track_id'].nunique()
        print(f"Tổng số đường đi (tracks) duy nhất được tạo ra: {num_tracks}")
        
        track_lengths = tracked_df.groupby('track_id').size()
        print(f"Độ dài trung bình của một track: {track_lengths.mean():.2f} frames")
        print(f"Track dài nhất: {track_lengths.max()} frames")
    else:
        print("Không có track nào được tạo ra.")

    # Lưu CSDL đã được làm giàu
    print(f"Đang lưu CSDL V7 vào file: {OUTPUT_DB_PATH}")
    df.to_feather(OUTPUT_DB_PATH)
    print("Lưu file thành công!")

if __name__ == "__main__":
    build_tracks_v6_5()