import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import sys
import time

# --- 1. CONFIGURATION ---

# --- Đường dẫn ---
ENRICHED_DB_PATH = Path('./enriched_tracklets.feather')
TEAM_ID = "AI25-15"
SUBMISSION_FILE_PATH = Path(f'./{TEAM_ID}.json')

# --- Tham số Truy vấn & Hậu xử lý ---
# Ngưỡng tin cậy cuối cùng cho tracklet. Giờ đây ta có thể dùng ngưỡng cao hơn
# vì chất lượng tracklet đã được thẩm định.
FINAL_TRACK_CONFIDENCE_THRESHOLD = 0.4
# Kích thước khoảng trống tối đa cho hàm fill_gaps
GAP_SIZE = 20 # ~0.7 giây ở 30fps

# --- 2. HÀM TIỆN ÍCH VÀ TRUY VẤN ---

def build_activity_schedule(df: pd.DataFrame):
    """
    Xây dựng một cấu trúc dữ liệu để truy vấn nhanh:
    {video_name: {frame_id: {'person': {track_id1, ...}, 'car': {track_id2, ...}}}}
    """
    print("Building activity schedule for fast querying...")
    schedule = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    track_info = df.set_index('track_id')['final_class_name'].to_dict()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  -> Indexing tracklets"):
        video = row['video_name']
        track_id = row['track_id']
        class_name = row['final_class_name']
        for frame in row['frames']:
            schedule[video][frame][class_name].add(track_id)
            
    return schedule, track_info


def fill_gaps(frame_list: list, max_gap_size: int = 15) -> list:
    if len(frame_list) < 2: return frame_list
    filled_list = [frame_list[0]]
    for i in range(1, len(frame_list)):
        prev_frame, current_frame = frame_list[i-1], frame_list[i]
        gap = current_frame - prev_frame
        if 1 < gap <= max_gap_size + 1:
            filled_list.extend(range(prev_frame + 1, current_frame))
        filled_list.append(current_frame)
    return filled_list

def apply_post_processing(result_dict: dict, max_gap: int) -> dict:
    if not result_dict: return {}
    processed_dict = {}
    for video, frames in result_dict.items():
        if frames:
            processed_dict[video] = fill_gaps(sorted(frames), max_gap_size=max_gap)
        else:
            processed_dict[video] = []
    return processed_dict

def run_all_queries(schedule: dict, track_info: dict):
    """Chạy tất cả 8 truy vấn trên lịch trình hoạt động."""
    final_results = {}
    
    for q_id in tqdm(range(1, 9), desc="Running all queries"):
        q_id_str = str(q_id)
        query_results_by_video = defaultdict(list)
        
        for video, frames_data in schedule.items():
            for frame, active_classes in frames_data.items():
                
                # Lấy số lượng active track cho mỗi class
                person_count = len(active_classes.get('person', set()))
                moto_count = len(active_classes.get('motorcycle', set()))
                bicycle_count = len(active_classes.get('bicycle', set()))
                car_count = len(active_classes.get('car', set()))
                
                # Logic cho từng câu hỏi
                match q_id_str:
                    case "1":
                        if person_count >= 1 and moto_count >= 1:
                            query_results_by_video[video].append(frame)
                    case "2":
                        if person_count >= 1 and bicycle_count >= 1:
                            query_results_by_video[video].append(frame)
                    case "3":
                        if car_count >= 1:
                            query_results_by_video[video].append(frame)
                    case "4": # Đã sửa thành "ít nhất 1"
                        if person_count >= 1 and bicycle_count >= 1:
                            query_results_by_video[video].append(frame)
                    case "5":
                        if person_count >= 1 and moto_count >= 1 and car_count >= 1:
                            query_results_by_video[video].append(frame)
                    case "6":
                        if person_count > 1:
                            query_results_by_video[video].append(frame)
                    case "7":
                        if moto_count > 1:
                            query_results_by_video[video].append(frame)
                    case "8":
                        total_tracks = sum(len(s) for s in active_classes.values())
                        if person_count == 3 and total_tracks == 3:
                            query_results_by_video[video].append(frame)
                            
        # Hậu xử lý và định dạng lại
        result_dict = {video: sorted(list(set(frames))) for video, frames in query_results_by_video.items()}
        final_results[q_id_str] = apply_post_processing(result_dict, max_gap=GAP_SIZE)

    return final_results

# --- 3. MAIN SCRIPT ---

def create_tracking_based_submission():
    start_time = time.time()
    print("--- GĐ 3 (V-Tracking): TẠO FILE SUBMISSION TỪ TRACKLETS ---")

    # --- Tải CSDL đã làm giàu ---
    if not ENRICHED_DB_PATH.exists():
        print(f"LỖI: Không tìm thấy CSDL tracklet đã làm giàu tại '{ENRICHED_DB_PATH}'.")
        print("Hãy chạy Giai đoạn 2 (enrich_tracklets) trước.")
        sys.exit(1)
        
    print("Đang tải CSDL tracklet đã làm giàu...")
    df = pd.read_feather(ENRICHED_DB_PATH)
    print(f"Tải thành công {len(df)} tracklets đã được xác thực.")

    # --- Lọc tracklet theo ngưỡng tin cậy cuối cùng ---
    print(f"Áp dụng ngưỡng tin cậy cuối cùng cho tracklet: {FINAL_TRACK_CONFIDENCE_THRESHOLD}")
    df_reliable = df[df['avg_confidence'] >= FINAL_TRACK_CONFIDENCE_THRESHOLD].copy()
    print(f"Còn lại {len(df_reliable)} tracklets đáng tin cậy để truy vấn.")

    # --- Xây dựng lịch trình và chạy truy vấn ---
    schedule, track_info = build_activity_schedule(df_reliable)
    final_submission = run_all_queries(schedule, track_info)

    # --- Lưu file kết quả ---
    print(f"\nĐang lưu kết quả vào file nộp bài: {SUBMISSION_FILE_PATH}")
    try:
        with open(SUBMISSION_FILE_PATH, 'w', encoding='utf8') as f:
            json.dump(final_submission, f, indent=4)
        
        end_time = time.time()
        print("\n--- HOÀN TẤT! ---")
        print(f"File nộp bài '{SUBMISSION_FILE_PATH}' đã được tạo thành công.")
        print(f"Tổng thời gian chạy Giai đoạn 3: {end_time - start_time:.2f} giây.")
        print("Hãy chạy Giai đoạn 4 để xác minh trực quan kết quả.")
    except Exception as e:
        print(f"\n!!! LỖI khi đang lưu file JSON: {e}")

if __name__ == "__main__":
    create_tracking_based_submission()