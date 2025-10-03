import pandas as pd
from tqdm import tqdm
from pathlib import Path

# --- 1. CONFIGURATION ---

INPUT_DB_PATH = Path('./evidence_database_v5_expert.feather')
OUTPUT_DB_PATH = Path('./tracked_database_v6.feather')

# --- Tham số của Thuật toán Tracking ---
# Ngưỡng IoU tối thiểu để coi hai box là cùng một đối tượng
IOU_THRESHOLD = 0.55 
# Số frame tối đa một track có thể bị "mất dấu" trước khi bị coi là kết thúc.
# Ví dụ: nếu một xe bị che khuất trong 3 frame rồi xuất hiện lại, nó vẫn được giữ lại track cũ.
MAX_FRAMES_TO_LOSE_TRACK = 250

# --- 2. HÀM TIỆN ÍCH ---

def calculate_iou(boxA, boxB):
    """
    Tính chỉ số Intersection over Union (IoU) giữa hai bounding box.
    boxes: [x1, y1, x2, y2]
    """
    # Xác định tọa độ của vùng giao nhau (intersection)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Tính diện tích vùng giao nhau
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0

    # Tính diện tích của mỗi bounding box
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Tính IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# --- 3. SCRIPT CHÍNH ---

def build_tracks():
    """Hàm chính để xây dựng các đường đi."""
    print("--- GIAI ĐOẠN 1.5: BẮT ĐẦU XÂY DỰNG TRACKS ---")

    if not INPUT_DB_PATH.exists():
        print(f"LỖI: Không tìm thấy file CSDL V5 tại '{INPUT_DB_PATH}'.")
        return

    print("Đang tải CSDL V5...")
    df = pd.read_feather(INPUT_DB_PATH)
    print(f"Tải thành công. Tổng số {len(df):,} phát hiện.")

    # Khởi tạo cột track_id
    df['track_id'] = -1
    next_track_id = 0

    # Sắp xếp để đảm bảo xử lý tuần tự theo thời gian
    print("Đang sắp xếp dữ liệu theo video và frame...")
    df.sort_values(['video_name', 'frame_id'], inplace=True)
    
    # Dictionary để lưu thông tin về các track "đang hoạt động" cho mỗi video
    # Cấu trúc: { video_name: { track_id: detection_row } }
    live_tracks = {}

    # Lặp qua từng video trong DataFrame
    for video_name, video_group in tqdm(df.groupby('video_name'), desc="Đang xử lý các video"):
        live_tracks[video_name] = {}
        
        # Lặp qua từng frame trong video đó
        for frame_id, frame_group in video_group.groupby('frame_id'):
            current_detections_indices = list(frame_group.index)
            matched_indices = set()

            # --- Bước 1: Cố gắng khớp các phát hiện hiện tại với các track đang hoạt động ---
            
            # Tạo một danh sách các track ID cần xóa sau vòng lặp để tránh thay đổi dict khi đang lặp
            tracks_to_delete = []

            for track_id, last_det in live_tracks[video_name].items():
                
                # Kiểm tra xem track có bị mất dấu quá lâu không
                if frame_id - last_det['frame_id'] > MAX_FRAMES_TO_LOSE_TRACK:
                    tracks_to_delete.append(track_id)
                    continue

                best_match_idx = -1
                best_iou = IOU_THRESHOLD
                
                # Tìm phát hiện hiện tại nào khớp nhất với track này
                for idx in current_detections_indices:
                    # Bỏ qua nếu phát hiện này đã được khớp với một track khác
                    if idx in matched_indices:
                        continue
                    
                    iou = calculate_iou(last_det['bbox'], df.loc[idx]['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = idx
                
                # Nếu tìm thấy một khớp tốt
                if best_match_idx != -1:
                    # Gán track_id cho phát hiện mới
                    df.at[best_match_idx, 'track_id'] = track_id
                    # Cập nhật thông tin mới nhất cho track
                    live_tracks[video_name][track_id] = df.loc[best_match_idx]
                    # Đánh dấu phát hiện này là đã được xử lý
                    matched_indices.add(best_match_idx)
            
            # Xóa các track đã bị mất dấu quá lâu
            for track_id in tracks_to_delete:
                del live_tracks[video_name][track_id]

            # --- Bước 2: Tạo track mới cho các phát hiện chưa được khớp ---
            unmatched_indices = set(current_detections_indices) - matched_indices
            for idx in unmatched_indices:
                # Gán một ID mới
                df.at[idx, 'track_id'] = next_track_id
                # Thêm track mới này vào danh sách đang hoạt động
                live_tracks[video_name][next_track_id] = df.loc[idx]
                # Tăng bộ đếm ID
                next_track_id += 1

    print("\n--- Hoàn thành quá trình tracking. ---")
    
    # --- Phân tích kết quả ---
    num_tracks = df['track_id'].nunique()
    print(f"Tổng số đường đi (tracks) duy nhất được tạo ra: {num_tracks}")
    
    # Tính độ dài trung bình của mỗi track
    track_lengths = df.groupby('track_id').size()
    print(f"Độ dài trung bình của một track: {track_lengths.mean():.2f} frames")
    print(f"Track dài nhất: {track_lengths.max()} frames")

    # Lưu CSDL đã được làm giàu
    print(f"Đang lưu CSDL V6 vào file: {OUTPUT_DB_PATH}")
    df.to_feather(OUTPUT_DB_PATH)
    print("Lưu file thành công!")

if __name__ == "__main__":
    build_tracks()
