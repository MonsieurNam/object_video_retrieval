import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import time
import pickle
import gc
from tqdm import tqdm
# --- 1. CONFIGURATION ---
    # Import các thư viện cần thiết cho script chính
import cv2 # Cần cv2 để lấy total_frames
# --- Đường dẫn ---
VIDEO_SOURCE_DIR = Path('./Video_vong_loai/')
OUTPUT_DB_PATH = Path('./raw_tracklets.pkl')

# --- Cấu hình Model & Tracking ---
# Sử dụng model lớn nhất để có phát hiện chất lượng nhất
YOLO_MODEL_NAME = 'yolov8x.pt'
TRACKER_CONFIG_FILE = 'botsort.yaml'

# Các lớp quan tâm (ID từ bộ COCO)
TARGET_CLASSES_ID = [0, 1, 2, 3, 5, 7] # person, bicycle, car, motorcycle, bus, truck

# Ngưỡng tin cậy để tracker xem xét một phát hiện
# Đặt ở mức tương đối thấp để không bỏ lỡ các đối tượng khó, tracker sẽ giúp lọc bớt.
CONFIDENCE_THRESHOLD = 0.1

# --- 2. HELPER FUNCTION ---

def restructure_to_tracklets(video_name: str, detections: list) -> list:
    """
    Tái cấu trúc danh sách các phát hiện (detections) thành danh sách các tracklets.
    
    Args:
        video_name (str): Tên của video.
        detections (list): Danh sách các tuple (frame_id, cls_id, cls_name, bbox, conf, track_id).
    
    Returns:
        list: Danh sách các dictionary, mỗi dictionary là một tracklet.
    """
    if not detections:
        return []

    # Chuyển sang DataFrame để groupby hiệu quả
    df = pd.DataFrame(detections, columns=['frame_id', 'cls_id', 'cls_name', 'bbox', 'conf', 'track_id'])
    
    tracklets = []
    
    # Nhóm theo track_id
    for track_id, group in df.groupby('track_id'):
        # "Bầu chọn" class_name xuất hiện nhiều nhất cho track này
        final_class_name = group['cls_name'].mode()[0]
        final_class_id = group['cls_id'].mode()[0]
        
        # Sắp xếp các phát hiện trong track theo frame_id
        group = group.sort_values(by='frame_id')
        
        tracklet = {
            'video_name': video_name,
            'track_id': int(track_id),
            'class_id': int(final_class_id),
            'class_name': final_class_name,
            'frames': group['frame_id'].tolist(),
            'bboxes': group['bbox'].tolist(),
            'confidences': group['conf'].tolist(),
            'avg_confidence': group['conf'].mean()
        }
        tracklets.append(tracklet)
        
    return tracklets

def build_raw_tracklets_database():
    """Hàm chính để chạy tracking và xây dựng CSDL tracklet."""
    
    start_time = time.time()
    print("--- GIAI ĐOẠN 1: BẮT ĐẦU XÂY DỰNG CSDL TRACKLETS THÔ ---")
    
    # --- 3.1. Khởi tạo Model ---
    print(f"Đang tải model YOLO: {YOLO_MODEL_NAME}...")
    try:
        model = YOLO(YOLO_MODEL_NAME)
        print("Tải model thành công.")
    except Exception as e:
        print(f"LỖI: Không thể tải model YOLO. Lỗi: {e}")
        return

    # --- 3.2. Xử lý tất cả các video ---
    video_files = sorted(list(VIDEO_SOURCE_DIR.glob('*.mp4'))) # Sắp xếp để có thứ tự nhất quán
    if not video_files:
        print(f"LỖI: Không tìm thấy file video .mp4 nào trong '{VIDEO_SOURCE_DIR}'")
        return
        
    print(f"\nTìm thấy {len(video_files)} video để xử lý.")
    all_tracklets = []
    
    # Vòng lặp chính qua các file video
    for video_path in video_files:
        video_name = video_path.name
        print(f"\n--- Đang xử lý video: {video_name} ---")
        
        detections_in_video = []
        
        try:
            # Lấy tổng số frame để tqdm có thể hiển thị thanh tiến trình đầy đủ
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            results_generator = model.track(
                source=str(video_path),
                stream=True,
                persist=True,
                tracker=TRACKER_CONFIG_FILE,
                classes=TARGET_CLASSES_ID,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            # *** TÍCH HỢP TQDM VÀO ĐÂY ***
            # Bọc generator bằng tqdm để theo dõi tiến trình
            pbar = tqdm(results_generator, total=total_frames, desc=f"  -> Tracking Frames")
            for results in pbar:
                frame_idx = int(results.speed.get('frame', pbar.n)) # Lấy frame_idx chính xác hơn

                if results.boxes.id is None:
                    continue
                
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                
                for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                    detections_in_video.append(
                        (frame_idx, int(cls_id), model.names[int(cls_id)], box.tolist(), float(conf), int(track_id))
                    )
            
            # Tái cấu trúc thành tracklets sau khi xử lý xong video
            print(f"  -> Xử lý xong {len(detections_in_video)} phát hiện. Bắt đầu tái cấu trúc...")
            video_tracklets = restructure_to_tracklets(video_name, detections_in_video)
            print(f"  -> Tạo thành công {len(video_tracklets)} tracklets cho video này.")
            all_tracklets.extend(video_tracklets)

        except Exception as e:
            print(f"!!! LỖI khi xử lý video {video_name}: {e}")
        
        del detections_in_video
        gc.collect()

    # --- 3.3. Lưu CSDL Tracklets ---
    print(f"\n--- Hoàn thành xử lý video. Tổng cộng có {len(all_tracklets)} tracklets. ---")
    
    print(f"Đang lưu cơ sở dữ liệu vào file: {OUTPUT_DB_PATH}")
    with open(OUTPUT_DB_PATH, 'wb') as f:
        pickle.dump(all_tracklets, f)
    print("Lưu file thành công!")
    
    end_time = time.time()
    print(f"\n--- GIAI ĐOẠN 1 HOÀN TẤT. Tổng thời gian: {(end_time - start_time) / 60:.2f} phút. ---")


if __name__ == "__main__":
    build_raw_tracklets_database()