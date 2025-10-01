
import cv2
from ultralytics import YOLO
from pathlib import Path

# --- 1. CONFIGURATION ---

# Chọn một video để thử nghiệm
VIDEO_SOURCE_PATH = Path('./Video_vong_loai/File_2.mp4') 

# Chọn model YOLO. 'yolov8n.pt' (nano) là nhanh nhất để thử nghiệm.
# Khi chạy thật, chúng ta sẽ dùng 'yolov8x.pt'.
YOLO_MODEL_NAME = 'yolov8n.pt'
SAVE_VIDEO = True
OUTPUT_VIDEO_PATH = Path('./tracking_result.mp4')

# Chọn thuật toán tracking. 'botsort.yaml' là lựa chọn hàng đầu.
# Các lựa chọn khác: 'bytetrack.yaml'
TRACKER_CONFIG_FILE = 'botsort.yaml'

# Các lớp quan tâm (ID từ bộ COCO)
TARGET_CLASSES_ID = [0, 1, 2, 3, 5, 7] # person, bicycle, car, motorcycle, bus, truck

# Ngưỡng tin cậy để hiển thị. Có thể đặt thấp để xem tracker hoạt động ra sao.
CONFIDENCE_THRESHOLD = 0.3

# Tùy chọn: Có hiển thị video kết quả trực tiếp không?
# Đặt là False nếu bạn chạy trên môi trường không có giao diện đồ họa (như server).
SHOW_VIDEO = False

# --- 2. MAIN SCRIPT ---

def run_tracking_test():
    """Hàm chính để chạy thử nghiệm tracking."""
    
    print("--- BẮT ĐẦU KIỂM THỬ ENGINE TRACKING ---")

    # --- 2.1. Kiểm tra file và khởi tạo Model ---
    if not VIDEO_SOURCE_PATH.exists():
        print(f"LỖI: Không tìm thấy video thử nghiệm tại '{VIDEO_SOURCE_PATH}'")
        return

    print(f"Đang tải model YOLO: {YOLO_MODEL_NAME}...")
    try:
        model = YOLO(YOLO_MODEL_NAME)
        print("Tải model thành công.")
    except Exception as e:
        print(f"LỖI: Không thể tải model YOLO. Lỗi: {e}")
        return
    
    video_writer = None
    if SAVE_VIDEO:
        # Lấy thông tin của video gốc để tạo video mới
        cap_info = cv2.VideoCapture(str(VIDEO_SOURCE_PATH))
        width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap_info.get(cv2.CAP_PROP_FPS))
        cap_info.release()
        
        # Định nghĩa codec và tạo đối tượng VideoWriter
        # 'mp4v' là một lựa chọn phổ biến cho file .mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))
        print(f"Sẽ lưu kết quả tracking vào file: {OUTPUT_VIDEO_PATH}")
        
    # --- 2.2. Chạy Tracking trên Video Stream ---
    print(f"Bắt đầu tracking trên video: {VIDEO_SOURCE_PATH.name}")
    print(f"Sử dụng tracker: {TRACKER_CONFIG_FILE}")

    # model.track() trả về một generator, ta sẽ lặp qua nó
    try:
        results_generator = model.track(
            source=str(VIDEO_SOURCE_PATH),
            stream=True,  # Xử lý video như một stream, tiết kiệm bộ nhớ
            persist=True, # Giữ lại thông tin track qua các frame
            tracker=TRACKER_CONFIG_FILE,
            classes=TARGET_CLASSES_ID,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False # Tắt bớt log của ultralytics
        )
    except Exception as e:
        print(f"LỖI: Quá trình tracking gặp sự cố. Lỗi: {e}")
        return

    frame_count = 0
    
    for results in results_generator:
        # `results` chứa thông tin của một frame duy nhất
        frame = results.orig_img # Lấy frame gốc
        frame_id = int(results.speed.get('frame', frame_count)) # Lấy frame_id nếu có
        
        print(f"\n--- FRAME #{frame_id} ---")
        
        # Kiểm tra xem có đối tượng nào được track không
        if results.boxes.id is None:
            print("Không có đối tượng nào được theo dõi trong frame này.")
            continue
            
        # Lấy thông tin từ các box
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        
        # Lặp qua từng đối tượng được phát hiện trong frame
        for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
            class_name = model.names[cls_id]
            
            # In thông tin cốt lõi ra terminal
            print(f"  -> Track ID: {track_id}, Class: {class_name}, Confidence: {conf:.2f}, BBox: {box.tolist()}")
            
            # Vẽ lên frame để hiển thị trực quan (nếu SHOW_VIDEO = True)
            x1, y1, x2, y2 = box
            color = (int(track_id * 29 % 255), int(track_id * 53 % 255), int(track_id * 97 % 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {model.names[cls_id]}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame_count += 1
        
        if SAVE_VIDEO and video_writer is not None:
            video_writer.write(frame)

    # --- GIẢI PHÓNG VIDEOWRITER ---
    if SAVE_VIDEO and video_writer is not None:
        video_writer.release()
        print(f"\nĐã lưu video kết quả thành công!")

    if SHOW_VIDEO:
        cv2.destroyAllWindows()
        
    print("\n--- KIỂM THỬ HOÀN TẤT ---")

if __name__ == "__main__":
    run_tracking_test()