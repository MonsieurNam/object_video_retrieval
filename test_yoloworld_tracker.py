
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Import các thành phần cần thiết
from ultralytics import YOLOWorld
from boxmot import BoTSORT

# --- 1. CONFIGURATION ---

# --- Đường dẫn ---
VIDEO_SAMPLE_PATH = Path('./Video_vong_loai/File_2.mp4')
YOLO_MODEL_PATH = Path('./models/yoloworld-l.pt') # Sử dụng model Large
OUTPUT_VIDEO_PATH = Path('./yoloworld_tracker_test_output.mp4')

# --- Cấu hình Model & Tracker ---
# Các lớp đối tượng mà chúng ta muốn YOLOWorld tìm kiếm
TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# Ngưỡng tin cậy cho YOLOWorld. Bắt đầu với một ngưỡng an toàn để kiểm tra.
DETECTION_CONF_THRESHOLD = 0.3

# Số frame tối đa để xử lý cho video test (để chạy nhanh)
# Đặt là -1 để xử lý toàn bộ video.
MAX_FRAMES_TO_PROCESS = 300 

# --- 2. MAIN TEST SCRIPT ---

def run_integration_test():
    """Hàm chính để chạy kịch bản kiểm thử tích hợp."""
    
    print("--- BẮT ĐẦU KIỂM TRA TÍCH HỢP PIPELINE V5 ---")

    # --- 2.1. Kiểm tra file và thiết bị ---
    if not VIDEO_SAMPLE_PATH.exists():
        print(f"LỖI: Không tìm thấy video mẫu tại '{VIDEO_SAMPLE_PATH}'")
        return
    if not YOLO_MODEL_PATH.exists():
        print(f"LỖI: Không tìm thấy model YOLOWorld tại '{YOLO_MODEL_PATH}'")
        return

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")

    # --- 2.2. Khởi tạo Model và Tracker ---
    print("Đang tải model YOLOWorld...")
    model = YOLOWorld(YOLO_MODEL_PATH)
    
    print(f"Thiết lập các lớp đối tượng cho YOLOWorld: {TARGET_CLASSES}")
    model.set_classes(TARGET_CLASSES)
    
    print("Đang khởi tạo tracker (BoTSORT)...")
    # BoTSORT là một lựa chọn mạnh mẽ, cân bằng giữa tốc độ và độ chính xác
    tracker = BoTSORT(
        model_weights=Path('osnet_x0_25_msmt17.pt'), # boxmot sẽ tự tải nếu chưa có
        device=device,
        fp16=True, # Sử dụng fp16 để tăng tốc trên GPU hỗ trợ
    )
    print("Khởi tạo thành công.")

    # --- 2.3. Xử lý Video ---
    cap = cv2.VideoCapture(str(VIDEO_SAMPLE_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Chuẩn bị để ghi video output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))

    frames_to_process = total_frames if MAX_FRAMES_TO_PROCESS == -1 else MAX_FRAMES_TO_PROCESS

    print(f"\nBắt đầu xử lý {frames_to_process} frames của video '{VIDEO_SAMPLE_PATH.name}'...")
    
    # Tạo một bảng màu ngẫu nhiên cho các track ID
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

    for i in tqdm(range(frames_to_process)):
        ret, frame = cap.read()
        if not ret:
            break

        # Chạy YOLOWorld để phát hiện đối tượng
        results = model.predict(frame, conf=DETECTION_CONF_THRESHOLD, verbose=False)
        
        # Đưa kết quả phát hiện vào tracker
        # tracker.update() yêu cầu một mảng numpy, không phải object của ultralytics
        # boxmot đã tích hợp sẵn để xử lý trực tiếp kết quả từ ultralytics
        tracks = tracker.update(results[0], frame) # Truyền frame gốc vào để ReID
        
        # Vẽ kết quả tracking lên frame
        if tracks.shape[0] > 0:
            for track in tracks:
                # Định dạng của track: [x1, y1, x2, y2, track_id, conf, class_id]
                x1, y1, x2, y2 = [int(coord) for coord in track[:4]]
                track_id = int(track[4])
                conf = float(track[5])
                class_id = int(track[6])
                class_name = model.names[class_id]
                
                # Chọn màu dựa trên track_id
                color = colors[track_id % 1000].tolist()
                
                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Tạo và vẽ nhãn
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
        # Ghi frame đã được vẽ vào video output
        out_writer.write(frame)

    # --- 2.4. Dọn dẹp ---
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    
    print("\n--- KIỂM TRA TÍCH HỢP HOÀN TẤT ---")
    print(f"Video kết quả đã được lưu tại: '{OUTPUT_VIDEO_PATH}'")
    print("Hãy mở file video này lên và kiểm tra:")
    print("  - Các đối tượng có được phát hiện không?")
    print("  - Mỗi đối tượng có được gán một ID duy nhất và ổn định không?")
    print("  - ID có giữ nguyên khi đối tượng di chuyển không?")

if __name__ == "__main__":
    run_integration_test()