import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Import YOLOWorld
from ultralytics import YOLOWorld
# Import factory tạo tracker
from boxmot.tracker_zoo import create_tracker

# --- 1. CONFIGURATION ---

VIDEO_SAMPLE_PATH = Path('./Video_vong_loai/File_2.mp4')
YOLO_MODEL_PATH = Path('./models/yolov8l-world.pt')  # YOLOWorld-Large
OUTPUT_VIDEO_PATH = Path('./yoloworld_tracker_test_output.mp4')

TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
DETECTION_CONF_THRESHOLD = 0.1
MAX_FRAMES_TO_PROCESS = 300

# --- 2. MAIN TEST SCRIPT ---

def run_integration_test():
    print("--- BẮT ĐẦU KIỂM TRA TÍCH HỢP PIPELINE V5 ---")

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
    tracker = create_tracker(
        'botsort',
        device=device,
        half=True,
        reid_weights=Path("osnet_x0_25_msmt17.pt")
    )
    print("Khởi tạo thành công.")

    # --- 2.3. Xử lý Video ---
    cap = cv2.VideoCapture(str(VIDEO_SAMPLE_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))

    frames_to_process = total_frames if MAX_FRAMES_TO_PROCESS == -1 else MAX_FRAMES_TO_PROCESS

    print(f"\nBắt đầu xử lý {frames_to_process} frames của video '{VIDEO_SAMPLE_PATH.name}'...")

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

    for i in tqdm(range(frames_to_process)):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=DETECTION_CONF_THRESHOLD, imgsz=1280, verbose=False)

        # Cập nhật tracker
        res = results[0]
        # nếu không có detections
        if res.boxes is None or len(res.boxes) == 0:
            tracks = np.zeros((0, 7))  # hoặc tùy tracker output
        else:
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = res.boxes.cls.cpu().numpy().reshape(-1, 1)
            dets = np.concatenate([xyxy, conf, cls], axis=1)
            tracks = tracker.update(dets, frame)

        if tracks.shape[0] > 0:
            for track in tracks:
                x1, y1, x2, y2 = [int(coord) for coord in track[:4]]
                track_id = int(track[4])
                conf = float(track[5])
                class_id = int(track[6])
                class_name = model.names[class_id]

                color = colors[track_id % 1000].tolist()

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)

        out_writer.write(frame)

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

    print("\n--- KIỂM TRA TÍCH HỢP HOÀN TẤT ---")
    print(f"Video kết quả đã được lưu tại: '{OUTPUT_VIDEO_PATH}'")

if __name__ == "__main__":
    run_integration_test()
