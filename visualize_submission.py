# visualize_submission.py
import pandas as pd
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import math

def resize_frame(frame, max_size=640):
    height, width = frame.shape[:2]
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    return frame

# --- GIAI ĐOẠN 1: THIẾT LẬP VÀ CẤU HÌNH ---
print("--- KHỞI ĐỘNG SCRIPT VISUALIZE SUBMISSION ---")

# 1. Định nghĩa Đường dẫn
SUBMISSION_FILE = Path('./AI25-15.json')
DB_PATH = Path('./god_database_final.parquet')
VIDEO_DIR = Path('./Video_vong_loai/')
OUTPUT_DIR = Path('./visual_report/')

# 2. Cấu hình Hình ảnh
THUMBNAIL_SIZE = (480, 270) # (width, height)
IMAGES_PER_ROW = 4
ROWS_PER_SHEET = 5
IMAGES_PER_SHEET = IMAGES_PER_ROW * ROWS_PER_SHEET

# 3. Định nghĩa "Ngữ cảnh Truy vấn" để làm nổi bật đối tượng
# Cần được cập nhật nếu có thêm câu hỏi mới
QUERY_CONTEXT = {
    '1': {'classes': ['person', 'motorcycle'], 'desc': "Người VÀ Xe máy"},
    '2': {'classes': ['person', 'bicycle'], 'desc': "Người VÀ Xe đạp"},
    '3': {'classes': ['car'], 'desc': "Xe ô tô"},
    '4': {'classes': ['person', 'bicycle'], 'desc': "Người và Xe đạp (logic phức tạp)"},
    '5': {'classes': ['person', 'motorcycle', 'car'], 'desc': "Người, Xe máy VÀ Xe ô tô"},
    '6': {'classes': ['person'], 'desc': "Nhiều hơn 1 Người"},
    '7': {'classes': ['motorcycle'], 'desc': "Nhiều hơn 1 Xe máy"},
    '8': {'classes': ['person', 'truck', 'bus', 'car', 'motorcycle', 'bicycle'], 'desc': "Logic đếm chính xác"},
    # Thêm các câu hỏi khác vào đây
}

# 4. Định nghĩa Bảng màu
CLASS_COLORS = {
    'person': (255, 100, 100),   # Đỏ nhạt
    'car': (100, 255, 100),      # Xanh lá nhạt
    'motorcycle': (100, 100, 255), # Xanh dương nhạt
    'bicycle': (255, 255, 100),  # Vàng
    'bus': (200, 100, 255),      # Tím
    'truck': (255, 150, 50),     # Cam
    'default': (200, 200, 200)   # Xám
}

# --- GIAI ĐOẠN 2: CÁC HÀM "WORKER" ---

def draw_detections(frame, frame_detections, question_id):
    """
    Vẽ các bounding box lên frame, sử dụng đầy đủ thông tin từ tracking
    và làm nổi bật các đối tượng mục tiêu.
    """
    target_classes = QUERY_CONTEXT.get(question_id, {}).get('classes', [])
    
    for _, det in frame_detections.iterrows():
        # --- SỬ DỤNG DỮ LIỆU ĐÃ QUA SUY LUẬN ---
        final_class_name = det['consistent_class_name']
        original_class_name = det['class_name']
        track_id = det['track_id']
        is_target = final_class_name in target_classes
        
        # Lấy các thông tin khác
        try:
            bbox = [int(c) for c in det['bbox'].strip('[]').split(', ')]
        except:
            continue
            
        conf = det['confidence']
        dom_color = det['dominant_color']
        
        # --- LOGIC VẼ THÔNG MINH HƠN ---
        box_color = CLASS_COLORS.get(final_class_name, CLASS_COLORS['default'])
        thickness = 2 if is_target else 1
        
        # 1. Vẽ Bounding Box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, thickness)
        
        # 2. Tạo Label đa thông tin
        label = f"T{track_id}:{final_class_name}:{conf:.2f} [{dom_color}]"
        
        # Kiểm tra xem tracking có sửa lỗi không
        was_corrected = (final_class_name != original_class_name)
        
        # 3. Vẽ nền cho Label
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_label = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + h + 10
        
        # Nếu tracking đã sửa lỗi, dùng màu đặc biệt (ví dụ: Tím) cho nền label
        label_bg_color = (255, 0, 255) if was_corrected else box_color
        cv2.rectangle(frame, (bbox[0], y_label - h - 5), (bbox[0] + w, y_label), label_bg_color, -1)
        
        # 4. Viết chữ Label
        cv2.putText(frame, label, (bbox[0] + 2, y_label - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
    return frame


def create_thumbnail(frame, video_name, frame_id):
    """Tạo ảnh thu nhỏ và thêm label."""
    thumb = cv2.resize(frame, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
    
    # Thêm label vào ảnh
    label = f"{Path(video_name).stem} | F:{frame_id}"
    cv2.rectangle(thumb, (0, THUMBNAIL_SIZE[1] - 25), (THUMBNAIL_SIZE[0], THUMBNAIL_SIZE[1]), (0, 0, 0), -1)
    cv2.putText(thumb, label, (10, THUMBNAIL_SIZE[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    return thumb

def create_contact_sheet(thumbnails, images_per_row):
    """Ghép các thumbnail thành một tấm ảnh lớn."""
    # Tạo một ảnh nền đen để đặt các thumbnail vào
    h, w, _ = thumbnails[0].shape
    rows = math.ceil(len(thumbnails) / images_per_row)
    sheet = np.zeros((rows * h, images_per_row * w, 3), dtype=np.uint8)
    
    for i, thumb in enumerate(thumbnails):
        row_idx = i // images_per_row
        col_idx = i % images_per_row
        y_start, x_start = row_idx * h, col_idx * w
        sheet[y_start:y_start+h, x_start:x_start+w] = thumb
        
    return sheet

def generate_html_report(report_dir, num_pages, question_id, description):
    """Tạo file index.html để xem báo cáo."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>Báo cáo Trực quan - Câu hỏi {question_id}</title>
        <style>
            body {{ font-family: sans-serif; background-color: #f0f0f0; margin: 20px; }}
            h1, h2 {{ text-align: center; color: #333; }}
            .container {{ max-width: 90%; margin: auto; }}
            img {{ display: block; margin: 20px auto; max-width: 100%; border: 1px solid #ccc; box-shadow: 2px 2px 8px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Báo cáo Trực quan - Câu hỏi {question_id}</h1>
            <h2>{description}</h2>
    """
    
    for i in range(num_pages):
        page_name = f"page_{i+1:02d}.jpg"
        html_content += f'        <img src="{page_name}" alt="Trang {i+1}">\n'
        
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(report_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)


# --- GIAI ĐOẠN 3: SCRIPT ĐIỀU PHỐI CHÍNH ---
def main():
    # 1. Khởi tạo
    if not SUBMISSION_FILE.exists():
        print(f"❌ LỖI: Không tìm thấy file submission tại '{SUBMISSION_FILE}'.")
        return
    if not DB_PATH.exists():
        print(f"❌ LỖI: Không tìm thấy file CSDL tại '{DB_PATH}'.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Đang tải dữ liệu...")
    with open(SUBMISSION_FILE, 'r') as f:
        submission_data = json.load(f)
    df = pd.read_parquet(DB_PATH)
    print("Tải dữ liệu thành công.")

    # 2. Vòng lặp chính qua các câu hỏi
    for question_id, results in submission_data.items():
        q_context = QUERY_CONTEXT.get(question_id)
        if not q_context:
            print(f"⚠️ Bỏ qua Câu hỏi {question_id}: không có ngữ cảnh được định nghĩa.")
            continue
        
        q_desc = q_context['desc']
        print(f"\n--- Đang xử lý Câu hỏi {question_id}: {q_desc} ---")
        
        q_dir = OUTPUT_DIR / f"Q_{question_id}"
        q_dir.mkdir(exist_ok=True)
        
        all_thumbnails_for_question = []
        
        # Nhóm các frame cần xử lý theo video để tối ưu việc đọc file
        frames_by_video = {}
        for video_name, frame_ids in results.items():
            if frame_ids: # Chỉ xử lý nếu có frame
                frames_by_video[video_name] = sorted(frame_ids)

        # 3. Vòng lặp phụ qua các video
        pbar_videos = tqdm(frames_by_video.items(), desc="  -> Xử lý videos")
        for video_name, frame_ids in pbar_videos:
            pbar_videos.set_postfix_str(video_name)
            video_path = VIDEO_DIR / video_name
            if not video_path.exists():
                print(f"⚠️ CẢNH BÁO: Không tìm thấy file video '{video_path}'. Bỏ qua.")
                continue
                
            cap = cv2.VideoCapture(str(video_path))
            
            for frame_id in frame_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                
                if ret:
                    frame = resize_frame(frame, max_size=640)
                    frame_detections = df[(df['video_name'] == video_name) & (df['frame_id'] == frame_id)]
                    annotated_frame = draw_detections(frame, frame_detections, question_id)
                    thumbnail = create_thumbnail(annotated_frame, video_name, frame_id)
                    all_thumbnails_for_question.append(thumbnail)
            
            cap.release()

        # 4. Tạo báo cáo
        if not all_thumbnails_for_question:
            print("  -> Không có frame nào để hiển thị cho câu hỏi này.")
            continue
            
        print(f"  -> Đã tạo {len(all_thumbnails_for_question)} ảnh thu nhỏ. Đang ghép thành các trang...")
        num_pages = math.ceil(len(all_thumbnails_for_question) / IMAGES_PER_SHEET)
        
        for i in range(num_pages):
            start_idx = i * IMAGES_PER_SHEET
            end_idx = start_idx + IMAGES_PER_SHEET
            batch = all_thumbnails_for_question[start_idx:end_idx]
            
            sheet = create_contact_sheet(batch, IMAGES_PER_ROW)
            
            sheet_path = q_dir / f"page_{i+1:02d}.jpg"
            cv2.imwrite(str(sheet_path), sheet)
            
        generate_html_report(q_dir, num_pages, question_id, q_desc)
        print(f"  -> ✅ Báo cáo đã được tạo tại: {q_dir / 'index.html'}")

    print("\n--- HOÀN TẤT ---")
    print(f"Mở thư mục '{OUTPUT_DIR}' để xem các báo cáo.")

if __name__ == "__main__":
    main()