
import pandas as pd
import json
from pathlib import Path
import cv2
import numpy as np
import sys
from tqdm import tqdm
import math

# --- 1. CONFIGURATION ---
SUBMISSION_FILE_PATH = Path('./AI25-15.json')
DB_PATH = Path('./inference_graph_database_v7.feather')
VIDEO_SOURCE_DIR = Path('./Video_vong_loai/')
OUTPUT_DIR = Path('./visual_report_v7_FINAL/')

# --- Tham số (Đồng bộ với query_library_v7.py) ---
BASE_CONF = 0.35
COUNTING_CONF_BASE = 0.45 # Ngưỡng cho câu hỏi đếm
STRICT_COUNTING_CONF_BASE = 0.7 # Ngưỡng cho câu hỏi đếm chính xác
SAM_IOU = 0.90
COUNTING_CONF = 0.45
STRICT_COUNTING_CONF = 0.7
SHOW_TILE_LABEL=True
# --- Cấu hình Báo cáo ---
IMAGES_PER_ROW = 5
ROWS_PER_SHEET = 8
THUMBNAIL_MAX_W = 480
THUMBNAIL_MAX_H = 270
SAVE_FORMAT = 'webp'
WEBP_QUALITY = 85
LABEL_POS = "bottom"
LABEL_ALPHA = 0.7

QUERY_DESCRIPTIONS = {
    "1": "Người VÀ Xe gắn máy", "2": "Người VÀ Xe đạp", "3": "Xe ô tô",
    "4": ">=1 Người VÀ >=1 Xe đạp", "5": "Người, Xe máy, VÀ Xe ô tô",
    "6": "Nhiều hơn 1 Người", "7": "Nhiều hơn 1 Xe máy", "8": "CHỈ có 3 Người"
}

# --- 2. HÀM TIỆN ÍCH ---


_video_path_cache = {}
def find_video_path(video_name: str, root: Path) -> Path | None:
    if video_name in _video_path_cache: return _video_path_cache[video_name]
    try:
        path = next(root.rglob(f"**/{video_name}"))
        _video_path_cache[video_name] = path
        return path
    except StopIteration: return None

class VideoReaderPool:
    def __init__(self, max_open=8):
        self.max_open, self.pool = max_open, {}
    def get(self, video_path: Path):
        key = str(video_path)
        if key in self.pool:
            cap = self.pool.pop(key); self.pool[key] = cap; return cap
        if len(self.pool) >= self.max_open:
            old_key = next(iter(self.pool)); old_cap = self.pool.pop(old_key); old_cap.release()
        cap = cv2.VideoCapture(str(video_path)); self.pool[key] = cap; return cap
    def close_all(self):
        for cap in self.pool.values(): cap.release()
        self.pool.clear()


def make_thumbnail(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    y0, x0 = (max_h - nh) // 2, (max_w - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def draw_tile_label(img, text):
    h, w = img.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    bar_h = th + 8
    y1 = h - bar_h if LABEL_POS == "bottom" else 0
    y2 = h if LABEL_POS == "bottom" else bar_h
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, LABEL_ALPHA, img, 1 - LABEL_ALPHA, 0, dst=img)
    cv2.putText(img, text, (5, y2 - 5), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img

def save_contact_sheet(thumbnails, out_path):
    if not thumbnails: return
    sheet = np.vstack([np.hstack(thumbnails[i:i+IMAGES_PER_ROW]) for i in range(0, len(thumbnails), IMAGES_PER_ROW)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_WEBP_QUALITY), WEBP_QUALITY] if SAVE_FORMAT == 'webp' else []
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), params)

def write_html_index(q_dir, pages, title):
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>body{{font-family:sans-serif;background-color:#222;color:#eee}} h1,h3{{text-align:center}} img{{display:block;margin:20px auto;max-width:100%;height:auto;border:1px solid #444}}</style></head><body><h1>{title}</h1>"
    for p in pages: html += f"<h3>{Path(p).name}</h3><img loading='lazy' src='{Path(p).name}'>"
    html += "</body></html>"
    (q_dir / "index.html").write_text(html, encoding="utf-8")

# Tạo một bảng màu ngẫu nhiên cho các track_id
np.random.seed(42)
TRACK_COLORS = np.random.randint(50, 255, size=(1000, 3), dtype=np.uint8)

def draw_detections_v7(frame_bgr, det_rows, question_id):
    """Hàm vẽ bbox V7, tô màu theo track_id và hiển thị nhãn đã suy luận."""
    
    # Logic để quyết định ngưỡng tin cậy dựa trên câu hỏi
    conf_threshold = BASE_CONF
    if question_id in "67":
        conf_threshold = COUNTING_CONF_BASE
    elif question_id == "8":
        conf_threshold = STRICT_COUNTING_CONF_BASE

    for _, det in det_rows.iterrows():
        if det['confidence'] < conf_threshold:
            continue
            
        x1, y1, x2, y2 = [int(c) for c in det['bbox']]
        
        # Sử dụng nhãn đã suy luận
        inferred_cls = det['inferred_class_name']
        original_cls = det['class_name']
        conf = det['confidence']
        track_id = int(det['track_id'])
        
        # Chọn màu dựa trên track_id
        color = TRACK_COLORS[track_id % len(TRACK_COLORS)].tolist()
        
        label = f"T{track_id}:{inferred_cls}:{conf:.2f}"
        # Thêm ghi chú nếu nhãn đã bị sửa
        if inferred_cls != original_cls:
            label += f" (was {original_cls})"
        
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_bgr, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame_bgr

def read_frame_annotated_v7(pool, df_idx, video_name, frame_id, question_id):
    """Hàm đọc và chú thích frame phiên bản V7."""
    video_path = find_video_path(video_name, VIDEO_SOURCE_DIR)
    if not video_path: return None
    cap = pool.get(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok: return None
    rows = df_idx.get((video_name, frame_id))
    if rows is not None and not rows.empty:
        frame = draw_detections_v7(frame, rows, question_id)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# --- 3. SCRIPT ĐIỀU PHỐI CHÍNH ---
def generate_full_visual_report_v7():
    print("=== GENERATING V7 FINAL VISUAL REPORT ===")
    if not SUBMISSION_FILE_PATH.exists(): raise FileNotFoundError(f"Submission file not found: {SUBMISSION_FILE_PATH}")
    if not DB_PATH.exists(): raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    print("Loading database and submission file...")
    df = pd.read_feather(DB_PATH)
    df_idx = {k: v for k, v in df.groupby(['video_name', 'frame_id'])}
    with open(SUBMISSION_FILE_PATH, 'r') as f: submission = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_per_sheet = IMAGES_PER_ROW * ROWS_PER_SHEET
    pool = VideoReaderPool(max_open=8)

    try:
        for q_id, results in submission.items():
            desc = QUERY_DESCRIPTIONS.get(str(q_id), f"Question {q_id}")
            print("\n" + "="*80)
            print(f"[Q{q_id}] {desc}")

            frames_to_process = sorted([(v_name, int(fid)) for v_name, f_list in (results or {}).items() for fid in f_list])
            num_total = len(frames_to_process)
            
            if num_total == 0:
                print("→ Không có kết quả nào để hiển thị.")
                continue

            n_pages = math.ceil(num_total / total_per_sheet)
            print(f"→ Tổng số {num_total} khung hình kết quả. Sẽ tạo {n_pages} trang báo cáo.")
            
            q_dir = OUTPUT_DIR / f"Q{q_id}"
            q_dir.mkdir(parents=True, exist_ok=True)
            pages_paths = []
            
            pbar = tqdm(total=num_total, desc=f"  -> Rendering Q{q_id}", ncols=100)
            
            for page_idx in range(n_pages):
                batch = frames_to_process[page_idx*total_per_sheet : (page_idx+1)*total_per_sheet]
                thumbnails = []
                for (video_name, fid) in batch:
                    img = read_frame_annotated_v7(pool, df_idx, video_name, fid, q_id)
                    thumb = make_thumbnail(img if img is not None else np.zeros((THUMBNAIL_MAX_H, THUMBNAIL_MAX_W, 3), dtype=np.uint8), THUMBNAIL_MAX_W, THUMBNAIL_MAX_H)
                    if SHOW_TILE_LABEL:
                        thumb = draw_tile_label(thumb, f"{Path(video_name).stem} | F{int(fid)}")
                    thumbnails.append(thumb)
                    pbar.update(1)

                while len(thumbnails) % IMAGES_PER_ROW != 0:
                    thumbnails.append(np.zeros((THUMBNAIL_MAX_H, THUMBNAIL_MAX_W, 3), dtype=np.uint8))
                
                out_name = f"Q{q_id}_page_{page_idx+1:03d}.{SAVE_FORMAT}"
                out_path = q_dir / out_name
                save_contact_sheet(thumbnails, out_path)
                pages_paths.append(out_path)
                
                del thumbnails
                gc.collect()

            pbar.close()
            write_html_index(q_dir, pages_paths, f"Visual Results for Q{q_id}: {desc}")
            print(f"→ Báo cáo đã được tạo tại: {q_dir/'index.html'}")

    finally:
        pool.close_all()

    print("\n" + "="*80)
    print("✓ HOÀN TẤT. Mở các file index.html trong thư mục:", str(OUTPUT_DIR))

if __name__ == "__main__":
    generate_full_visual_report_v7()