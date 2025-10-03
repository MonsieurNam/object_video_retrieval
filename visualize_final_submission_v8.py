import pandas as pd
import json
from pathlib import Path
import cv2
import numpy as np
import sys
from tqdm import tqdm
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# --- 1. CONFIGURATION ---

SUBMISSION_FILE_PATH = Path('./AI25-15.json')
DB_PATH = Path('./deep_tracked_database_v8.feather')
VIDEO_SOURCE_DIR = Path('./Video_vong_loai/')
OUTPUT_DIR = Path('./visual_report_v8_FINAL/')

# --- Tham số Logic (PHẢI ĐỒNG BỘ VỚI query_library_v8.py) ---
BASE_CONF = 0.25
COUNTING_CONF_BASE = 0.50
STRICT_COUNTING_CONF_BASE = 0.65

# --- Cấu hình Báo cáo & Song song hóa ---
IMAGES_PER_ROW = 5
ROWS_PER_SHEET = 8
THUMBNAIL_MAX_W = 480; THUMBNAIL_MAX_H = 270
SAVE_FORMAT = 'webp'; WEBP_QUALITY = 85
SHOW_TILE_LABEL = True; LABEL_POS = "bottom"; LABEL_ALPHA = 0.7
try: NUM_WORKERS = max(1, os.cpu_count() - 2)
except: NUM_WORKERS = 4

QUERY_DESCRIPTIONS = {
    "1": "Người VÀ Xe gắn máy", "2": "Người VÀ Xe đạp", "3": "Xe ô tô",
    "4": "ĐÚNG 1 Người & 1 Xe đạp", "5": "Người, Xe máy, VÀ Xe ô tô",
    "6": "Nhiều hơn 1 Người", "7": "Nhiều hơn 1 Xe máy", "8": "CHỈ có 3 Người"
}

# --- 2. HÀM WORKER VÀ CÁC HÀM TIỆN ÍCH LIÊN QUAN (ĐẦY ĐỦ) ---

# Các biến toàn cục sẽ được khởi tạo một lần cho mỗi tiến trình con
_df_global = None
_video_root_global = None
_track_colors_global = None

def init_worker_v8(db_path, video_root):
    """Hàm khởi tạo, tải và tiền xử lý CSDL một lần cho mỗi worker."""
    global _df_global, _video_root_global, _track_colors_global
    cv2.setNumThreads(1)
    
    df = pd.read_feather(db_path)
    
    track_class_map = df.groupby('track_id')['class_name'].agg(lambda x: x.mode()[0])
    df['consistent_class_name'] = df['track_id'].map(track_class_map)
    
    _df_global = df
    _video_root_global = video_root
    
    np.random.seed(42)
    max_track_id = df['track_id'].max() if not df.empty else 0
    _track_colors_global = np.random.randint(60, 255, size=(max_track_id + 1, 3), dtype=np.uint8)

def _find_video_path(video_name: str, root: Path) -> Path | None:
    try:
        p = root / video_name
        if p.exists(): return p
        return next(root.rglob(f"**/{video_name}"))
    except StopIteration: return None

def _draw_detections_v8(frame_bgr, frame_id, video_name, question_id):
    global _df_global, _track_colors_global
    detections = _df_global[(_df_global['video_name'] == video_name) & (_df_global['frame_id'] == frame_id)]

    conf_threshold = BASE_CONF
    if question_id in "67": conf_threshold = COUNTING_CONF_BASE
    elif question_id in "48": conf_threshold = STRICT_COUNTING_CONF_BASE

    for _, det in detections.iterrows():
        if det['confidence'] < conf_threshold: continue
            
        x1, y1, x2, y2 = [int(c) for c in det['bbox']]
        
        inferred_cls = det['consistent_class_name']
        original_cls = det['class_name']
        conf = det['confidence']
        track_id = int(det['track_id'])
        
        color = _track_colors_global[track_id % len(_track_colors_global)].tolist()
        
        label = f"T{track_id}:{inferred_cls}:{conf:.2f}"
        if inferred_cls != original_cls:
            label += f" (was {original_cls})"
        
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_bgr, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame_bgr

def _read_frame_annotated_v8(video_name, frame_id, question_id):
    video_path = _find_video_path(video_name, _video_root_global)
    if not video_path: return None
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok: return None

    frame = _draw_detections_v8(frame, frame_id, video_name, question_id)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def _make_thumbnail(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h) if w > 0 and h > 0 else 0
    if scale == 0: return np.zeros((max_h, max_w, 3), dtype=np.uint8)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    y0, x0 = (max_h - nh) // 2, (max_w - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def _draw_tile_label(img, text):
    h, w = img.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    bar_h = th + 8
    y1 = h - bar_h if LABEL_POS == "bottom" else 0; y2 = h if LABEL_POS == "bottom" else bar_h
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, LABEL_ALPHA, img, 1 - LABEL_ALPHA, 0, dst=img)
    cv2.putText(img, text, (5, y2 - 5), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img

def render_single_thumbnail_v8(args):
    """
    Công việc chính của một worker: render một thumbnail duy nhất.
    Trả về ảnh thumbnail (RGB) hoặc một ảnh đen nếu có lỗi.
    """
    video_name, frame_id, question_id = args
    try:
        # Gọi các hàm helper đã được định nghĩa (với dấu gạch dưới)
        img = _read_frame_annotated_v8(video_name, frame_id, question_id)
        thumb = _make_thumbnail(img if img is not None else np.zeros((THUMBNAIL_MAX_H, THUMBNAIL_MAX_W, 3), dtype=np.uint8), THUMBNAIL_MAX_W, THUMBNAIL_MAX_H)
        if SHOW_TILE_LABEL:
            thumb = _draw_tile_label(thumb, f"{Path(video_name).stem} | F{int(frame_id)}")
        return thumb
    except Exception as e:
        # In lỗi để debug nếu cần
        # print(f"Worker error on {video_name}, frame {frame_id}: {e}")
        return np.zeros((THUMBNAIL_MAX_H, THUMBNAIL_MAX_W, 3), dtype=np.uint8)

# --- 3. CÁC HÀM TIỆN ÍCH CHO VIỆC TỔNG HỢP (CHẠY TUẦN TỰ) ---
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

# --- 4. SCRIPT ĐIỀU PHỐI CHÍNH ---
def generate_visual_report_v8():
    """Hàm chính điều phối toàn bộ quá trình tạo báo cáo V8."""
    print("=== GENERATING V8 FINAL VISUAL REPORT (with Tracking Consistency) ===")
    
    if not SUBMISSION_FILE_PATH.exists(): raise FileNotFoundError(f"Submission file not found: {SUBMISSION_FILE_PATH}")
    if not DB_PATH.exists(): raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    with open(SUBMISSION_FILE_PATH, 'r') as f: submission = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_per_sheet = IMAGES_PER_ROW * ROWS_PER_SHEET

    for q_id, results in submission.items():
        desc = QUERY_DESCRIPTIONS.get(str(q_id), f"Question {q_id}")
        print("\n" + "="*80)
        print(f"[Q{q_id}] {desc}")

        frames_to_process = sorted([(v_name, int(fid)) for v_name, f_list in (results or {}).items() for fid in f_list])
        num_total = len(frames_to_process)
        if num_total == 0:
            print("→ Không có kết quả nào để hiển thị."); continue
        
        n_pages = math.ceil(num_total / total_per_sheet)
        print(f"→ Tổng số {num_total} khung hình. Sẽ tạo {n_pages} trang báo cáo sử dụng {NUM_WORKERS} workers.")
        
        q_dir = OUTPUT_DIR / f"Q{q_id}"
        q_dir.mkdir(parents=True, exist_ok=True)
        
        job_args = [(video_name, fid, q_id) for (video_name, fid) in frames_to_process]
        thumbnails_all = [None] * num_total
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker_v8, initargs=(DB_PATH, VIDEO_SOURCE_DIR)) as executor:
            future_to_idx = {executor.submit(render_single_thumbnail_v8, arg): i for i, arg in enumerate(job_args)}
            for future in tqdm(as_completed(future_to_idx), total=num_total, desc=f"  -> Rendering Q{q_id}"):
                idx = future_to_idx[future]
                try: thumbnails_all[idx] = future.result()
                except Exception as exc:
                    print(f"Job {idx} generated an exception: {exc}")
                    thumbnails_all[idx] = np.zeros((THUMBNAIL_MAX_H, THUMBNAIL_MAX_W, 3), dtype=np.uint8)

        print("  -> Đang ghép ảnh và lưu báo cáo...")
        pages_paths = []
        for page_idx in range(n_pages):
            batch_thumbnails = thumbnails_all[page_idx*total_per_sheet : (page_idx+1)*total_per_sheet]
            while len(batch_thumbnails) < total_per_sheet:
                batch_thumbnails.append(np.zeros((THUMBNAIL_MAX_H, THUMBNAIL_MAX_W, 3), dtype=np.uint8))
            
            out_name = f"Q{q_id}_page_{page_idx+1:03d}.{SAVE_FORMAT}"
            out_path = q_dir / out_name
            save_contact_sheet(batch_thumbnails, out_path)
            pages_paths.append(out_path)
            
        write_html_index(q_dir, pages_paths, f"Visual Results for Q{q_id}: {desc}")
        print(f"→ Báo cáo đã được tạo tại: {q_dir/'index.html'}")

    print("\n" + "="*80)
    print("✓ HOÀN TẤT. Mở các file index.html trong thư mục:", str(OUTPUT_DIR))

if __name__ == "__main__":
    generate_visual_report_v8()