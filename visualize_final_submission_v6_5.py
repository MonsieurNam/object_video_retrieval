import pandas as pd
import json
from pathlib import Path
import cv2
import numpy as np
import sys
import glob
import random
from tqdm import tqdm
import math
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# --- 1. CONFIGURATION (Cần kiểm tra và cập nhật) ---

# --- Đường dẫn ---
SUBMISSION_FILE_PATH = Path('./AI25-15_v6_5.json')
DB_PATH = Path('./tracked_database_v6_5.feather')
VIDEO_SOURCE_DIR = Path('./Video_vong_loai/')
OUTPUT_DIR = Path('./visual_report_v6_5_FINAL/')

# --- Tham số Logic (PHẢI ĐỒNG BỘ VỚI query_library_v6.py) ---
BASE_CONF = 0.3
COUNTING_CONF = 0.45
STRICT_COUNTING_CONF = 0.65

# --- Cấu hình Báo cáo ---
IMAGES_PER_ROW = 5
ROWS_PER_SHEET = 8
THUMBNAIL_MAX_W = 480
THUMBNAIL_MAX_H = 270
SAVE_FORMAT = 'webp'
WEBP_QUALITY = 85
SHOW_TILE_LABEL = True
LABEL_POS = "bottom"
LABEL_ALPHA = 0.7

# --- Cấu hình Song song hóa ---
try:
    NUM_WORKERS = max(1, os.cpu_count() - 2)
except:
    NUM_WORKERS = 4 # Giá trị an toàn

QUERY_DESCRIPTIONS = {
    "1": "Người VÀ Xe gắn máy", "2": "Người VÀ Xe đạp", "3": "Xe ô tô",
    "4": ">=1 Người VÀ >=1 Xe đạp", "5": "Người, Xe máy, VÀ Xe ô tô",
    "6": "Nhiều hơn 1 Người", "7": "Nhiều hơn 1 Xe máy", "8": "CHỈ có 3 Người"
}

# --- 2. HÀM WORKER VÀ CÁC HÀM TIỆN ÍCH LIÊN QUAN ---

# Các biến toàn cục sẽ được khởi tạo một lần cho mỗi tiến trình con
_df_idx_global = None
_video_root_global = None

def init_worker(db_path, video_root):
    """Hàm khởi tạo, tải và tiền xử lý CSDL một lần cho mỗi worker."""
    global _df_idx_global, _video_root_global
    cv2.setNumThreads(1)
    df = pd.read_feather(db_path)
    
    if 'track_id' in df.columns:
        track_class_map = df.groupby('track_id')['class_name'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        df['consistent_class_name'] = df['track_id'].map(track_class_map)
    else:
        df['consistent_class_name'] = df['class_name']
        
    _df_idx_global = {k: v for k, v in df.groupby(['video_name', 'frame_id'])}
    _video_root_global = video_root

def _find_video_path(video_name: str, root: Path) -> Path | None:
    """Hàm tìm video, được sử dụng bên trong worker."""
    try:
        p = root / video_name
        if p.exists(): return p
        return next(root.rglob(f"**/{video_name}"))
    except StopIteration:
        return None

def _draw_detections_v6(frame_bgr, det_rows, question_id):
    """Hàm vẽ bbox thông minh phiên bản V6, hiển thị cả sự thay đổi của tracking."""
    q_targets_map = {
        "1": ['person', 'motorcycle'], "2": ['person', 'bicycle'], "3": ['car'],
        "4": ['person', 'bicycle'], "5": ['person', 'motorcycle', 'car'],
        "6": ['person'], "7": ['motorcycle'], "8": ['person']
    }
    q_targets = q_targets_map.get(question_id, [])
    
    for _, det in det_rows.iterrows():
        is_valid_to_draw = False
        conf = det['confidence']
        consistent_cls = det['consistent_class_name']
        original_cls = det['class_name']

        if question_id in "12345":
            if conf >= BASE_CONF: is_valid_to_draw = True
        elif question_id in "67":
            if conf >= COUNTING_CONF: is_valid_to_draw = True
        elif question_id == "8":
            if conf >= STRICT_COUNTING_CONF: is_valid_to_draw = True
        
        if is_valid_to_draw:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            is_target = consistent_cls in q_targets
            
            if consistent_cls != original_cls:
                color = (255, 0, 255) # Màu tím cho các box được sửa lỗi
            else:
                color = (36, 255, 12) if is_target else (255, 100, 100)
            
            thickness = 2 if is_target or (consistent_cls != original_cls) else 1
            label = f"{consistent_cls}:{conf:.2f}"
            if consistent_cls != original_cls:
                label += f" (was {original_cls})"
            
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame_bgr, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return frame_bgr

def _read_frame_annotated_v6(video_name, frame_id, question_id):
    """Hàm đọc và chú thích frame, được dùng bởi worker."""
    global _video_root_global, _df_idx_global
    video_path = _find_video_path(video_name, _video_root_global)
    if not video_path: return None
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok: return None

    rows = _df_idx_global.get((video_name, frame_id))
    if rows is not None and not rows.empty:
        frame = _draw_detections_v6(frame, rows, question_id)
        
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def _make_thumbnail(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h) if w > 0 and h > 0 else 0
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
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
    y1 = h - bar_h if LABEL_POS == "bottom" else 0
    y2 = h if LABEL_POS == "bottom" else bar_h
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, LABEL_ALPHA, img, 1 - LABEL_ALPHA, 0, dst=img)
    cv2.putText(img, text, (5, y2 - 5), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img

def render_single_thumbnail(args):
    """Công việc chính của một worker: render một thumbnail."""
    video_name, frame_id, question_id = args
    try:
        img = _read_frame_annotated_v6(video_name, frame_id, question_id)
        thumb = _make_thumbnail(img if img is not None else np.zeros((THUMBNAIL_MAX_H, THUMBNAIL_MAX_W, 3), dtype=np.uint8), THUMBNAIL_MAX_W, THUMBNAIL_MAX_H)
        if SHOW_TILE_LABEL:
            thumb = _draw_tile_label(thumb, f"{Path(video_name).stem} | F{int(frame_id)}")
        return thumb
    except Exception:
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
    for p in pages:
        html += f"<h3>{Path(p).name}</h3><img loading='lazy' src='{Path(p).name}'>"
    html += "</body></html>"
    (q_dir / "index.html").write_text(html, encoding="utf-8")

# --- 4. SCRIPT ĐIỀU PHỐI CHÍNH ---
def generate_visual_report_v6():
    print("=== TẠO BÁO CÁO TRỰC QUAN V6 (Tracking Verified) ===")
    
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
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker, initargs=(DB_PATH, VIDEO_SOURCE_DIR)) as executor:
            future_to_idx = {executor.submit(render_single_thumbnail, arg): i for i, arg in enumerate(job_args)}
            for future in tqdm(as_completed(future_to_idx), total=num_total, desc=f"  -> Rendering Q{q_id}"):
                idx = future_to_idx[future]
                try: thumbnails_all[idx] = future.result()
                except Exception as exc:
                    print(f"Job {idx} generated an exception: {exc}")
                    thumbnails_all[idx] = np.zeros((THUMBNAIL_MAX_H, THUMBNAIL_MAX_W, 3), dtype=np.uint8)

        print("  -> Đang ghép ảnh và lưu báo cáo...")
        pages_paths = []
        for page_idx in range(n_pages):
            batch = thumbnails_all[page_idx*total_per_sheet : (page_idx+1)*total_per_sheet]
            while len(batch) < total_per_sheet:
                batch.append(np.zeros((THUMBNAIL_MAX_H, THUMBNAIL_MAX_W, 3), dtype=np.uint8))
            
            out_name = f"Q{q_id}_page_{page_idx+1:03d}.{SAVE_FORMAT}"
            out_path = q_dir / out_name
            save_contact_sheet(batch, out_path)
            pages_paths.append(out_path)

        write_html_index(q_dir, pages_paths, f"Visual Results for Q{q_id}: {desc}")
        print(f"→ Báo cáo đã được tạo tại: {q_dir/'index.html'}")

    print("\n" + "="*80)
    print("✓ HOÀN TẤT. Mở các file index.html trong thư mục:", str(OUTPUT_DIR))

if __name__ == "__main__":
    generate_visual_report_v6()