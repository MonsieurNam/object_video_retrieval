import pandas as pd
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import random
from tqdm import tqdm
import math
import gc
# --- 1. CONFIGURATION ---

# --- Đường dẫn ---
SUBMISSION_FILE_PATH = Path('./AI25-15.json') 
DB_PATH = Path('/root/object_video_retrieval/evidence_database_v5_expert.feather')
VIDEO_SOURCE_DIR = Path('./Video_vong_loai/')
OUTPUT_DIR = Path('./visual_report_v5_FULL/') # Đổi tên thư mục output để phân biệt

# --- Tham số Logic (PHẢI ĐỒNG BỘ VỚI FILE query_library_v4.py) ---
BASE_CONF = 0.35
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

# --- 2. CÁC LỚP VÀ HÀM TIỆN ÍCH (Đã hoàn thiện) ---

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

def draw_detections_v4(frame_bgr, det_rows, question_id):
    q_targets_map = {
        "1": ['person', 'motorcycle'], "2": ['person', 'bicycle'], "3": ['car'],
        "4": ['person', 'bicycle'], "5": ['person', 'motorcycle', 'car'],
        "6": ['person'], "7": ['motorcycle'], "8": ['person']
    }
    q_targets = q_targets_map.get(question_id, [])
    for _, det in det_rows.iterrows():
        is_valid = False
        conf, sam, cls = det['confidence'], det['sam_iou_score'], det['class_name']
        if question_id in "12345":
            if conf >= BASE_CONF or sam >= SAM_IOU: is_valid = True
        elif question_id in "67":
            if conf >= COUNTING_CONF: is_valid = True
        elif question_id == "8":
            if conf >= STRICT_COUNTING_CONF: is_valid = True
        if is_valid:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            is_target = cls in q_targets
            color = (36, 255, 12) if is_target else (255, 100, 100)
            thick = 2 if is_target else 1
            label = f"{cls}:{conf:.2f}"
            if det.get('validation_method') == 'sam_clip_validated': label += " (S+C)"
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thick)
            cv2.putText(frame_bgr, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return frame_bgr

def read_frame_annotated_v4(pool, df_idx, video_name, frame_id, question_id):
    video_path = find_video_path(video_name, VIDEO_SOURCE_DIR)
    if not video_path: return None
    cap = pool.get(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok: return None
    rows = df_idx.get((video_name, frame_id))
    if rows is not None and not rows.empty:
        frame = draw_detections_v4(frame, rows, question_id)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

# --- 3. SCRIPT ĐIỀU PHỐI CHÍNH ---

def generate_full_visual_report():
    print("=== TẠO BÁO CÁO TRỰC QUAN TOÀN DIỆN (V4) ===")
    
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
                    img = read_frame_annotated_v4(pool, df_idx, video_name, fid, q_id)
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
    generate_full_visual_report()