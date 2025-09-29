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

# --- 1. CONFIGURATION ---

SUBMISSION_FILE_PATH = Path('./AI25-15.json')
DB_PATH = Path('./evidence_database_v4_high_quality.feather')
VIDEO_SOURCE_DIR = Path('./Video_vong_loai/')
OUTPUT_DIR = Path('./visual_report_v4/')

# --- PARAMETERS (MUST SYNC WITH query_library_v4.py) ---
BASE_CONF = 0.35
# Note: SAM_IOU is not directly used here as the V4 database already pre-filtered based on it.
# We only need the confidence thresholds to replicate the final filtering logic.
COUNTING_CONF = 0.45
STRICT_COUNTING_CONF = 0.7

# --- REPORT CONFIG ---
IMAGES_PER_ROW = 5
ROWS_PER_SHEET = 8
THUMBNAIL_MAX_W = 480
THUMBNAIL_MAX_H = 270
SAVE_FORMAT = 'webp'
WEBP_QUALITY = 85
LABEL_POS = "bottom"
LABEL_ALPHA = 0.7
MAX_FRAMES_TO_DISPLAY = 120 # Limit frames per query to avoid overly large reports

QUERY_DESCRIPTIONS = {
    "1": "Người VÀ Xe gắn máy", "2": "Người VÀ Xe đạp", "3": "Xe ô tô",
    "4": ">=1 Người VÀ >=1 Xe đạp", "5": "Người, Xe máy, VÀ Xe ô tô",
    "6": "Nhiều hơn 1 Người", "7": "Nhiều hơn 1 Xe máy", "8": "CHỈ có 3 Người"
}

# --- 2. UTILITY CLASSES AND FUNCTIONS ---

_video_path_cache = {}
def find_video_path(video_name: str, root: Path) -> Path | None:
    if video_name in _video_path_cache: return _video_path_cache[video_name]
    try:
        path = next(root.rglob(f"**/{video_name}"))
        _video_path_cache[video_name] = path
        return path
    except StopIteration:
        _video_path_cache[video_name] = None
        return None

class VideoReaderPool:
    def __init__(self, max_open=8):
        self.max_open = max_open
        self.pool = {}
    def get(self, video_path: Path):
        key = str(video_path)
        if key in self.pool:
            cap = self.pool.pop(key)
            self.pool[key] = cap
            return cap
        if len(self.pool) >= self.max_open:
            oldest_key = next(iter(self.pool))
            old_cap = self.pool.pop(oldest_key)
            try: old_cap.release()
            except: pass
        cap = cv2.VideoCapture(str(video_path))
        self.pool[key] = cap
        return cap
    def close_all(self):
        for cap in self.pool.values():
            try: cap.release()
            except: pass
        self.pool.clear()

def draw_detections_v4(frame_bgr, det_rows, question_id):
    q_targets = QUERY_DESCRIPTIONS.get(question_id, "").split(" VÀ ")
    q_targets = [s.split(" ")[-1].lower() for s in q_targets] # Heuristic to get target classes
    
    for _, det in det_rows.iterrows():
        is_valid_to_draw = False
        conf, cls_name = det['confidence'], det['class_name']
        val_method = det.get('validation_method', 'yolo_high_conf')

        if question_id in ["1", "2", "3", "4", "5"]:
            if conf >= BASE_CONF: is_valid_to_draw = True
        elif question_id in ["6", "7"]:
            if conf >= COUNTING_CONF: is_valid_to_draw = True
        elif question_id == "8":
            if conf >= STRICT_COUNTING_CONF: is_valid_to_draw = True
        
        if is_valid_to_draw:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            is_target = any(target in cls_name for target in q_targets)
            color = (36, 255, 12) if is_target else (255, 100, 100)
            thickness = 2 if is_target else 1
            
            label = f"{cls_name}:{conf:.2f}"
            if val_method == 'sam_clip_validated':
                label += " (S+C)"
            
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame_bgr, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return frame_bgr

def read_frame_annotated_v4(pool, df_evi_idx, video_name, frame_id, question_id):
    video_path = find_video_path(video_name, VIDEO_SOURCE_DIR)
    if not video_path: return None

    cap = pool.get(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok: return None

    key = (video_name, frame_id)
    det_rows = df_evi_idx.get(key)
    if det_rows is not None and not det_rows.empty:
        frame = draw_detections_v4(frame, det_rows, question_id)
        
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def make_thumbnail(img_rgb, max_w, max_h):
    h, w = img_rgb.shape[:2]
    scale = min(max_w / w, max_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    y0, x0 = (max_h - nh) // 2, (max_w - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def draw_tile_label(img_rgb, text):
    h, w = img_rgb.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    
    bar_h = th + 8
    y1 = h - bar_h if LABEL_POS == "bottom" else 0
    y2 = h if LABEL_POS == "bottom" else bar_h
    
    overlay = img_rgb.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, LABEL_ALPHA, img_rgb, 1 - LABEL_ALPHA, 0, dst=img_rgb)
    
    cv2.putText(img_rgb, text, (5, y2 - 5), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img_rgb

def save_contact_sheet(thumbnails, out_path):
    if not thumbnails: return
    sheet = np.vstack([np.hstack(thumbnails[i:i+IMAGES_PER_ROW]) for i in range(0, len(thumbnails), IMAGES_PER_ROW)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_FORMAT == 'webp':
        cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_WEBP_QUALITY), WEBP_QUALITY])
    else:
        cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

def write_html_index(q_dir, pages, title):
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
    html += "<style>body{font-family:sans-serif;background-color:#222;color:#eee} h1,h3{text-align:center} img{display:block;margin:20px auto;max-width:100%;height:auto;border:1px solid #444}</style>"
    html += f"</head><body><h1>{title}</h1>"
    for p in pages:
        html += f"<h3>{Path(p).name}</h3><img loading='lazy' src='{Path(p).name}'>"
    html += "</body></html>"
    (q_dir / "index.html").write_text(html, encoding="utf-8")

# --- 3. MAIN ORCHESTRATOR SCRIPT ---

def generate_visual_report_v4():
    print("=== GENERATING V4 VISUAL REPORT ===")
    
    if not SUBMISSION_FILE_PATH.exists(): raise FileNotFoundError(f"Submission file not found: {SUBMISSION_FILE_PATH}")
    if not DB_PATH.exists(): raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    print("Loading database (this may take a moment)...")
    df = pd.read_feather(DB_PATH)
    print("Indexing database for fast access...")
    df_evi_idx = {k: v for k, v in df.groupby(['video_name', 'frame_id'])}
    print("Loading submission file...")
    with open(SUBMISSION_FILE_PATH, 'r') as f:
        submission = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_per_sheet = IMAGES_PER_ROW * ROWS_PER_SHEET
    pool = VideoReaderPool(max_open=8)

    try:
        for q_id, results in submission.items():
            desc = QUERY_DESCRIPTIONS.get(str(q_id), f"Question {q_id}")
            print("\n" + "="*80)
            print(f"[Q{q_id}] {desc}")

            frames_list = sorted([(video_name, int(fid)) for video_name, frame_list in (results or {}).items() for fid in frame_list])
            num = len(frames_list)
            print(f"→ Total result frames: {num}")
            if num == 0: continue

            display_frames = frames_list
            if MAX_FRAMES_TO_DISPLAY is not None and num > MAX_FRAMES_TO_DISPLAY:
                print(f"  (Warning: Too many results. Displaying a random sample of {MAX_FRAMES_TO_DISPLAY})")
                display_frames = random.sample(frames_list, MAX_FRAMES_TO_DISPLAY)
            
            num_to_display = len(display_frames)
            if num_to_display == 0: continue

            q_dir = OUTPUT_DIR / f"Q{q_id}"
            q_dir.mkdir(parents=True, exist_ok=True)
            pages_paths = []
            n_pages = math.ceil(num_to_display / total_per_sheet)
            
            pbar = tqdm(total=num_to_display, desc=f"  -> Rendering Q{q_id}", ncols=100)
            
            for page_idx in range(n_pages):
                batch = display_frames[page_idx*total_per_sheet : (page_idx+1)*total_per_sheet]
                thumbnails = []
                for (video_name, fid) in batch:
                    img = read_frame_annotated_v4(pool, df_evi_idx, video_name, fid, q_id)
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
            print(f"→ Report generated: {q_dir/'index.html'}")

    finally:
        pool.close_all()

    print("\n" + "="*80)
    print("✓ ALL VISUAL REPORTS GENERATED. Open the index.html files in the output directory:", str(OUTPUT_DIR))

if __name__ == "__main__":
    generate_visual_report_v4()