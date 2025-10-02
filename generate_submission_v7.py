import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import sys
import time
import warnings

# --- 1. CONFIGURATION ---
DB_PATH = Path('./inference_graph_database_v7_new.feather')
TEAM_ID = "AI25-15"
SUBMISSION_FILE_PATH = Path(f'./{TEAM_ID}.json')

# *** FINAL TUNING PARAMETER ***
# Vì CSDL V7 đã bao gồm các frame được nội suy (có conf=0.0),
# chúng ta chỉ cần một ngưỡng rất nhỏ để loại bỏ các phát hiện yếu không mong muốn,
# nhưng vẫn giữ lại các frame đã được suy luận.
# Ngưỡng >= 0.0 sẽ lấy tất cả.
FINAL_CONFIDENCE_THRESHOLD = 0.0

# --- 2. V7 QUERY LIBRARY (Tích hợp trực tiếp vì quá đơn giản) ---

def to_submission_format(df: pd.DataFrame) -> dict:
    """Hàm tiện ích chuyển DataFrame kết quả sang định dạng dict."""
    if df.empty:
        return {}
    # Sắp xếp frame_id trước khi nhóm
    df = df.sort_values('frame_id')
    grouped = df.groupby('video_name')['frame_id'].unique().apply(list)
    return grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()

def query_v7_base_AND(df: pd.DataFrame, classes: list) -> dict:
    """Hàm cơ sở cho các câu hỏi AND (có chứa A và B và C...)."""
    # Lấy danh sách các frame duy nhất cho mỗi class
    frames_per_class = [
        df[df['class_name'] == cls][['video_name', 'frame_id']].drop_duplicates()
        for cls in classes
    ]
    # Merge tuần tự để tìm giao điểm
    result_df = frames_per_class[0]
    for i in range(1, len(frames_per_class)):
        result_df = pd.merge(result_df, frames_per_class[i], on=['video_name', 'frame_id'])
    return to_submission_format(result_df)

def query_v7_base_COUNT(df: pd.DataFrame, class_name: str, condition) -> dict:
    """Hàm cơ sở cho các câu hỏi đếm."""
    df_class = df[df['class_name'] == class_name]
    counts = df_class.groupby(['video_name', 'frame_id']).size().reset_index(name='count')
    result_frames = counts[condition(counts['count'])]
    return to_submission_format(result_frames)

def query_v7_q8(df: pd.DataFrame) -> dict:
    """Câu 8 (V7): CHỈ chứa ba người."""
    counts = df.groupby(['video_name', 'frame_id', 'class_name']).size().unstack(fill_value=0)
    # Lấy tất cả các cột class có trong CSDL
    all_cols = list(counts.columns)
    other_cols = [col for col in all_cols if col != 'person']
    
    if 'person' not in counts.columns: counts['person'] = 0
    if not other_cols: # Nếu chỉ có cột 'person'
        counts['other_count'] = 0
    else:
        counts['other_count'] = counts[other_cols].sum(axis=1)
        
    result_frames = counts[(counts['person'] == 3) & (counts['other_count'] == 0)]
    return to_submission_format(result_frames.reset_index())

# --- 3. MAIN SCRIPT ---
def create_final_submission_v7():
    start_time = time.time()
    print("--- PHASE 3 (V7): GENERATING FINAL SUBMISSION FILE ---")

    print(f"Loading inferred database from: {DB_PATH}")
    if not DB_PATH.exists():
        print(f"\n!!! ERROR: Inferred database not found at '{DB_PATH}'. !!!")
        sys.exit(1)
        
    df = pd.read_feather(DB_PATH)
    # Chỉ giữ lại các phát hiện "thật" hoặc đã được suy luận
    df_filtered = df[df['confidence'] >= FINAL_CONFIDENCE_THRESHOLD]
    print(f"Database loaded. Using {len(df_filtered):,} consistent detections.")
    
    final_submission = {}
    
    print("\nRunning all queries on the inferred database...")
    pbar = tqdm(total=8, desc="Running Queries")
    
    final_submission["1"] = query_v7_base_AND(df_filtered, ['person', 'motorcycle']); pbar.update(1)
    final_submission["2"] = query_v7_base_AND(df_filtered, ['person', 'bicycle']); pbar.update(1)
    final_submission["3"] = to_submission_format(df_filtered[df_filtered['class_name'] == 'car']); pbar.update(1)
    final_submission["4"] = query_v7_base_AND(df_filtered, ['person', 'bicycle']); pbar.update(1) # Logic Q4 mới giống Q2
    final_submission["5"] = query_v7_base_AND(df_filtered, ['person', 'motorcycle', 'car']); pbar.update(1)
    final_submission["6"] = query_v7_base_COUNT(df_filtered, 'person', lambda c: c > 1); pbar.update(1)
    final_submission["7"] = query_v7_base_COUNT(df_filtered, 'motorcycle', lambda c: c > 1); pbar.update(1)
    final_submission["8"] = query_v7_q8(df_filtered); pbar.update(1)
    
    pbar.close()
    
    print(f"\nSaving submission file to: {SUBMISSION_FILE_PATH}")
    try:
        with open(SUBMISSION_FILE_PATH, 'w', encoding='utf8') as f:
            json.dump(final_submission, f, indent=4)
        
        end_time = time.time()
        print("\n--- COMPLETE! ---")
        print(f"Submission file '{SUBMISSION_FILE_PATH}' created successfully.")
        print(f"Total runtime for Phase 3: {end_time - start_time:.2f} seconds.")
        print("Proceed to Phase 4 for final visual verification.")
    except Exception as e:
        print(f"\n!!! ERROR while saving JSON file: {e}")

if __name__ == "__main__":
    create_final_submission_v7()