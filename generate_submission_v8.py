
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import sys
import time

# --- 1. CONFIGURATION ---

DB_PATH = Path('./deep_tracked_database_v8.feather')
TEAM_ID = "AI25-15"
SUBMISSION_FILE_PATH = Path(f'./{TEAM_ID}.json')

# *** FINAL TUNING PARAMETER ***
# Với dữ liệu đã được tracking và sửa lỗi, chúng ta có thể tự tin
# sử dụng một ngưỡng tin cậy thấp để tối đa hóa Recall.
FINAL_CONFIDENCE_THRESHOLD = 0.2

# --- 2. MAIN SCRIPT ---

def create_final_submission_v8():
    """Hàm chính điều phối toàn bộ quá trình tạo file nộp bài V8."""
    start_time = time.time()
    print("--- PHASE 3 (V8): GENERATING FINAL SUBMISSION FILE ---")
    
    # --- 2.1. Import và Sanity Checks ---
    try:
        from query_library_v8 import (
            preprocess_df_with_voting, # <<<< IMPORT HÀM TIỀN XỬ LÝ
            query_1, query_2, query_3, query_4, 
            query_5, query_6, query_7, query_8
        )
        print("Successfully imported V8 query library.")
    except ImportError:
        print("\n!!! ERROR: Could not find 'query_library_v8.py'. !!!")
        sys.exit(1)

    print(f"Loading deep tracked database from: {DB_PATH}")
    if not DB_PATH.exists():
        print(f"\n!!! ERROR: Database file not found at '{DB_PATH}'. !!!")
        sys.exit(1)
        
    df = pd.read_feather(DB_PATH)
    print(f"Database loaded successfully with {len(df):,} tracked detections.")
    
    # --- 2.2. TIỀN XỬ LÝ QUAN TRỌNG: BẦU CỬ DANH TÍNH ---
    df = preprocess_df_with_voting(df)
    
    # --- 2.3. Chạy Các Truy vấn ---
    print(f"\nApplying final base confidence threshold: {FINAL_CONFIDENCE_THRESHOLD}")
    final_submission = {}
    query_map = {
        "1": query_1, "2": query_2, "3": query_3, "4": query_4,
        "5": query_5, "6": query_6, "7": query_7, "8": query_8
    }
    
    for question_id, query_func in tqdm(query_map.items(), desc="Running all queries"):
        # Tất cả các hàm truy vấn giờ sẽ chạy trên DataFrame đã được xử lý
        result = query_func(df, FINAL_CONFIDENCE_THRESHOLD)
        final_submission[question_id] = result
        
    # --- 2.4. Lưu File Kết quả ---
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
    create_final_submission_v8()