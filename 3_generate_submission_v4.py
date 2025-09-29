import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import sys
import time

# --- 1. CONFIGURATION ---

DB_PATH = Path('./evidence_database_v4_high_quality.feather')
TEAM_ID = "AI25-15"
SUBMISSION_FILE_PATH = Path(f'./{TEAM_ID}.json')

# *** FINAL TUNING PARAMETER ***
# This is the base confidence threshold.
# The query functions will internally use stricter thresholds if needed.
# Based on our analysis, a value between 0.3 and 0.4 should be a good starting point.
FINAL_CONFIDENCE_THRESHOLD = 0.35

# --- 2. MAIN SCRIPT ---

def create_final_submission_v4():
    """
    Main function to orchestrate the final submission file generation.
    """
    start_time = time.time()
    print("--- PHASE 3 (V4): GENERATING FINAL SUBMISSION FILE ---")
    
    # --- 2.1. Import and Sanity Checks ---
    try:
        from query_library_v4 import (
            query_1, query_2, query_3, query_4, 
            query_5, query_6, query_7, query_8
        )
        print("Successfully imported V4 query library.")
    except ImportError:
        print("\n!!! ERROR: Could not find 'query_library_v4.py'. !!!")
        print("Please ensure the file from Phase 2 is in the same directory and named correctly.")
        sys.exit(1)

    print(f"Loading high-quality database from: {DB_PATH}")
    if not DB_PATH.exists():
        print(f"\n!!! ERROR: Database file not found at '{DB_PATH}'. !!!")
        print("Please run '1_build_database_v4.py' first.")
        sys.exit(1)
        
    df = pd.read_feather(DB_PATH)
    print(f"Database loaded successfully with {len(df):,} high-quality detections.")
    
    # --- 2.2. Run Queries ---
    print(f"\nApplying final base confidence threshold: {FINAL_CONFIDENCE_THRESHOLD}")
    
    final_submission = {}
    
    query_map = {
        "1": query_1, "2": query_2, "3": query_3, "4": query_4,
        "5": query_5, "6": query_6, "7": query_7, "8": query_8
    }
    
    for question_id, query_func in tqdm(query_map.items(), desc="Running all queries"):
        result = query_func(df, FINAL_CONFIDENCE_THRESHOLD)
        final_submission[question_id] = result
        
    # --- 2.3. Save Final Result ---
    print(f"\nSaving submission file to: {SUBMISSION_FILE_PATH}")
    
    try:
        with open(SUBMISSION_FILE_PATH, 'w', encoding='utf8') as f:
            json.dump(final_submission, f, indent=4)
        
        end_time = time.time()
        print("\n--- COMPLETE! ---")
        print(f"Submission file '{SUBMISSION_FILE_PATH}' created successfully.")
        print(f"Total runtime for Phase 3: {end_time - start_time:.2f} seconds.")
        print("It is highly recommended to run the visualization script (Phase 4) to verify the results.")
    except Exception as e:
        print(f"\n!!! ERROR while saving JSON file: {e}")

if __name__ == "__main__":
    create_final_submission_v4()