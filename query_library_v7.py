
import pandas as pd
from pathlib import Path
import warnings

# --- 1. POST-PROCESSING UTILITY (Giữ nguyên) ---

def fill_gaps(frame_list: list, max_gap_size: int = 15) -> list:
    if len(frame_list) < 2: return frame_list
    frame_set = set(frame_list)
    full_range = range(frame_list[0], frame_list[-1] + 1)
    filled_list = [f for f in full_range if f in frame_set or \
                   (min([abs(f - x) for x in frame_set if x < f] or [max_gap_size+2]) + \
                    min([abs(f - x) for x in frame_set if x > f] or [max_gap_size+2])) <= max_gap_size + 1]
    return filled_list

def apply_post_processing(result_dict: dict, max_gap: int = 20) -> dict:
    if not result_dict: return {}
    return {video: fill_gaps(sorted(frames), max_gap_size=max_gap) for video, frames in result_dict.items()}

# --- 2. V7 QUERY FUNCTIONS ---
# TẤT CẢ các hàm giờ đây sẽ sử dụng cột 'inferred_class_name'

def query_1(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[df['confidence'] >= conf]
    frames_with_person = reliable_df[reliable_df['inferred_class_name'] == 'person'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_moto = reliable_df[reliable_df['inferred_class_name'] == 'motorcycle'][['video_name', 'frame_id']].drop_duplicates()
    result_df = pd.merge(frames_with_person, frames_with_moto, on=['video_name', 'frame_id'])
    
    if result_df.empty: return {}
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict)

def query_2(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[df['confidence'] >= conf]
    frames_with_person = reliable_df[reliable_df['inferred_class_name'] == 'person'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_bicycle = reliable_df[reliable_df['inferred_class_name'] == 'bicycle'][['video_name', 'frame_id']].drop_duplicates()
    result_df = pd.merge(frames_with_person, frames_with_bicycle, on=['video_name', 'frame_id'])
    
    if result_df.empty: return {}
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict)

def query_3(df: pd.DataFrame, conf: float) -> dict:
    result_df = df[(df['inferred_class_name'] == 'car') & (df['confidence'] >= conf)]
    
    if result_df.empty: return {}
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict)

def query_4(df: pd.DataFrame, conf: float) -> dict:
    # Yêu cầu mới: >=1 người và >=1 xe đạp
    return query_2(df, conf) # Logic giống hệt Q2

def query_5(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[df['confidence'] > conf]
    frames_with_person = reliable_df[reliable_df['inferred_class_name'] == 'person'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_moto = reliable_df[reliable_df['inferred_class_name'] == 'motorcycle'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_car = reliable_df[reliable_df['inferred_class_name'] == 'car'][['video_name', 'frame_id']].drop_duplicates()
    
    result_df = pd.merge(frames_with_person, frames_with_moto, on=['video_name', 'frame_id'])
    result_df = pd.merge(result_df, frames_with_car, on=['video_name', 'frame_id'])
    
    if result_df.empty: return {}
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict)

def query_6(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[(df['inferred_class_name'] == 'person') & (df['confidence'] >= conf)]
    # Đếm số lượng track_id duy nhất trên mỗi frame, thay vì chỉ đếm số dòng
    counts = reliable_df.groupby(['video_name', 'frame_id'])['track_id'].nunique().reset_index(name='count')
    result_frames = counts[counts['count'] > 1]
    
    if result_frames.empty: return {}
    grouped = result_frames.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict)

def query_7(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[(df['inferred_class_name'] == 'motorcycle') & (df['confidence'] >= conf)]
    counts = reliable_df.groupby(['video_name', 'frame_id'])['track_id'].nunique().reset_index(name='count')
    result_frames = counts[counts['count'] > 1]
    
    if result_frames.empty: return {}
    grouped = result_frames.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict)

def query_8(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[df['confidence'] >= conf]
    # Đếm số track_id duy nhất cho mỗi class trên mỗi frame
    counts = reliable_df.groupby(['video_name', 'frame_id'])['inferred_class_name'].value_counts().unstack(fill_value=0)
    
    all_known_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    for cls in all_known_classes:
        if cls not in counts.columns:
            counts[cls] = 0
            
    other_classes = list(set(all_known_classes) - {'person'})
    counts['other_count'] = counts[other_classes].sum(axis=1)
    
    result_frames = counts[(counts['person'] == 3) & (counts['other_count'] == 0)] # Quay lại logic nghiêm ngặt
    
    if result_frames.empty: return {}
    grouped = result_frames.reset_index().groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return result_dict # Không áp dụng gap-filling cho câu hỏi chính xác này

# --- TESTING BLOCK ---
if __name__ == "__main__":
    DB_PATH = Path('./inference_graph_database_v7.feather')
    if not DB_PATH.exists():
        print(f"ERROR: Database file not found at '{DB_PATH}'. Run Phase 2 first.")
    else:
        print("Loading V7 database for testing...")
        test_df = pd.read_feather(DB_PATH)
        print("Load successful!")
        
        TEST_CONF = 0.3 