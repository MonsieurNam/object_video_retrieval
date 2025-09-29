import pandas as pd
from pathlib import Path
import warnings
import cv2
import matplotlib.pyplot as plt
import random
import glob

warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)

# --- 1. POST-PROCESSING UTILITY ---

def fill_gaps(frame_list: list, max_gap_size: int = 15) -> list:
    if len(frame_list) < 2:
        return frame_list
    
    filled_list = []
    
    # Sử dụng set để thêm và kiểm tra sự tồn tại nhanh hơn
    frame_set = set(frame_list)
    min_frame, max_frame = frame_list[0], frame_list[-1]
    
    for frame_num in range(min_frame, max_frame + 1):
        if frame_num in frame_set:
            filled_list.append(frame_num)
        else:
            # Tìm frame trước và sau gần nhất trong set
            prev_in_set = max([f for f in frame_set if f < frame_num], default=None)
            next_in_set = min([f for f in frame_set if f > frame_num], default=None)

            if prev_in_set is not None and next_in_set is not None:
                gap = next_in_set - prev_in_set
                if gap <= max_gap_size + 1:
                    filled_list.append(frame_num)

    return filled_list


def apply_post_processing(result_dict: dict, max_gap: int = 20) -> dict:
    if not result_dict: return {}
    
    processed_dict = {}
    for video, frames in result_dict.items():
        if frames:
            sorted_frames = sorted(frames)
            processed_dict[video] = fill_gaps(sorted_frames, max_gap_size=max_gap)
        else:
            processed_dict[video] = []
            
    return processed_dict

# --- 2. V4 QUERY FUNCTIONS ---

def query_1(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[df['confidence'] >= conf]
    frames_with_person = reliable_df[reliable_df['class_name'] == 'person'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_moto = reliable_df[reliable_df['class_name'] == 'motorcycle'][['video_name', 'frame_id']].drop_duplicates()
    result_df = pd.merge(frames_with_person, frames_with_moto, on=['video_name', 'frame_id'])
    
    if result_df.empty: return {}
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict)

def query_2(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[df['confidence'] >= conf]
    frames_with_person = reliable_df[reliable_df['class_name'] == 'person'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_bicycle = reliable_df[reliable_df['class_name'] == 'bicycle'][['video_name', 'frame_id']].drop_duplicates()
    result_df = pd.merge(frames_with_person, frames_with_bicycle, on=['video_name', 'frame_id'])
    
    if result_df.empty: return {}
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict)

def query_3(df: pd.DataFrame, conf: float) -> dict:
    result_df = df[(df['class_name'] == 'car') & (df['confidence'] >= conf)]
    
    if result_df.empty: return {}
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict)

def query_4(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[df['confidence'] >= conf]
    frames_with_person = reliable_df[reliable_df['class_name'] == 'person'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_bicycle = reliable_df[reliable_df['class_name'] == 'bicycle'][['video_name', 'frame_id']].drop_duplicates()
    result_df = pd.merge(frames_with_person, frames_with_bicycle, on=['video_name', 'frame_id'])
    
    if result_df.empty: return {}
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict)

def query_5(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[df['confidence'] >= conf]
    frames_with_person = reliable_df[reliable_df['class_name'] == 'person'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_moto = reliable_df[reliable_df['class_name'] == 'motorcycle'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_car = reliable_df[reliable_df['class_name'] == 'car'][['video_name', 'frame_id']].drop_duplicates()
    
    result_df = pd.merge(frames_with_person, frames_with_moto, on=['video_name', 'frame_id'])
    result_df = pd.merge(result_df, frames_with_car, on=['video_name', 'frame_id'])
    
    if result_df.empty: return {}
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict)

def query_6(df: pd.DataFrame, conf: float) -> dict:
    counting_conf = max(conf, 0.45)
    reliable_df = df[(df['class_name'] == 'person') & (df['confidence'] >= counting_conf)]
    counts = reliable_df.groupby(['video_name', 'frame_id']).size().reset_index(name='count')
    result_frames = counts[counts['count'] > 1]
    
    if result_frames.empty: return {}
    grouped = result_frames.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict)

def query_7(df: pd.DataFrame, conf: float) -> dict:
    counting_conf = max(conf, 0.45)
    reliable_df = df[(df['class_name'] == 'motorcycle') & (df['confidence'] >= counting_conf)]
    counts = reliable_df.groupby(['video_name', 'frame_id']).size().reset_index(name='count')
    result_frames = counts[counts['count'] > 1]
    
    if result_frames.empty: return {}
    grouped = result_frames.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict)

def query_8(df: pd.DataFrame, conf: float) -> dict:
    strict_counting_conf = max(conf, 0.7)
    reliable_df = df[df['confidence'] >= strict_counting_conf]
    counts = reliable_df.groupby(['video_name', 'frame_id', 'class_name']).size().unstack(fill_value=0)
    
    all_known_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    for cls in all_known_classes:
        if cls not in counts.columns:
            counts[cls] = 0
            
    other_classes = list(set(all_known_classes) - {'person'})
    counts['other_count'] = counts[other_classes].sum(axis=1)
    
    result_frames = counts[(counts['person'] == 3) & (counts['other_count'] == 0)]
    
    if result_frames.empty: return {}
    grouped = result_frames.reset_index().groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: [int(i) for i in lst]).to_dict()
    return apply_post_processing(result_dict, max_gap=5) # Smaller gap for strict counting

# --- TESTING BLOCK ---
if __name__ == "__main__":
    DB_PATH = Path('./evidence_database_v4_high_quality.feather')
    VIDEO_SOURCE_DIR = Path('./Video_vong_loai/')

    if not DB_PATH.exists():
        print(f"ERROR: Database file not found at '{DB_PATH}'. Run Phase 1 first.")
    else:
        print("Loading V4 database for testing...")
        test_df = pd.read_feather(DB_PATH)
        print("Load successful!")
        
        TEST_CONF = 0.35 
        print(f"Using base confidence threshold for testing: {TEST_CONF}")
        
        queries_to_test = {
            "Q1: Người & Xe máy": query_1,
            "Q2: Người & Xe đạp": query_2,
            "Q3: Xe ô tô": query_3,
            "Q4: >=1 Người & >=1 Xe đạp": query_4,
            "Q5: Người & Xe máy & Xe ô tô": query_5,
            "Q6: >1 Người": query_6,
            "Q7: >1 Xe máy": query_7,
            "Q8: Chỉ có 3 Người": query_8,
        }
        
        for name, func in queries_to_test.items():
            print(f"\n--- Testing: {name} ---")
            try:
                result = func(test_df, TEST_CONF)
                if not result:
                    print("=> No results found.")
                else:
                    print(f"=> Found results in {len(result)} videos.")
                    first_video = list(result.keys())[0]
                    first_video_frames = result[first_video]
                    print(f"   Example: Video '{first_video}' has {len(first_video_frames)} frames. Sample: {first_video_frames[:5]}")
            except Exception as e:
                print(f"!!! ERROR during query execution: {e}")

        print("\n--- TESTING COMPLETE ---")