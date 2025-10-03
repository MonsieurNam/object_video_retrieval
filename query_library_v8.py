import pandas as pd
from pathlib import Path
import warnings

# --- 1. HÀM TIỀN XỬ LÝ QUAN TRỌNG NHẤT ---

def preprocess_df_with_voting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thực hiện "bầu cử đa số" để quyết định một danh tính nhất quán (consistent_class_name)
    cho mỗi track_id.
    """
    print("--- Bắt đầu quá trình bầu cử danh tính cho các track ---")
    
    # Tính toán class_name xuất hiện nhiều nhất cho mỗi track_id
    # agg(lambda x: x.mode()[0]) sẽ lấy giá trị mode (phần tử xuất hiện nhiều nhất)
    # Nếu có nhiều mode, nó sẽ lấy cái đầu tiên.
    track_class_map = df.groupby('track_id')['class_name'].agg(lambda x: x.mode()[0])
    
    print(f"Đã xác định danh tính cho {len(track_class_map)} tracks.")
    
    # Sử dụng .map() để gán lại danh tính một cách hiệu quả
    df['consistent_class_name'] = df['track_id'].map(track_class_map)
    
    # (Tùy chọn) In ra số lượng nhãn đã được sửa
    changed_labels = (df['class_name'] != df['consistent_class_name']).sum()
    print(f"Hoàn thành. {changed_labels:,} phát hiện đã được điều chỉnh danh tính.")
    
    return df

# --- 2. CÁC HÀM HẬU XỬ LÝ & TIỆN ÍCH ---

def fill_gaps(frame_list: list, max_gap_size: int = 15) -> list:
    """Lấp đầy các khoảng trống nhỏ trong danh sách frame liên tiếp."""
    if len(frame_list) < 2:
        return frame_list
    
    filled_list = []
    frame_set = set(frame_list)
    min_frame, max_frame = frame_list[0], frame_list[-1]
    
    for frame_num in range(min_frame, max_frame + 1):
        if frame_num in frame_set:
            filled_list.append(frame_num)
        else:
            prev_in_set = max((f for f in frame_set if f < frame_num), default=None)
            next_in_set = min((f for f in frame_set if f > frame_num), default=None)

            if prev_in_set is not None and next_in_set is not None:
                gap = next_in_set - prev_in_set
                if gap <= max_gap_size + 1:
                    filled_list.append(frame_num)

    return filled_list


def apply_post_processing(result_dict: dict, max_gap: int = 20) -> dict:
    """Áp dụng hậu xử lý lấp đầy khoảng trống cho kết quả cuối cùng."""
    if not result_dict: return {}
    
    processed_dict = {}
    for video, frames in result_dict.items():
        if frames:
            sorted_frames = sorted(frames)
            processed_dict[video] = fill_gaps(sorted_frames, max_gap_size=max_gap)
        else:
            processed_dict[video] = []
            
    return processed_dict

# --- 3. V8 QUERY FUNCTIONS (ĐÃ SỬA LỖI) ---
# Tất cả các hàm này giờ hoạt động trên cột 'consistent_class_name' và có logic đếm chính xác.

def query_1(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[df['confidence'] >= conf]
    frames_with_person = reliable_df[reliable_df['consistent_class_name'] == 'person'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_moto = reliable_df[reliable_df['consistent_class_name'] == 'motorcycle'][['video_name', 'frame_id']].drop_duplicates()
    result_df = pd.merge(frames_with_person, frames_with_moto, on=['video_name', 'frame_id'])
    
    if result_df.empty: return {}
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: sorted([int(i) for i in lst])).to_dict()
    return apply_post_processing(result_dict)

def query_2(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[df['confidence'] >= conf]
    frames_with_person = reliable_df[reliable_df['consistent_class_name'] == 'person'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_bicycle = reliable_df[reliable_df['consistent_class_name'] == 'bicycle'][['video_name', 'frame_id']].drop_duplicates()
    result_df = pd.merge(frames_with_person, frames_with_bicycle, on=['video_name', 'frame_id'])
    
    if result_df.empty: return {}
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: sorted([int(i) for i in lst])).to_dict()
    return apply_post_processing(result_dict)

def query_3(df: pd.DataFrame, conf: float) -> dict:
    result_df = df[(df['consistent_class_name'] == 'car') & (df['confidence'] >= conf)]
    if result_df.empty: return {}
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: sorted([int(i) for i in lst])).to_dict()
    return apply_post_processing(result_dict)

def query_4(df: pd.DataFrame, conf: float) -> dict:
    """Câu 4: ĐÚNG một người VÀ ĐÚNG một xe đạp."""
    counting_conf = max(conf, 0.50)
    reliable_df = df[df['confidence'] >= counting_conf]
    
    # --- SỬA LỖI LOGIC ĐẾM ---
    # Đếm số lượng track_id duy nhất cho mỗi class trên từng frame, thay vì đếm số dòng.
    counts = reliable_df.groupby(['video_name', 'frame_id', 'consistent_class_name'])['track_id'].nunique().unstack(fill_value=0)
    
    # Đảm bảo các cột cần thiết tồn tại để tránh lỗi
    if 'person' not in counts.columns: counts['person'] = 0
    if 'bicycle' not in counts.columns: counts['bicycle'] = 0

    # Logic lọc giờ đã chính xác
    result_frames = counts[(counts['person'] == 1) & (counts['bicycle'] == 1)]
    
    if result_frames.empty: return {}
    grouped = result_frames.reset_index().groupby('video_name')['frame_id'].unique().apply(list)
    
    # Với các câu hỏi đếm chính xác, không nên dùng post-processing để tránh tạo ra False Positive
    return grouped.apply(lambda lst: sorted([int(i) for i in lst])).to_dict()


def query_5(df: pd.DataFrame, conf: float) -> dict:
    # --- SỬA LỖI LOGIC ---
    # 1. Dùng '>=' thay vì '=='
    # 2. Dùng cột 'consistent_class_name' đã được sửa lỗi.
    reliable_df = df[df['confidence'] >= conf]
    frames_with_person = reliable_df[reliable_df['consistent_class_name'] == 'person'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_moto = reliable_df[reliable_df['consistent_class_name'] == 'motorcycle'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_car = reliable_df[reliable_df['consistent_class_name'] == 'car'][['video_name', 'frame_id']].drop_duplicates()
    
    result_df = pd.merge(frames_with_person, frames_with_moto, on=['video_name', 'frame_id'])
    result_df = pd.merge(result_df, frames_with_car, on=['video_name', 'frame_id'])
    
    if result_df.empty: return {}
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: sorted([int(i) for i in lst])).to_dict()
    return apply_post_processing(result_dict)

def query_6(df: pd.DataFrame, conf: float) -> dict:
    # Hàm này đã có logic đúng: đếm track_id duy nhất.
    reliable_df = df[(df['consistent_class_name'] == 'person') & (df['confidence'] >= conf)]
    counts = reliable_df.groupby(['video_name', 'frame_id'])['track_id'].nunique().reset_index(name='count')
    result_frames = counts[counts['count'] > 1]
    
    if result_frames.empty: return {}
    grouped = result_frames.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: sorted([int(i) for i in lst])).to_dict()
    return apply_post_processing(result_dict)

def query_7(df: pd.DataFrame, conf: float) -> dict:
    counting_conf = max(conf, 0.45)
    
    # --- SỬA LỖI LOGIC ---
    # 1. Dùng 'consistent_class_name'.
    # 2. Đếm số track_id duy nhất (nunique) thay vì đếm số dòng (size).
    reliable_df = df[(df['consistent_class_name'] == 'motorcycle') & (df['confidence'] >= counting_conf)]
    counts = reliable_df.groupby(['video_name', 'frame_id'])['track_id'].nunique().reset_index(name='count')
    result_frames = counts[counts['count'] > 1]
    
    if result_frames.empty: return {}
    grouped = result_frames.groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: sorted([int(i) for i in lst])).to_dict()
    return apply_post_processing(result_dict)

def query_8(df: pd.DataFrame, conf: float) -> dict:
    strict_counting_conf = max(conf, 0.65)
    reliable_df = df[df['confidence'] >= strict_counting_conf]
    
    # --- SỬA LỖI LOGIC ĐẾM ---
    # Đếm số lượng track_id duy nhất cho mỗi class trên từng frame.
    counts = reliable_df.groupby(['video_name', 'frame_id', 'consistent_class_name'])['track_id'].nunique().unstack(fill_value=0)
    
    all_known_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    for cls in all_known_classes:
        if cls not in counts.columns: counts[cls] = 0
            
    other_classes = list(set(all_known_classes) - {'person'})
    counts['other_count'] = counts[other_classes].sum(axis=1)
    
    # Logic lọc giờ đã chính xác
    result_frames = counts[(counts['person'] == 3) & (counts['other_count'] == 0)]
    
    if result_frames.empty: return {}
    grouped = result_frames.reset_index().groupby('video_name')['frame_id'].unique().apply(list)
    
    # Với các câu hỏi đếm chính xác, không nên dùng post-processing
    return grouped.apply(lambda lst: sorted([int(i) for i in lst])).to_dict()

# --- TESTING BLOCK ---
if __name__ == "__main__":
    # (Giữ nguyên khối testing để bạn có thể kiểm tra lại sau khi sửa)
    DB_PATH = Path('./deep_tracked_database_v8.feather')
    if not DB_PATH.exists():
        print(f"ERROR: Database file not found at '{DB_PATH}'. Run Phase 1.5 first.")
    else:
        print("Loading V8 database for testing...")
        test_df = pd.read_feather(DB_PATH)
        
        print("\n*** BƯỚC TIỀN XỬ LÝ QUAN TRỌNG ***")
        processed_df = preprocess_df_with_voting(test_df)
        
        print("\n--- Bắt đầu kiểm thử các hàm truy vấn trên dữ liệu đã xử lý ---")
        TEST_CONF = 0.25 
        
        queries_to_test = {
            "Q1: Người & Xe máy": query_1,
            "Q4: ĐÚNG 1 Người & 1 Xe đạp (Đã sửa)": query_4,
            "Q5: Người & Xe máy & Xe ô tô (Đã sửa)": query_5,
            "Q6: >1 Người": query_6,
            "Q7: >1 Xe máy (Đã sửa)": query_7,
            "Q8: Chỉ có 3 Người (Đã sửa)": query_8,
        }
        
        for name, func in queries_to_test.items():
            print(f"\n--- Testing: {name} ---")
            result = func(processed_df, TEST_CONF)
            if not result:
                print("=> No results found.")
            else:
                print(f"=> Found results in {len(result)} videos.")
                # In ra một vài kết quả mẫu
                sample_video = list(result.keys())[0]
                sample_frames = result[sample_video][:5]
                print(f"   Sample for '{sample_video}': {sample_frames}...")

        print("\n--- TESTING COMPLETE ---")