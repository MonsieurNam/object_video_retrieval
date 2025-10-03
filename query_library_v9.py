import pandas as pd
from pathlib import Path
import warnings

# --- 1. HÀM TIỀN XỬ LÝ NÂNG CẤP ---
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

def preprocess_df_v9(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thực hiện "bầu cử" và tính toán "độ tin cậy điều chỉnh theo track".
    """
    print("--- Bắt đầu Tiền xử lý V9: Bầu cử và Điều chỉnh Độ tin cậy ---")
    
    # Bước 1: Bầu cử danh tính (như V8)
    track_class_map = df.groupby('track_id')['class_name'].agg(lambda x: x.mode()[0])
    df['consistent_class_name'] = df['track_id'].map(track_class_map)
    
    # Bước 2: Tính toán các chỉ số của track
    track_stats = df.groupby('track_id')['confidence'].agg(['max', 'mean']).rename(columns={'max': 'track_max_conf', 'mean': 'track_avg_conf'})
    
    # Bước 3: Merge các chỉ số vào DataFrame chính
    df = df.merge(track_stats, on='track_id', how='left')
    
    # Bước 4: Tạo cột 'adjusted_confidence'
    # Trọng số: tin vào điểm cao nhất của track (0.4) và điểm hiện tại (0.6)
    df['adjusted_confidence'] = 0.6 * df['confidence'] + 0.4 * df['track_max_conf']
    
    changed_labels = (df['class_name'] != df['consistent_class_name']).sum()
    print(f"Hoàn thành. {changed_labels} nhãn đã được điều chỉnh. Cột 'adjusted_confidence' đã được tạo.")
    
    return df

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
  
def fill_gaps_for_q8(df_full: pd.DataFrame, seed_frames_df: pd.DataFrame) -> dict:
    """
    Điền khoảng trống thông minh DÀNH RIÊNG cho Q8.
    Nó kiểm tra lại trạng thái "có đúng 3 track person không" ở các frame trong khoảng trống.
    """
    if seed_frames_df.empty:
        return {}
        
    result_dict = {}
    for video, group in seed_frames_df.groupby('video_name'):
        seed_frames = sorted(group['frame_id'].unique())
        if not seed_frames: continue
            
        full_video_df = df_full[df_full['video_name'] == video]
        final_frames = set(seed_frames)
        
        for i in range(len(seed_frames) - 1):
            start_frame = seed_frames[i]
            end_frame = seed_frames[i+1]
            gap = end_frame - start_frame
            
            # Chỉ xem xét các khoảng trống nhỏ
            if 1 < gap <= 30: # max_gap = 30 frames
                for frame_in_gap in range(start_frame + 1, end_frame):
                    # Kiểm tra lại trạng thái ở frame trong khoảng trống
                    frame_data = full_video_df[full_video_df['frame_id'] == frame_in_gap]
                    person_tracks = frame_data[frame_data['consistent_class_name'] == 'person']['track_id'].nunique()
                    
                    if person_tracks == 3:
                        final_frames.add(frame_in_gap)
                        
        result_dict[video] = sorted([int(f) for f in final_frames])
        
    return result_dict

# --- 3. V9 QUERY FUNCTIONS ---
# Các hàm query từ 1 đến 7 sẽ được sửa để dùng 'adjusted_confidence'
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
    """
    Câu 4 (V8 - Sửa lỗi): CHỈ chứa ĐÚNG một người VÀ ĐÚNG một xe đạp.
    """
    # Sử dụng ngưỡng tin cậy cao hơn cho việc đếm chính xác
    counting_conf = max(conf, 0.55)
    reliable_df = df[df['confidence'] >= counting_conf]
    
    # Đếm số lượng track_id duy nhất cho mỗi class nhất quán trên từng frame
    # value_counts() tự động xử lý việc đếm các giá trị duy nhất
    counts = reliable_df.groupby(['video_name', 'frame_id'])['consistent_class_name'].value_counts().unstack(fill_value=0)
    
    # --- LOGIC LỌC NGHIÊM NGẶT ---
    
    # 1. Đảm bảo các cột cần thiết tồn tại để tránh lỗi
    all_known_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    for cls in all_known_classes:
        if cls not in counts.columns:
            counts[cls] = 0
            
    # 2. Xác định các cột "không mong muốn"
    other_classes = list(set(all_known_classes) - {'person', 'bicycle'})
    
    # 3. Tính tổng số lượng các đối tượng không mong muốn
    counts['other_count'] = counts[other_classes].sum(axis=1)
    
    # 4. Áp dụng điều kiện lọc cuối cùng
    #    - Có đúng 1 người
    #    - Có đúng 1 xe đạp
    #    - VÀ không có bất kỳ đối tượng nào khác trong danh sách
    result_frames = counts[
        (counts['person'] == 1) & 
        (counts['bicycle'] == 1) & 
        (counts['other_count'] == 0)
    ]
    
    if result_frames.empty:
        return {}
        
    # Lấy danh sách video_name và frame_id từ kết quả
    grouped = result_frames.reset_index().groupby('video_name')['frame_id'].unique().apply(list)
    result_dict = grouped.apply(lambda lst: sorted([int(i) for i in lst])).to_dict()
    
    # Không áp dụng 'fill_gaps' cho câu hỏi đếm chính xác
    return result_dict


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
    reliable_df = df[(df['consistent_class_name'] == 'person') & (df['confidence'] >= conf)]
    counts = reliable_df.groupby(['video_name', 'frame_id'])['track_id'].nunique().reset_index(name='count')
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

def query_8_v9(df: pd.DataFrame, conf: float) -> dict:
    """
    Câu 8 (V9): CHỈ chứa ba người, với logic "đếm linh hoạt".
    """
    # Sử dụng adjusted_confidence để có một thước đo tin cậy tốt hơn
    strict_counting_conf = max(conf, 0.6)
    reliable_df = df[df['adjusted_confidence'] >= strict_counting_conf]

    # Đếm số lượng track_id duy nhất cho mỗi class trên từng frame
    counts = reliable_df.groupby(['video_name', 'frame_id'])['consistent_class_name'].value_counts().unstack(fill_value=0)

    all_known_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    for cls in all_known_classes:
        if cls not in counts.columns:
            counts[cls] = 0

    other_classes = list(set(all_known_classes) - {'person'})
    
    # --- LOGIC ĐẾM LINH HOẠT ---
    # Thay vì đếm số lượng, ta tính tổng confidence của các đối tượng nhiễu
    # Ta cần join lại với DataFrame gốc để lấy confidence
    counts_with_conf = reliable_df.groupby(['video_name', 'frame_id', 'consistent_class_name'])['confidence'].sum().unstack(fill_value=0)
    for cls in other_classes:
        if cls not in counts_with_conf.columns:
            counts_with_conf[cls] = 0
            
    counts['other_conf_sum'] = counts_with_conf[other_classes].sum(axis=1)

    # Điều kiện lọc: có đúng 3 track person VÀ tổng confidence của nhiễu rất thấp
    seed_frames = counts[(counts['person'] == 3) & (counts['other_conf_sum'] < 0.5)]
    
    if seed_frames.empty:
        return {}
    
    # --- HẬU XỬ LÝ SUY LUẬN ---
    final_result_dict = fill_gaps_for_q8(df, seed_frames.reset_index())
    
    return final_result_dict

# --- TESTING BLOCK ---
if __name__ == "__main__":
    DB_PATH = Path('./deep_tracked_database_v8.feather')
    if not DB_PATH.exists():
        print(f"ERROR: Database file not found at '{DB_PATH}'.")
    else:
        print("Loading V8 database for V9 processing...")
        test_df = pd.read_feather(DB_PATH)
        
        # *** BƯỚC TIỀN XỬ LÝ QUAN TRỌNG ***
        processed_df = preprocess_df_v9(test_df)
        
        print("\n--- Bắt đầu kiểm thử Query 8 (V9) ---")
        TEST_CONF = 0.3 # Ngưỡng cơ bản
        
        result_q8 = query_8_v9(processed_df, TEST_CONF)
        
        if not result_q8:
            print("=> Q8: Không tìm thấy kết quả nào.")
        else:
            print(f"=> Q8: Tìm thấy kết quả trong {len(result_q8)} video.")
            for video, frames in result_q8.items():
                print(f"   - Video '{video}': {len(frames)} frames. Sample: {frames[:10]}...")
        
        print("\n--- TESTING COMPLETE ---")