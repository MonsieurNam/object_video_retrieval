# ==============================================================================
# === GĐ 2 (V6): THƯ VIỆN TRUY VẤN VỚI SUY LUẬN THEO THỜI GIAN ==================
# ==============================================================================
# Mục đích:
# 1. Tải CSDL đã được tracking (V6).
# 2. Thực hiện bước tiền xử lý "Bầu cử Danh tính" để làm nhất quán
#    class_name cho mỗi đường đi (track).
# 3. Chạy các hàm truy vấn đã được điều chỉnh để hoạt động trên
#    dữ liệu đã được làm nhất quán này.
# ==============================================================================

import pandas as pd
from pathlib import Path
import warnings
from tqdm import tqdm
import random
import numpy as np

from build_trackv6 import calculate_iou

# Tắt các cảnh báo không cần thiết
# warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. HÀM TIỀN XỬ LÝ (PRE-PROCESSING) ---

def preprocess_df_with_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thực hiện "Bầu cử Danh tính" để tạo cột 'consistent_class_name'.
    """
    print("Bắt đầu bước tiền xử lý: Bầu cử Danh tính cho các Tracks...")
    
    # Kiểm tra xem df có rỗng hoặc thiếu cột cần thiết không
    if df.empty or 'track_id' not in df.columns or 'class_name' not in df.columns:
        print("CSDL rỗng hoặc thiếu cột 'track_id'/'class_name'. Bỏ qua tiền xử lý.")
        if 'consistent_class_name' not in df.columns:
             df['consistent_class_name'] = df.get('class_name', pd.Series(dtype='str'))
        return df

    # Tìm class_name phổ biến nhất cho mỗi track
    # agg(lambda x: x.mode()[0] if not x.mode().empty else None) xử lý trường hợp track rỗng
    track_class_map = df.groupby('track_id')['class_name'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    
    # Ánh xạ kết quả bầu cử vào một cột mới
    df['consistent_class_name'] = df['track_id'].map(track_class_map)
    
    print("Tiền xử lý hoàn tất. Cột 'consistent_class_name' đã được tạo.")
    return df

# --- 2. HÀM HẬU XỬ LÝ (POST-PROCESSING) ---

def fill_gaps(frame_list: list, max_gap_size: int = 10) -> list:
    """Lấp đầy các khoảng trống nhỏ trong một danh sách frame đã được sắp xếp."""
    if len(frame_list) < 2:
        return frame_list
    
    filled_list = []
    filled_list.append(frame_list[0])
    
    for i in range(1, len(frame_list)):
        prev_frame = frame_list[i-1]
        current_frame = frame_list[i]
        gap = current_frame - prev_frame
        
        if 1 < gap <= max_gap_size + 1:
            for frame_to_add in range(prev_frame + 1, current_frame):
                filled_list.append(frame_to_add)
        
        filled_list.append(current_frame)
        
    return filled_list

def apply_post_processing(result_dict: dict, max_gap: int = 20) -> dict:
    """Áp dụng hàm fill_gaps cho tất cả các video trong kết quả."""
    if not result_dict: return {}
    processed_dict = {video: fill_gaps(sorted(frames), max_gap_size=max_gap) if frames else [] for video, frames in result_dict.items()}
    return processed_dict

def non_max_suppression_on_frame(df_frame: pd.DataFrame, iou_threshold: float = 0.6) -> pd.DataFrame:
    """Áp dụng Non-Max Suppression cho các phát hiện trên cùng một frame."""
    if df_frame.empty: return df_frame
    
    df_frame = df_frame.sort_values('confidence', ascending=False)
    keep_indices = []
    suppressed_indices = set()

    for i in df_frame.index:
        if i in suppressed_indices: continue
        keep_indices.append(i)
        row_i = df_frame.loc[i]
        
        for j in df_frame.index:
            if j <= i or j in suppressed_indices: continue
            row_j = df_frame.loc[j]
            
            if row_i['consistent_class_name'] != row_j['consistent_class_name']: continue

            b_i, b_j = row_i['bbox'], row_j['bbox']
            inter_x1, inter_y1 = max(b_i[0], b_j[0]), max(b_i[1], b_j[1])
            inter_x2, inter_y2 = min(b_i[2], b_j[2]), min(b_i[3], b_j[3])
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            
            if inter_area > 0:
                area_i = (b_i[2] - b_i[0]) * (b_i[3] - b_i[1])
                area_j = (b_j[2] - b_j[0]) * (b_j[3] - b_j[1])
                iou = inter_area / float(area_i + area_j - inter_area)
                if iou > iou_threshold: suppressed_indices.add(j)
                
    return df_frame.loc[keep_indices]

# --- 3. CÁC HÀM TRUY VẤN V6 ---

def to_final_dict(grouped_series):
    """Hàm tiện ích để chuyển đổi grouped series sang dict cuối cùng."""
    return grouped_series.apply(lambda lst: sorted([int(i) for i in lst])).to_dict()

def query_1(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[df['confidence'] >= conf]
    frames_with_person = reliable_df[reliable_df['consistent_class_name'] == 'person'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_moto = reliable_df[reliable_df['consistent_class_name'] == 'motorcycle'][['video_name', 'frame_id']].drop_duplicates()
    result_df = pd.merge(frames_with_person, frames_with_moto, on=['video_name', 'frame_id'])
    if result_df.empty: return {}
    return apply_post_processing(to_final_dict(result_df.groupby('video_name')['frame_id'].unique().apply(list)))

def query_2(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[df['confidence'] >= conf]
    frames_with_person = reliable_df[reliable_df['consistent_class_name'] == 'person'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_bicycle = reliable_df[reliable_df['consistent_class_name'] == 'bicycle'][['video_name', 'frame_id']].drop_duplicates()
    result_df = pd.merge(frames_with_person, frames_with_bicycle, on=['video_name', 'frame_id'])
    if result_df.empty: return {}
    return apply_post_processing(to_final_dict(result_df.groupby('video_name')['frame_id'].unique().apply(list)))

def query_3(df: pd.DataFrame, conf: float) -> dict:
    result_df = df[(df['consistent_class_name'] == 'car') & (df['confidence'] >= conf)]
    if result_df.empty: return {}
    return apply_post_processing(to_final_dict(result_df.groupby('video_name')['frame_id'].unique().apply(list)))

def query_4_improved(df: pd.DataFrame, conf: float) -> dict:
    """
    Câu 4 NÂNG CAO: Tìm các frame có CHÍNH XÁC MỘT CẶP (người, xe đạp) đang đi cùng nhau
    và không có người hay xe đạp nào khác trong frame.

    Cải tiến so với phiên bản cũ:
    - Loại bỏ giả định `num_persons == num_bicycles` không đáng tin cậy.
    - Logic ghép cặp mạnh mẽ hơn, tìm ra cặp có IoU tốt nhất cho mỗi người.
    - Đảm bảo rằng frame chỉ chứa đúng một cặp và không có đối tượng "lẻ loi" nào khác.
    """
    # Ngưỡng tin cậy có thể cao hơn một chút cho các bài toán đếm
    counting_conf = max(conf, 0.50) 
    
    # Ngưỡng IoU để coi một người và một xe đạp là một "cặp".
    # Có thể tinh chỉnh giá trị này. 0.1 có nghĩa là cần có sự chồng chéo đáng kể.
    IOU_RIDING_THRESHOLD = 0.1 

    reliable_df = df[df['confidence'] >= counting_conf]
    
    # Tối ưu hóa: chỉ xử lý các frame có cả 'person' và 'bicycle'
    candidate_frames = reliable_df[reliable_df['consistent_class_name'].isin(['person', 'bicycle'])]
    
    final_result_frames = []

    # Lặp qua từng frame ứng viên
    # Sử dụng `leave=False` để thanh tiến trình không làm rối màn hình khi chạy trong submission script
    grouped_frames = candidate_frames.groupby(['video_name', 'frame_id'])
    for (video_name, frame_id), frame_group in tqdm(grouped_frames, desc="Query 4 (Improved Spatial Check)", leave=False):
        
        persons = frame_group[frame_group['consistent_class_name'] == 'person']
        bicycles = frame_group[frame_group['consistent_class_name'] == 'bicycle']
        
        # Bỏ qua ngay nếu thiếu một trong hai loại đối tượng
        if persons.empty or bicycles.empty:
            continue

        # Lấy ra các track duy nhất trong frame này để tránh đếm đúp
        person_tracks = {track_id: group.iloc[0]['bbox'] for track_id, group in persons.groupby('track_id')}
        bicycle_tracks = {track_id: group.iloc[0]['bbox'] for track_id, group in bicycles.groupby('track_id')}
        
        num_persons = len(person_tracks)
        num_bicycles = len(bicycle_tracks)

        # --- LOGIC CỐT LÕI MỚI ---
        # Chỉ xem xét nếu có khả năng tạo thành một cặp duy nhất
        if num_persons != 1 or num_bicycles != 1:
            continue

        # Bây giờ chúng ta biết chắc chắn chỉ có 1 người và 1 xe đạp.
        # Chỉ cần kiểm tra xem chúng có "gắn kết" với nhau không.
        person_bbox = list(person_tracks.values())[0]
        bicycle_bbox = list(bicycle_tracks.values())[0]

        iou = calculate_iou(person_bbox, bicycle_bbox)
        
        # Điều kiện cuối cùng: có đúng 1 người, 1 xe đạp, và chúng có tương tác vật lý (IoU > ngưỡng)
        if iou > IOU_RIDING_THRESHOLD:
             final_result_frames.append({'video_name': video_name, 'frame_id': frame_id})

    if not final_result_frames: 
        return {}
    
    result_df = pd.DataFrame(final_result_frames)
    grouped = result_df.groupby('video_name')['frame_id'].unique().apply(list)
    
    # Áp dụng hậu xử lý và chuyển đổi sang định dạng cuối cùng
    final_dict = grouped.apply(lambda lst: sorted([int(i) for i in lst])).to_dict()
    return apply_post_processing(final_dict) # Áp dụng fill_gaps


def query_5(df: pd.DataFrame, conf: float) -> dict:
    reliable_df = df[df['confidence'] >= conf]
    frames_with_person = reliable_df[reliable_df['consistent_class_name'] == 'person'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_moto = reliable_df[reliable_df['consistent_class_name'] == 'motorcycle'][['video_name', 'frame_id']].drop_duplicates()
    frames_with_car = reliable_df[reliable_df['consistent_class_name'] == 'car'][['video_name', 'frame_id']].drop_duplicates()
    result_df = pd.merge(frames_with_person, frames_with_moto, on=['video_name', 'frame_id'])
    result_df = pd.merge(result_df, frames_with_car, on=['video_name', 'frame_id'])
    if result_df.empty: return {}
    return apply_post_processing(to_final_dict(result_df.groupby('video_name')['frame_id'].unique().apply(list)))

def query_6_and_7_base(df: pd.DataFrame, conf: float, class_name: str) -> dict:
    counting_conf = max(conf, 0.45)
    reliable_df = df[(df['consistent_class_name'] == class_name) & (df['confidence'] >= counting_conf)]
    processed_df = reliable_df.groupby(['video_name', 'frame_id']).apply(non_max_suppression_on_frame).reset_index(drop=True)
    counts = processed_df.groupby(['video_name', 'frame_id']).size().reset_index(name='count')
    result_frames = counts[counts['count'] > 1]
    if result_frames.empty: return {}
    return apply_post_processing(to_final_dict(result_frames.groupby('video_name')['frame_id'].unique().apply(list)))

def query_6(df: pd.DataFrame, conf: float) -> dict:
    return query_6_and_7_base(df, conf, 'person')

def query_7(df: pd.DataFrame, conf: float) -> dict:
    return query_6_and_7_base(df, conf, 'motorcycle')

def query_8(df: pd.DataFrame, conf: float) -> dict:
    strict_counting_conf = max(conf, 0.65)
    reliable_df = df[df['confidence'] >= strict_counting_conf]
    processed_df = reliable_df.groupby(['video_name', 'frame_id']).apply(non_max_suppression_on_frame).reset_index(drop=True)
    counts = processed_df.groupby(['video_name', 'frame_id', 'consistent_class_name']).size().unstack(fill_value=0)
    
    all_known_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    for cls in all_known_classes:
        if cls not in counts.columns: counts[cls] = 0
            
    other_classes = list(set(all_known_classes) - {'person'})
    counts['other_count'] = counts[other_classes].sum(axis=1)
    
    result_frames = counts[(counts['person'] == 3) & (counts['other_count'] <= 1)]
    if result_frames.empty: return {}
    return to_final_dict(result_frames.reset_index().groupby('video_name')['frame_id'].unique().apply(list))

# --- 4. KHỐI KIỂM THỬ ---
if __name__ == "__main__":
    DB_PATH = Path('./tracked_database_v6.feather')
    if not DB_PATH.exists():
        print(f"LỖI: Không tìm thấy file CSDL đã tracking '{DB_PATH}'. Chạy GĐ 1.5 trước.")
    else:
        print("Đang tải CSDL V6 để kiểm thử...")
        test_df = pd.read_feather(DB_PATH)
        
        test_df = preprocess_df_with_tracking(test_df)
        
        print("\nKiểm tra nhanh kết quả tiền xử lý:")
        inconsistent_tracks = test_df[test_df['class_name'] != test_df['consistent_class_name']]['track_id'].unique()
        if len(inconsistent_tracks) > 0:
            track_to_show = inconsistent_tracks[0]
            print(f"Ví dụ về track ID #{track_to_show} đã được làm nhất quán:")
            print(test_df[test_df['track_id'] == track_to_show][['frame_id', 'class_name', 'consistent_class_name']].head(10))
        else:
            print("Không tìm thấy track nào có sự nhầm lẫn để sửa lỗi. Dữ liệu đã rất nhất quán.")

        TEST_CONF = 0.35 
        print(f"\nSử dụng ngưỡng tin cậy để kiểm thử: {TEST_CONF}")
        
        queries_to_test = {
            "Q1: Người & Xe máy": query_1, "Q2: Người & Xe đạp": query_2,
            "Q3: Xe ô tô": query_3, "Q4: >=1 Người & >=1 Xe đạp": query_4,
            "Q5: Người & Xe máy & Xe ô tô": query_5, "Q6: >1 Người": query_6,
            "Q7: >1 Xe máy": query_7, "Q8: Chỉ có 3 Người": query_8,
        }
        
        for name, func in queries_to_test.items():
            print(f"\n--- Testing: {name} ---")
            try:
                result = func(test_df, TEST_CONF)
                if not result:
                    print("=> Không tìm thấy kết quả nào.")
                else:
                    print(f"=> Tìm thấy kết quả trong {len(result)} video.")
                    first_video = list(result.keys())[0]
                    first_video_frames = result[first_video]
                    print(f"   Ví dụ: Video '{first_video}' có {len(first_video_frames)} frame. Sample: {first_video_frames[:10]}")
            except Exception as e:
                print(f"!!! LỖI khi chạy truy vấn: {e}")

        print("\n--- KIỂM THỬ HOÀN TẤT ---")