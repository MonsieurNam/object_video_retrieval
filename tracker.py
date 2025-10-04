
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def bbox_to_z(bbox):
    """
    Chuyển đổi bbox [x1, y1, x2, y2] thành dạng đo lường z [cx, cy, a, h]
    cho Bộ lọc Kalman.
    - cx, cy: Tọa độ tâm
    - a: Tỉ lệ khung hình (aspect ratio)
    - h: Chiều cao
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.
    cy = bbox[1] + h / 2.
    a = w / h if h > 0 else 0
    return np.array([cx, cy, a, h]).reshape((4, 1))

def z_to_bbox(z):
    """
    Chuyển đổi từ trạng thái dự đoán của Kalman [cx, cy, a, h, ...]
    trở lại thành bbox [x1, y1, x2, y2] để hiển thị hoặc tính IoU.
    """
    h = z[3]
    w = z[2] * h
    x1 = z[0] - w / 2.
    y1 = z[1] - h / 2.
    return np.array([x1, y1, x1 + w, y1 + h]).reshape((1, 4))


class Track:
    """
    Đại diện cho một đối tượng duy nhất đang được theo dõi.
    (Giữ lại docstring cũ)
    """
    def __init__(self, track_id: int, initial_detection: dict, nn_budget: int):
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = 'Tentative'
        
        # Lưu lại thông tin của detection gần nhất đã được khớp
        # Rất quan trọng để gán lại track_id vào dataframe ở script chính
        self.last_matched_detection = initial_detection
        self.bbox = initial_detection['bbox']
        self.features = [initial_detection['clip_feature']]
        self.nn_budget = nn_budget
        
        # --- CẤU HÌNH KALMAN FILTER ---
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        # Ma trận chuyển đổi trạng thái (State Transition Matrix)
        # Mô hình chuyển động vận tốc không đổi
        self.kf.F = np.array([
            [1,0,0,0,1,0,0,0],  # cx' = cx + vx
            [0,1,0,0,0,1,0,0],  # cy' = cy + vy
            [0,0,1,0,0,0,1,0],  # a'  = a  + va
            [0,0,0,1,0,0,0,1],  # h'  = h  + vh
            [0,0,0,0,1,0,0,0],  # vx' = vx
            [0,0,0,0,0,1,0,0],  # vy' = vy
            [0,0,0,0,0,0,1,0],  # va' = va
            [0,0,0,0,0,0,0,1]   # vh' = vh
        ])

        # Ma trận đo lường (Measurement Matrix)
        # Chúng ta chỉ đo được vị trí, không đo được vận tốc
        self.kf.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]
        ])

        # Ma trận hiệp phương sai của nhiễu quá trình (Process Noise Covariance)
        self.kf.Q[4:,4:] *= 0.01
        self.kf.P[4:,4:] *= 1000. 
        self.kf.P *= 10.
        
        # Ma trận hiệp phương sai của nhiễu đo lường (Measurement Noise Covariance)
        self.kf.R[2:,2:] *= 10.
        
        # Khởi tạo trạng thái ban đầu [cx, cy, a, h, 0, 0, 0, 0]
        self.kf.x[:4] = bbox_to_z(self.bbox)

    def predict(self):
        """Dự đoán vị trí tiếp theo của bounding box."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: dict):
        """Cập nhật track với một phát hiện mới đã được khớp."""
        # Cập nhật trạng thái Kalman với đo lường mới
        self.kf.update(bbox_to_z(detection['bbox']))
        
        # Cập nhật thông tin của track
        self.last_matched_detection = detection
        self.bbox = detection['bbox']
        
        self.time_since_update = 0
        self.hits += 1
        if self.state == 'Tentative' and self.hits >= 3:
            self.state = 'Confirmed'
        
        # Cập nhật ngân sách đặc trưng ngoại hình
        self.features.append(detection['clip_feature'])
        if len(self.features) > self.nn_budget:
            self.features.pop(0)
            
    def get_predicted_bbox(self):
        """Lấy bbox được dự đoán từ bộ lọc Kalman."""
        return z_to_bbox(self.kf.x)

    def is_confirmed(self):
        return self.state == 'Confirmed'
    
    
class DeepSORTTracker:
    """
    (Giữ lại docstring cũ)
    """
    def __init__(self, max_age: int = 30, nn_budget: int = 100, lambda_val: float = 0.3):
        self.max_age = max_age
        self.nn_budget = nn_budget
        self.lambda_val = lambda_val
        self.tracks = []
        self.next_id = 0

    def _cosine_distance(self, features_a, features_b):
        """Tính toán khoảng cách cosine giữa hai bộ đặc trưng."""
        # 1 - similarity để có distance, vì thuật toán Hungary tìm chi phí nhỏ nhất
        return 1. - np.dot(features_a, features_b.T) / (np.linalg.norm(features_a, axis=1, keepdims=True) * np.linalg.norm(features_b, axis=1, keepdims=True).T)
    
    def _get_appearance_cost_matrix(self, tracks, detections):
        """Xây dựng ma trận chi phí dựa trên ngoại hình (CLIP features)."""
        n_tracks, n_dets = len(tracks), len(detections)
        if n_tracks == 0 or n_dets == 0:
            return np.empty((0, 0))
            
        cost_matrix = np.zeros((n_tracks, n_dets))
        
        # Lấy feature của tất cả detections
        det_features = np.asarray([d['clip_feature'] for d in detections])
        
        for i, track in enumerate(tracks):
            # Tính khoảng cách giữa feature của detection và TẤT CẢ features trong budget của track,
            # sau đó lấy giá trị nhỏ nhất (khớp nhất).
            track_features = np.asarray(track.features)
            cost_matrix[i, :] = np.min(self._cosine_distance(track_features, det_features), axis=0)
            
        return cost_matrix
        
    def _get_motion_cost_matrix(self, tracks, detections):
        """Xây dựng ma trận chi phí dựa trên chuyển động (Khoảng cách Mahalanobis)."""
        n_tracks, n_dets = len(tracks), len(detections)
        if n_tracks == 0 or n_dets == 0:
            return np.empty((0, 0))
        
        cost_matrix = np.zeros((n_tracks, n_dets))
        for i, track in enumerate(tracks):
            # Khoảng cách Mahalanobis từ bộ lọc Kalman
            # Nó đo lường "mức độ bất ngờ" của một phát hiện mới so với dự đoán
            cost_matrix[i, :] = track.kf.mahlanobis(np.array([bbox_to_z(d['bbox']) for d in detections]))
        return cost_matrix

    def update(self, detections_in_frame: list):
        """
        Đây là hàm chính, được gọi mỗi frame một lần.
        """
        # --- 1. Dự đoán trạng thái mới và tách các track đang hoạt động ---
        for t in self.tracks:
            t.predict()
        
        active_tracks = [t for t in self.tracks if not (t.time_since_update > 1 and t.state == 'Tentative')]
        
        # --- 2. Xây dựng ma trận chi phí và liên kết (Cascade 1: Ưu tiên ngoại hình) ---
        appearance_cost = self._get_appearance_cost_matrix(active_tracks, detections_in_frame)
        motion_cost = self._get_motion_cost_matrix(active_tracks, detections_in_frame)
        
        # Ở đây chúng ta có thể làm phức tạp hơn (gating), nhưng để đơn giản:
        # chúng ta sẽ kết hợp chúng lại.
        cost_matrix = self.lambda_val * motion_cost + (1 - self.lambda_val) * appearance_cost
        
        # --- 3. Thực hiện liên kết bằng Thuật toán Hungary ---
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # --- 4. Xử lý kết quả liên kết ---
        matched_indices = []
        unmatched_track_indices = set(range(len(active_tracks)))
        unmatched_detection_indices = set(range(len(detections_in_frame)))

        for r, c in zip(row_ind, col_ind):
            # Chỉ khớp nếu chi phí nhỏ hơn một ngưỡng nhất định
            # Ngưỡng này cần được tinh chỉnh. 0.9 cho cosine dist là khá lỏng lẻo.
            if cost_matrix[r, c] < 0.9:
                active_tracks[r].update(detections_in_frame[c])
                matched_indices.append((r,c))
                unmatched_track_indices.remove(r)
                unmatched_detection_indices.remove(c)

        # --- 5. (Tùy chọn, nâng cao) Cascade 2: Thử liên kết các track chưa khớp chỉ bằng IoU ---
        # Đây là bước dự phòng, nếu một đối tượng bị che khuất quá lâu,
        # đặc trưng ngoại hình và chuyển động đều không còn tin cậy.
        # Để đơn giản, chúng ta có thể bỏ qua bước này trong lần triển khai đầu tiên.

        # --- 6. Tạo track mới cho các phát hiện chưa được khớp ---
        for det_idx in unmatched_detection_indices:
            new_track = Track(self.next_id, detections_in_frame[det_idx], self.nn_budget)
            self.tracks.append(new_track)
            self.next_id += 1
            
        # --- 7. Xóa các track đã cũ và không còn được cập nhật ---
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]