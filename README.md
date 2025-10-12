# **Video Analysis Pipeline V6 — “Hội Đồng Thẩm Định” với Suy Luận Theo Thời Gian**

Dự án này cung cấp một pipeline hoàn chỉnh để xử lý video và trả lời các câu hỏi truy vấn phức tạp về nội dung.

Pipeline V6 là một bản nâng cấp lớn, giữ lại kiến trúc “Hội đồng thẩm định” (YOLO, SAM, CLIP) và bổ sung thêm khả năng **theo dõi đối tượng (tracking)** và **suy luận theo thời gian** để tăng cường đáng kể độ chính xác và ổn định của kết quả.

---

## **Điểm Nâng Cấp Chính (V6 so với V4)**
- **Tracking Đối Tượng**: Bổ sung một giai đoạn xử lý để liên kết các phát hiện của cùng một đối tượng qua nhiều khung hình, tạo thành các "đường đi" (tracks).
- **Suy Luận Theo Thời Gian**: Dựa trên dữ liệu tracking, triển khai kỹ thuật **"Bầu cử Danh tính"** để tự động sửa các lỗi nhận dạng nhất thời, làm cho nhãn của đối tượng nhất quán trong suốt quá trình xuất hiện.

---

## **Tổng quan Kiến trúc**

Pipeline bao gồm **5 giai đoạn chính**, mỗi giai đoạn được triển khai trong một script độc lập:

1.  **Giai đoạn 1: Xây dựng CSDL Chứng cứ (`build_database_v5.py`)**
    - Quét toàn bộ video đầu vào.
    - Sử dụng **YOLO** để phát hiện đối tượng, sau đó dùng **SAM** và **CLIP** trong "hội đồng thẩm định" để xác thực và lọc ra các phát hiện nhiễu.
    - Kết quả là một **CSDL bằng chứng** (`evidence_database_v5_expert.feather`) chất lượng cao.

2.  **Giai đoạn 1.5: Xây dựng Đường đi (`build_trackv6.py`)**
    - Lấy CSDL chứng cứ làm đầu vào.
    - Áp dụng thuật toán tracking dựa trên IoU để liên kết các phát hiện thành các đường đi của đối tượng.
    - Kết quả là một **CSDL đã được tracking** (`tracked_database_v6.feather`) chứa `track_id` cho mỗi phát hiện.

3.  **Giai đoạn 2: Xây dựng Thư viện Truy vấn (`query_library_v6.py`)**
    - Định nghĩa logic trả lời 8 câu hỏi của cuộc thi trên dữ liệu đã được tracking.
    - Tích hợp logic suy luận theo thời gian, nổi bật là kỹ thuật **"Bầu cử Danh tính"** để làm nhất quán nhãn của đối tượng trong suốt một track.
    - Chứa các hàm kiểm thử để xác nhận tính chính xác.

4.  **Giai đoạn 3: Sinh File Submission (`generate_submission_v6.py`)**
    - Tải CSDL đã được tracking.
    - Chạy toàn bộ 8 truy vấn và xuất kết quả cuối cùng dưới dạng file JSON (`[TEAM_ID].json`).

5.  **Giai đoạn 4: Tạo Báo cáo Trực quan (`visualize_final_submission_v6.py`)**
    - Sinh báo cáo HTML phân trang từ file submission.
    - Trực quan hóa cả các box được sửa lỗi bởi tracking, giúp kiểm chứng hiệu quả của việc làm nhất quán dữ liệu một cách nhanh chóng.

---

## **Yêu cầu Hệ thống**

### **Phần cứng**
- **Tối thiểu:** GPU NVIDIA ≥ 12 GB VRAM.
- **Khuyến nghị:** GPU 16 GB VRAM, CPU ≥ 8 cores, RAM ≥ 32 GB.

### **Phần mềm**
- Python ≥ 3.8
- CUDA/cuDNN (nếu chạy bằng GPU)

---

## **Cài đặt**

1.  **Clone repository**
    ```bash
    git clone https://github.com/MonsieurNam/object_video_retrieval.git
    cd object_video_retrieval
    ```

    Tải và giải nén dataset vòng loại:
    ```bash
    curl -L -o /root/aicontest-vongloai.zip \
      https://www.kaggle.com/api/v1/datasets/download/nguyenngonhatnam/aicontest-vongloai
    unzip aicontest-vongloai.zip
    ```

2.  **Tạo môi trường ảo (khuyến nghị)**
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

    Cài đặt các thư viện hệ thống cần thiết:
    ```bash
    apt-get update -y
    apt-get install -y libglib2.0-0 libgl1 libsm6 libxext6 libxrender-dev
    ```

3.  **Cài đặt dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    > ⚠️ Việc cài đặt `torch` và `segment-anything` có thể mất nhiều thời gian.

4.  **Chuẩn bị model checkpoints**
    - Tạo thư mục:
      ```bash
      mkdir models
      ```
    - Tải model **SAM ViT-H** từ [repo chính thức](https://github.com/facebookresearch/segment-anything#model-checkpoints).
    - Đặt file `sam_vit_h_4b8939.pth` vào thư mục `models/`.
    - Tải và đặt model YOLO (ví dụ `yolo11x.pt`) vào thư mục gốc hoặc cập nhật đường dẫn trong script.

---

## **Hướng dẫn Sử dụng**

### **Chuẩn bị dữ liệu**
- Tạo thư mục `Video_vong_loai/` tại thư mục gốc của dự án.
- Copy toàn bộ file video `.mp4` vào thư mục này.

### **Quy trình chạy**

1.  **Xây dựng CSDL bằng chứng (V5)**
    ```bash
    python3 build_database_v5.py
    ```
    - **Kết quả**: `evidence_database_v5_expert.feather`
    > ⏳ Quá trình này có thể kéo dài nhiều giờ.

2.  **Xây dựng Đường đi (V6)**
    ```bash
    python3 build_trackv6.py
    ```
    - **Đầu vào**: `evidence_database_v5_expert.feather`
    - **Kết quả**: `tracked_database_v6.feather`
    > ⚡ Quá trình này tương đối nhanh.

3.  **Kiểm thử thư viện truy vấn (tùy chọn nhưng khuyến khích)**
    ```bash
    python3 query_library_v6.py
    ```

4.  **Sinh file submission (V6)**
    - Mở file `generate_submission_v6.py` và cập nhật:
      ```python
      TEAM_ID = "AI25-15"               # thay bằng mã đội của bạn
      FINAL_CONFIDENCE_THRESHOLD = 0.2  # chỉnh ngưỡng nếu cần
      ```
    - Chạy script:
      ```bash
      python3 generate_submission_v6.py
      ```
    - **Kết quả**: file JSON, ví dụ `AI25-15.json`.

5.  **Sinh báo cáo trực quan (V6)**
    - Đảm bảo các tham số trong `visualize_final_submission_v6.py` khớp với `query_library_v6.py`.
    - Chạy:
      ```bash
      python3 visualize_final_submission_v6.py
      ```
    - **Kết quả**: thư mục `visual_report_v6_FINAL/` chứa báo cáo HTML cho từng câu hỏi (`Q1/index.html`, `Q2/index.html`, ...).