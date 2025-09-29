# **Pipeline Phân Tích Video V4 ("Hội Đồng Thẩm Định")**

Dự án này chứa mã nguồn hoàn chỉnh cho pipeline xử lý và trả lời các câu hỏi. Pipeline được xây dựng theo kiến trúc V4, sử dụng một "Hội đồng Thẩm định" bao gồm YOLO, SAM và CLIP để đạt được độ chính xác và khả năng truy xuất cao nhất.

## **Tổng Quan về Pipeline**

Pipeline được chia thành 4 giai đoạn chính, mỗi giai đoạn tương ứng với một script có thể thực thi:

1.  **Giai đoạn 1 (`1_build_database_v4.py`):** Xây dựng một "Cơ sở dữ liệu bằng chứng" chất lượng cao. Script này quét qua tất cả video, dùng YOLO để phát hiện đối tượng, sau đó dùng SAM và CLIP để xác thực và làm giàu thông tin cho các phát hiện đó.
2.  **Giai đoạn 2 (`2_query_library_v4.py`):** Xây dựng và kiểm thử các hàm truy vấn. File này chứa logic để trả lời 8 câu hỏi của cuộc thi và có tích hợp sẵn chức năng kiểm thử trực quan.
3.  **Giai đoạn 3 (`3_generate_submission_v4.py`):** Tạo file nộp bài cuối cùng. Script này sẽ tải CSDL đã xử lý, chạy tất cả các truy vấn và xuất ra file `[TEAM_ID].json`.
4.  **Giai đoạn 4 (`4_visualize_final_submission_v4.py`):** Tạo báo cáo trực quan. Script này tạo ra một báo cáo HTML phân trang, giúp rà soát toàn bộ kết quả trong file submission một cách hiệu quả.

---

## **Hướng Dẫn Cài Đặt**

### **Yêu cầu về Phần cứng:**
*   **Bắt buộc:** Một GPU NVIDIA với ít nhất 12GB VRAM (khuyến nghị 16GB+).
*   **Khuyến nghị:** CPU đa nhân (8+ cores), 32GB+ RAM.

### **Các bước cài đặt:**

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/MonsieurNam/object_video_retrieval.git
    cd object_video_retrieval
    ```

2.  **Tạo Môi trường Ảo (Khuyến khích):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Linux/macOS
    # venv\Scripts\activate    # Trên Windows
    ```

3.  **Cài đặt các Thư viện:**
    Cài đặt tất cả các gói cần thiết bằng file `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    **Lưu ý:** Bước này có thể mất một vài phút. Đặc biệt, việc cài đặt `segment-anything` từ Git và `torch` có thể cần thời gian.

4.  **Tải các Model Checkpoints:**
    Tạo một thư mục `models/` trong thư mục gốc của dự án.
    ```bash
    mkdir models
    ```
    Tải về file model checkpoint của **Segment Anything Model (ViT-H)** từ [repo chính thức](https://github.com/facebookresearch/segment-anything#model-checkpoints) và đặt nó vào thư mục `models/`. File cần tải là `sam_vit_h_4b8939.pth`.

---

## **Hướng Dẫn Sử Dụng**

### **Chuẩn bị Dữ liệu:**

1.  Tạo một thư mục `Video_vong_loai/` trong thư mục gốc.
2.  Copy tất cả các file video `.mp4` của vòng loại vào thư mục này.

### **Quy trình Chạy 4 Giai Đoạn:**

Thực hiện các bước sau theo đúng thứ tự.

#### **1. Chạy Giai đoạn 1: Xây dựng Cơ sở dữ liệu**
Chạy script để xử lý video và tạo file `evidence_database_v4_high_quality.feather`. Quá trình này sẽ rất tốn thời gian (có thể nhiều giờ).
```bash
python 1_build_database_v4.py
```

#### **2. Chạy Giai đoạn 2: Kiểm thử Logic Truy vấn (Tùy chọn nhưng khuyến khích)**
Chạy script này để xác nhận các hàm truy vấn hoạt động đúng và xem một vài kết quả trực quan mẫu.
```bash
python 2_query_library_v4.py
```

#### **3. Chạy Giai đoạn 3: Tạo File Nộp Bài**
Mở file `3_generate_submission_v4.py` và **cập nhật 2 biến sau**:
*   `TEAM_ID = "AI25-15"` (thay bằng mã số nhóm của bạn).
*   `FINAL_CONFIDENCE_THRESHOLD = 0.35` (tinh chỉnh ngưỡng này nếu cần).

Sau đó, chạy script:
```bash
python 3_generate_submission_v4.py
```
Kết quả sẽ là một file `.json` (ví dụ: `AI25-15.json`).

#### **4. Chạy Giai đoạn 4: Xác Minh Trực Quan**
Mở file `4_generate_visual_report_v4.py` và đảm bảo các giá trị ngưỡng ở phần `CONFIGURATION` khớp với file `query_library`. Sau đó, chạy script:
```bash
python 4_generate_visual_report_v4.py
```
Script này sẽ tạo ra một thư mục `visual_report_v4/`. Bên trong sẽ có các thư mục con `Q1`, `Q2`,... Mỗi thư mục chứa một file `index.html`. Hãy mở các file này bằng trình duyệt để rà soát toàn bộ kết quả của bạn.
````