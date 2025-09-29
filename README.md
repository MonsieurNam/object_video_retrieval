# **Video Analysis Pipeline V4 — “Hội Đồng Thẩm Định”**

Dự án này cung cấp một pipeline hoàn chỉnh để xử lý video và trả lời câu hỏi theo yêu cầu.  
Pipeline V4 được thiết kế theo kiến trúc “Hội đồng thẩm định”, kết hợp sức mạnh của **YOLO**, **SAM**, và **CLIP** nhằm đạt được độ chính xác cao và khả năng truy xuất mạnh mẽ.

---

## **Tổng quan Kiến trúc**

Pipeline bao gồm **4 giai đoạn chính**, mỗi giai đoạn được triển khai trong một script độc lập:

1. **Xây dựng cơ sở dữ liệu (`1_build_database_v4.py`)**  
   - Quét toàn bộ video đầu vào.  
   - Sử dụng **YOLO** để phát hiện đối tượng, sau đó dùng **SAM** và **CLIP** để xác thực và bổ sung thông tin.  
   - Kết quả là một **CSDL bằng chứng** chất lượng cao.

2. **Xây dựng thư viện truy vấn (`2_query_library_v4.py`)**  
   - Định nghĩa logic trả lời 8 câu hỏi của cuộc thi.  
   - Tích hợp chức năng kiểm thử trực quan để xác nhận tính chính xác.

3. **Sinh file submission (`3_generate_submission_v4.py`)**  
   - Tải cơ sở dữ liệu đã xử lý.  
   - Chạy toàn bộ truy vấn và xuất kết quả dưới dạng file JSON (`[TEAM_ID].json`).

4. **Tạo báo cáo trực quan (`4_visualize_final_submission_v4.py`)**  
   - Sinh báo cáo HTML phân trang.  
   - Hỗ trợ rà soát toàn bộ kết quả trong submission một cách trực quan, nhanh chóng.

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

1. **Clone repository**
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

2. **Tạo môi trường ảo (khuyến nghị)**
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

3. **Cài đặt dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > ⚠️ Việc cài đặt `torch` và `segment-anything` có thể mất nhiều thời gian.

4. **Chuẩn bị model checkpoints**
   - Tạo thư mục:
     ```bash
     mkdir models
     ```
   - Tải model **SAM ViT-H** từ [repo chính thức](https://github.com/facebookresearch/segment-anything#model-checkpoints).  
   - Đặt file `sam_vit_h_4b8939.pth` vào thư mục `models/`.

---

## **Hướng dẫn Sử dụng**

### **Chuẩn bị dữ liệu**
- Tạo thư mục `Video_vong_loai/` tại thư mục gốc.  
- Copy toàn bộ file video `.mp4` vào thư mục này.  

### **Quy trình chạy**

1. **Xây dựng CSDL bằng chứng**
   ```bash
   python3 build_database_v4.py
   ```
   Kết quả: `evidence_database_v4_high_quality.feather`  
   > ⏳ Quá trình này có thể kéo dài nhiều giờ.

2. **Kiểm thử thư viện truy vấn (tùy chọn nhưng khuyến khích)**
   ```bash
   python3 query_library_v4.py
   ```

3. **Sinh file submission**
   - Mở file `generate_submission_v4.py` và cập nhật:
     ```python
     TEAM_ID = "AI25-15"               # thay bằng mã đội của bạn
     FINAL_CONFIDENCE_THRESHOLD = 0.35 # chỉnh ngưỡng nếu cần
     ```
   - Chạy script:
     ```bash
     python3 generate_submission_v4.py
     ```
   - Kết quả: file JSON, ví dụ `AI25-15.json`.

4. **Sinh báo cáo trực quan**
   - Đảm bảo `CONFIGURATION` trong `4_visualize_final_submission_v4.py` khớp với `query_library`.  
   - Chạy:
     ```bash
     python3 visualize_final_submission_v4.py
     ```
   - Kết quả: thư mục `visual_report_v4/` chứa báo cáo HTML cho từng câu hỏi (`Q1/index.html`, `Q2/index.html`, ...).  
