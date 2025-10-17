import gradio as gr
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pathlib import Path
import traceback
import json
import torch
import torch.nn.functional as F
import numpy as np  
# Import các hàm từ Giai đoạn 0 (cần file utils.py)
from utils import get_clip_text_features, load_all_models # Chỉ cần 2 hàm này

# --- 1. CẤU HÌNH & KHỞI ĐỘNG ---
print("--- KHỞI ĐỘNG BUỒNG LÁI AI TÁC CHIẾN (PHIÊN BẢN THI ĐẤU V4) ---")

# Tải key và kết nối tới Gemini
load_dotenv(); genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')
print("✅ Kết nối tới Gemini API thành công.")

# Tải CSDL
DB_PATH = Path('./god_database_final.parquet')
df = pd.read_parquet(DB_PATH)
# Chuyển đổi feature từ list về numpy array để tính toán
df['clip_feature'] = df['clip_feature'].apply(np.array)
print(f"✅ 'Siêu CSDL' và Kho dự phòng CLIP đã được tải thành công.")

# Tải một phần của model CLIP để encode text query (chỉ cần model và processor)
MODELS_DIR = Path('./models/')
try:
    _, clip_model, clip_proc, _ = load_all_models(
        'yolov8x.pt', # Tên placeholder, vì load_yolo=False
        "openai/clip-vit-base-patch32",
        "vit_h",      # Tên placeholder, vì load_sam=False
        MODELS_DIR / "sam_vit_h_4b8939.pth",
        'cpu',
        models_dir=MODELS_DIR, # Thêm tham số bắt buộc
        load_yolo=False,
        load_sam=False
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model.to(device)
    print(f"✅ Model CLIP đã sẵn sàng cho suy luận ngữ nghĩa trên thiết bị {device}.")
except Exception as e:
    print(f"❌ LỖI khi tải model CLIP: {e}")
    clip_model, clip_proc = None, None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model.to(device)
print(f"✅ Model CLIP đã sẵn sàng cho suy luận ngữ nghĩa trên thiết bị {device}.")

# [CẢI TIẾN GĐ3] Quản lý trạng thái và file submission
SUBMISSION_PATH = Path('./AI25-15.json') # Tên file nộp bài
final_submission = {} # Biến state trong bộ nhớ

def load_submission_state():
    """Tải trạng thái từ file JSON nếu tồn tại."""
    global final_submission
    if SUBMISSION_PATH.exists():
        try:
            with open(SUBMISSION_PATH, 'r', encoding='utf8') as f:
                final_submission = json.load(f)
            print(f"✅ Đã tải lại trạng thái submission từ file '{SUBMISSION_PATH}'.")
        except json.JSONDecodeError:
            print(f"⚠️ CẢNH BÁO: File '{SUBMISSION_PATH}' bị lỗi, bắt đầu với submission rỗng.")
            final_submission = {}
    else:
        print("✅ Khởi tạo submission mới.")

def save_submission_state():
    """Lưu trạng thái hiện tại vào file JSON."""
    global final_submission
    try:
        with open(SUBMISSION_PATH, 'w', encoding='utf8') as f:
            json.dump(final_submission, f, indent=4)
        return True
    except Exception as e:
        print(f"❌ LỖI khi lưu file submission: {e}")
        return False

# Tải trạng thái lần đầu khi khởi động
load_submission_state()


# --- 2. PROMPT V4 (Giữ nguyên) ---
df_info_str = ""
if df is not None:
    df_info_str = f"""- DataFrame `df` có {len(df)} dòng.
- Các cột chính bao gồm: {list(df.columns)}
- Ví dụ 2 dòng đầu tiên:\n{df.head(2).to_string()}"""

MASTER_PROMPT_TEMPLATE = """
BẠN LÀ MỘT ENGINE DỊCH NGÔN NGỮ TỰ NHIÊN SANG CODE PANDAS CỰC KỲ CHÍNH XÁC.
Mục tiêu của bạn là dịch yêu cầu của người dùng thành một khối code Pandas duy nhất.

## BỐI CẢNH MÔI TRƯỜNG THỰC THI:
{df_info}

## QUY TRÌNH LÀM VIỆC BẮT BUỘC:
1.  **KHÔNG** tự tạo DataFrame mẫu. DataFrame `df` đã tồn tại trong môi trường thực thi.
2.  Phân tích yêu cầu của người dùng để tạo ra một DataFrame cuối cùng có tên `filtered_df`. `filtered_df` chỉ nên chứa các dòng thỏa mãn TẤT CẢ các điều kiện.
3.  **SỬ DỤNG KHUÔN MẪU SAU ĐỂ TẠO KẾT QUẢ CUỐI CÙNG:** Sau khi đã có `filtered_df`, hãy kết thúc đoạn code của bạn bằng **CHÍNH XÁC** 3 dòng sau. Đừng thay đổi chúng.

    ```python
    # --- KHUÔN MẪU KẾT QUẢ (KHÔNG THAY ĐỔI) ---
    if filtered_df.empty:
        q_result = {{}}
    else:
        # [SỬA LỖI] Thêm bước ép kiểu `int()` để đảm bảo tương thích với JSON
        q_result = filtered_df.groupby('video_name')['frame_id'].apply(lambda x: sorted([int(i) for i in x.unique()])).to_dict()
    ```

## YÊU CẦU TUYỆT ĐỐI:
- **KHÔNG** định nghĩa hàm mới.
- **KHÔNG** sử dụng `re` (regular expressions) hoặc `eval()`.
- **KHÔNG** tham chiếu đến biến `user_query`.
- Toàn bộ logic của bạn phải dẫn đến việc tạo ra một DataFrame tên là `filtered_df` để đưa vào Khuôn Mẫu Kết Quả.

---
YÊU CẦU CỦA NGƯỜI DÙNG:
{user_query}
"""

# --- 3. HÀM LÕI TƯƠNG TÁC (Giữ nguyên logic, thêm cập nhật state) ---
def generate_code_from_query(question_id: str, user_query: str):
    # Hàm này giữ nguyên như GĐ2, chỉ tạo code và không thay đổi state
    if not model or df is None or not question_id or not user_query:
        error_msg = "LỖI: Vui lòng kiểm tra kết nối API, file CSDL và nhập đủ Mã câu hỏi & Yêu cầu."
        return "# Lỗi đầu vào", error_msg, final_submission
    status_log = f"▶️ Đang tạo lệnh cho câu hỏi [{question_id}]...\n"
    status_log += "  -> Đang gửi yêu cầu tới Gemini API...\n"
    try:
        request_options = {"timeout": 30}
        response = model.generate_content(MASTER_PROMPT_TEMPLATE.format(df_info=df_info_str, user_query=user_query), request_options=request_options)
        generated_code = response.text.replace("```python", "").replace("```", "").strip()
        status_log += "  -> Đã nhận được phản hồi. Code đã sẵn sàng.\n"
    except Exception as e:
        status_log += f"❌ LỖI KHI GỌI GEMINI API: {e}\n"
        generated_code = f"# Lỗi API: {e}"
    return generated_code, status_log, final_submission

def handle_semantic_query(user_query, similarity_threshold=0.25):
    status_log = "▶️ Chế độ Suy luận Ngữ nghĩa (CLIP) được kích hoạt.\n"
    try:
        # 1. Encode text query
        status_log += f"  -> Đang encode yêu cầu: '{user_query}'...\n"
        text_feature = get_clip_text_features([user_query], clip_model, clip_proc, device).numpy()

        # 2. Tính toán cosine similarity trực tiếp trên cột 'clip_feature'
        status_log += f"  -> Đang so sánh với {len(df)} vector ảnh...\n"
        image_features_stack = np.stack(df['clip_feature'].values)
        
        text_tensor = torch.from_numpy(text_feature).to(device)
        image_tensor = torch.from_numpy(image_features_stack).to(device)
        
        similarities = F.cosine_similarity(text_tensor, image_tensor).cpu().numpy()
        
        # 3. Lọc trực tiếp DataFrame chính
        matched_df = df[similarities >= similarity_threshold]
        status_log += f"  -> Tìm thấy {len(matched_df)} đối tượng tiềm năng.\n"

        # 4. Gán `filtered_df` và dùng khuôn mẫu để tạo kết quả
        # Đây là bước quan trọng để tái sử dụng logic tạo q_result
        filtered_df = matched_df
        
        if filtered_df.empty:
            q_result = {}
        else:
            q_result = filtered_df.groupby('video_name')['frame_id'].apply(lambda x: sorted([int(i) for i in x.unique()])).to_dict()
        
        status_log += "✅ Suy luận ngữ nghĩa hoàn tất.\n"
        return q_result, status_log

    except Exception as e:
        tb_str = traceback.format_exc()
        status_log += f"❌ LỖI trong quá trình suy luận ngữ nghĩa:\n{tb_str}\n"
        return None, status_log


def execute_code(question_id, manual_code, use_clip_mode, user_query):
    global final_submission
    if df is None or not question_id:
        return "LỖI: Thiếu thông tin đầu vào.", final_submission

    q_result = None
    status_log = ""

    if use_clip_mode:
        # Chế độ suy luận ngữ nghĩa nâng cao
        if not user_query:
            status_log = "LỖI: Chế độ Suy luận Ngữ nghĩa cần có 'Yêu cầu truy vấn' để hoạt động."
            return status_log, final_submission
        q_result, status_log = handle_semantic_query(user_query)
    else:
        # Chế độ sinh code Pandas thông thường
        if not manual_code:
            status_log = "LỖI: Không có code để thực thi."
            return status_log, final_submission
        status_log = f"▶️ Đang thực thi code Pandas cho câu hỏi [{question_id}]...\n"
        try:
            execution_scope = {'df': df}
            exec(manual_code, globals(), execution_scope)
            q_result = execution_scope.get('q_result')
            if q_result is None: raise ValueError("Không tìm thấy biến 'q_result'.")
        except Exception:
            tb_str = traceback.format_exc()
            status_log += f"❌ LỖI KHI THỰC THI CODE:\n{tb_str}\n"
            return status_log, final_submission
            
    # Cập nhật state và lưu file nếu có kết quả hợp lệ
    if q_result is not None:
        final_submission[question_id] = q_result
        if save_submission_state():
            status_log += f"✅ THÀNH CÔNG! Kết quả cho câu hỏi [{question_id}] đã được CẬP NHẬT và LƯU.\n"
        else:
            status_log += f"⚠️ LỖI: Không thể lưu file submission!\n"
    
    return status_log, final_submission

# --- 4. GIAO DIỆN GRADIO (PHIÊN BẢN THI ĐẤU) ---
with gr.Blocks(theme=gr.themes.Monochrome(), title="Buồng lái AI") as app:
    gr.Markdown("# 🚀 BUỒNG LÁI AI TÁC CHIẾN (V4 - THI ĐẤU) 🚀")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 1. Ra Lệnh")
            q_id_input = gr.Textbox(label="Mã câu hỏi", placeholder="Ví dụ: 1")
            user_query_input = gr.Textbox(label="Yêu cầu truy vấn", placeholder="Ví dụ: tìm xe bus màu xám", lines=3)
            # [CẢI TIẾN GĐ4] Checkbox kích hoạt chế độ CLIP
            clip_mode_checkbox = gr.Checkbox(label="Sử dụng Suy luận Ngữ nghĩa (CLIP - Chậm)", info="Dùng cho các câu hỏi phức tạp về hành động, thuộc tính không có sẵn.")
            with gr.Row():
                # [CẢI TIẾN GĐ4] Nút Clear
                clear_button = gr.ClearButton(value="Xóa Lệnh")
                generate_button = gr.Button("Tạo Lệnh (AI)", variant="secondary")
        
        with gr.Column(scale=3):
            gr.Markdown("### 3. Nhật Ký & Trạng Thái")
            status_log_output = gr.Textbox(label="Nhật ký hệ thống", lines=8, interactive=False)

    gr.Markdown("### 2. Tinh Chỉnh & Thi Hành")
    manual_code_input = gr.Code(label="Bàn làm việc: Code do AI tạo (sửa nếu cần)", language="python", interactive=True)
    execute_button = gr.Button("Thi Hành & Lưu Kết Quả", variant="primary")
    
    gr.Markdown("---")
    gr.Markdown("### 4. Kết Quả Bài Làm Hiện Tại")
    submission_output = gr.JSON(value=final_submission, label=f"Nội dung file: {SUBMISSION_PATH}")

    # Kết nối sự kiện
    generate_button.click(fn=generate_code_from_query, inputs=[q_id_input, user_query_input], outputs=[manual_code_input, status_log_output, submission_output])
    execute_button.click(fn=execute_code, inputs=[q_id_input, manual_code_input, clip_mode_checkbox, user_query_input], outputs=[status_log_output, submission_output])
    clear_button.add([q_id_input, user_query_input, manual_code_input])

if __name__ == "__main__":
    app.launch()