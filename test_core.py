#
# DÁN LẠI TOÀN BỘ FILE NÀY ĐỂ ĐẢM BẢO KHÔNG CÒN LỖI
#

# test_core.py
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pathlib import Path
import google.api_core.exceptions

# --- 1. CẤU HÌNH & KHỞI ĐỘNG ---

load_dotenv()
try:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key: raise ValueError("GEMINI_API_KEY không được tìm thấy trong file .env")
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-2.5-flash') # Sử dụng model ổn định
    print("✅ Kết nối tới Gemini API thành công.")
except Exception as e:
    print(f"❌ LỖI: Không thể kết nối tới Gemini API. Chi tiết: {e}")
    model = None; exit()

DB_PATH = Path('./god_database_final.parquet')
try:
    df = pd.read_parquet(DB_PATH)
    print(f"✅ 'Siêu CSDL' đã được tải thành công với {len(df):,} dòng.")
except FileNotFoundError:
    print(f"❌ LỖI: Không tìm thấy file CSDL tại '{DB_PATH}'. Vui lòng chạy build_god_database.py trước.")
    exit()

# --- 2. "SIÊU PROMPT" DỰA TRÊN CSDL THỰC TẾ ---

df_info_str = f"""
- DataFrame `df` có {len(df)} dòng.
- Các cột chính bao gồm: {list(df.columns)}
- Ví dụ 2 dòng đầu tiên:
{df.head(2).to_string()}
"""

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
        q_result = filtered_df.groupby('video_name')['frame_id'].apply(lambda x: sorted(list(x.unique()))).to_dict()
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

# --- 3. KIỂM THỬ LÕI LOGIC ---

def test_query(user_query: str):
    """
    Hàm kiểm thử: Nhận yêu cầu -> Gọi AI -> In code -> Thực thi -> In kết quả.
    """
    print("\n" + "="*50)
    print(f"▶️ ĐANG KIỂM THỬ YÊU CẦU: '{user_query}'")
    
    if not model:
        print("❌ LỖI: Model Gemini chưa được khởi tạo."); return

    # [SỬA LỖI] Format MỘT LẦN DUY NHẤT với TẤT CẢ các key cần thiết
    prompt = MASTER_PROMPT_TEMPLATE.format(df_info=df_info_str, user_query=user_query)
    
    print("  -> Đang gửi yêu cầu tới Gemini API (timeout sau 30 giây)...")
    try:
        request_options = {"timeout": 30}
        response = model.generate_content(prompt, request_options=request_options)
        
        print("  -> Đã nhận được phản hồi từ API.")
        generated_code = response.text.replace("```python", "").replace("```", "").strip()
        print("\n--- Code do Gemini tạo ra ---")
        print(generated_code)
        print("---------------------------\n")

    except google.api_core.exceptions.DeadlineExceeded:
        print("❌ LỖI: Yêu cầu tới Gemini API đã hết thời gian (timeout)."); return
    except Exception as e:
        print(f"❌ LỖI KHI GỌI GEMINI API: {e}"); return

    try:
        execution_scope = {'df': df}
        exec(generated_code, globals(), execution_scope)
        
        q_result = execution_scope.get('q_result')
        
        if q_result is None: raise ValueError("Code đã chạy nhưng không tìm thấy biến 'q_result'.")
        if not isinstance(q_result, dict): raise TypeError(f"Biến 'q_result' phải là dictionary, nhưng lại là {type(q_result)}.")

        print("✅ Thực thi thành công!")
        print("\n--- Kết quả (biến q_result) ---")
        if not q_result: print("  (Không có kết quả nào được tìm thấy)")
        preview_count = 0
        for video, frames in q_result.items():
            print(f"  - Video '{video}': {len(frames)} frames. Ví dụ: {frames[:5]}")
            preview_count += 1
            if preview_count >= 2: print("  ..."); break
        print("---------------------------------\n")

    except Exception as e:
        print(f"❌ LỖI KHI THỰC THI CODE: {e}")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    print("--- BẮT ĐẦU KIỂM THỬ LÕI LOGIC (GIAI ĐOẠN 1) ---")
    
    test_query("tìm tất cả các frame có xe ô tô")
    test_query("tìm tất cả các xe ô tô màu trắng")
    test_query("tìm các frame có nhiều hơn 1 người")
    test_query("tìm các frame có đúng 1 chiếc xe tải màu xám")
    
    print("--- KIỂM THỬ LÕI LOGIC HOÀN TẤT ---")