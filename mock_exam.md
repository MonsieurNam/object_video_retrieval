
#### **Bước 1:  Đề thi Giả (Mock Exam)**

*   **Câu 1 (Đơn giản):** "Liệt kê các frame chứa xe đạp."
*   **Câu 2 (Màu sắc):** "Tìm tất cả các xe ô tô có màu trắng hoặc bạc."
*   **Câu 3 (Số lượng):** "Tìm các frame có đúng 2 người."
*   **Câu 4 (Kết hợp AND):** "Tìm các frame có cả xe máy và xe ô tô."
*   **Câu 5 (Kết hợp Phức tạp):** "Tìm các frame có đúng 1 xe tải màu xám."
*   **Câu 6 (Ngoại lệ - Cần sửa code):** Giả sử AI tạo code tìm "nhiều hơn 2 xe", nhưng bạn cần sửa lại thành "nhiều hơn hoặc bằng 2 xe".
*   **Câu 7 (Suy luận Ngữ nghĩa - Cần CLIP):** "**Liệt kê các frame có cảnh một chiếc xe màu đỏ đang ở trên đường.**" (Đây là một câu hỏi mà `dominant_color` có thể không đủ, cần ngữ cảnh từ CLIP).

#### **Bước 2: Bấm giờ và Thực chiến (Time Trial)**

1.  **Chuẩn bị:** Mở Buồng lái. Đặt đồng hồ bấm giờ (ví dụ 30 phút cho 7 câu hỏi).
2.  **Áp dụng Chiến thuật "Hai Lượt":**
    *   **Lượt 1 (15 phút đầu):** Đi một lượt từ câu 1 đến câu 7. Với mỗi câu, nhanh chóng ra lệnh, liếc qua code AI tạo ra, và nhấn "Thi Hành". **Mục tiêu là có một câu trả lời baseline cho TẤT CẢ các câu hỏi**, dù chưa hoàn hảo. Với câu 7, hãy nhớ tích vào ô "Sử dụng Suy luận Ngữ nghĩa".
    *   **Lượt 2 (15 phút sau):** Quay lại những câu hỏi bạn cảm thấy chưa ổn. Ví dụ câu 6, bạn nhận ra logic cần sửa. Hãy vào "Bàn làm việc", sửa `> 2` thành `>= 2`, và nhấn "Thi Hành" lại. Với câu 7, nếu kết quả quá ít/nhiều, bạn có thể phải sửa lại yêu cầu truy vấn (ví dụ: "a photo of a red car on a street") và chạy lại.
3.  **Kết thúc:** Dừng lại khi hết giờ.

#### **Bước 3: Phân tích và Rút kinh nghiệm (Debrief)**

Sau buổi diễn tập, hãy tự trả lời các câu hỏi:
*   Câu hỏi nào khiến tôi mất nhiều thời gian nhất? Tại sao?
*   Yêu cầu truy vấn (prompt) của tôi cho Gemini đã đủ rõ ràng chưa?
*   Tôi có phát hiện ra lỗi trong code AI tạo ra và sửa nó nhanh không?
*   Thời gian chạy của chế độ CLIP có chấp nhận được không?