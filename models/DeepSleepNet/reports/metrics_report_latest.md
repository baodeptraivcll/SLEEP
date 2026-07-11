# BÁO CÁO ĐÁNH GIÁ: DEEPSLEEPNET (KẾT QUẢ THỰC TẾ)
**Ngày cập nhật:** 08/07/2026
**Cấu hình Data:** `seq_len = 20`, `stride = 1`
**Tập dữ liệu Test:** 16 file được cố định chính xác từ quá trình K-Fold trên Kaggle (43,040 epochs)

*Báo cáo này đã khắc phục triệt để hiện tượng Data Leakage nhờ việc đồng bộ chính xác danh sách file Test từ môi trường Kaggle (Linux) sang Local (Windows).*

## 1. Các chỉ số tổng quát
*   **Độ chính xác tổng thể (Accuracy):** 90.61%
*   **Chỉ số Macro F1 (MF1):** 74.29%
*   **Hệ số Cohen's Kappa:** 0.8089
*   **Thời gian Inference:** 19.12 giây (CPU)
*   **Tổng số tham số mạng (Params):** 22,791,178

## 2. Chỉ số chi tiết từng Pha giấc ngủ
*Lưu ý: Pha N1 luôn là pha khó đoán nhất trong bài toán phân loại giấc ngủ. F1-Score của pha N1 đạt 43.26% là một con số phản ánh đúng thực tế khách quan của mô hình DeepSleepNet.*

## 3. Ảnh biểu đồ đính kèm
Các biểu đồ trực quan đã được tự động vẽ lại dựa trên dữ liệu chuẩn và lưu tại cùng thư mục:
1.  **Ma trận nhầm lẫn (Confusion Matrix):** `confusion_matrix.png`
2.  **Đồ thị Hypnogram:** `hypnogram_comparison.png`
