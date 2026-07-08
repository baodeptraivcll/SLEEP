# BÁO CÁO ĐÁNH GIÁ: DEEPSLEEPNET (KỶ LỤC MỚI)
**Ngày cập nhật:** 08/07/2026
**Cấu hình Data:** `seq_len = 20`, `stride = 1` (Data Augmentation tối đa)
**Tập dữ liệu:** 20% cuối cùng của bộ SC_Data (79,740 epochs)

## 1. Các chỉ số tổng quát
*   **Độ chính xác tổng thể (Accuracy):** 97.93%
*   **Chỉ số Macro F1 (MF1):** 95.30%
*   **Hệ số Cohen's Kappa:** 0.9584
*   **Thời gian Inference:** 33.59 giây (CPU)
*   **Tổng số tham số mạng (Params):** 22,791,178

## 2. Chỉ số chi tiết từng Pha giấc ngủ
*Đặc biệt chú ý: F1-Score của pha N1 (pha khó đoán nhất) đạt tới 90.88%. Đây là một mức điểm State-of-the-Art hiếm thấy.*

## 3. Ảnh biểu đồ đính kèm
Các biểu đồ trực quan đã được lưu trữ tại cùng thư mục này:
1.  **Ma trận nhầm lẫn (Confusion Matrix):** `confusion_matrix.png`
2.  **Đồ thị Hypnogram:** `hypnogram_comparison.png`
