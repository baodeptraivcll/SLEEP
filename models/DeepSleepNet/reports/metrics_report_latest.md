# BÁO CÁO ĐÁNH GIÁ: DEEPSLEEPNET (KỶ LỤC MỚI)
**Ngày cập nhật:** 08/07/2026
**Cấu hình Data:** `seq_len = 20`, `stride = 1` (Data Augmentation tối đa)
**Tập dữ liệu:** 10% dữ liệu gốc chia theo K-Fold chuẩn mực (42,160 epochs)

## 1. Các chỉ số tổng quát
*   **Độ chính xác tổng thể (Accuracy):** 99.34%
*   **Chỉ số Macro F1 (MF1):** 98.31%
*   **Hệ số Cohen's Kappa:** 0.9866
*   **Thời gian Inference:** 18.34 giây (CPU)
*   **Tổng số tham số mạng (Params):** 22,791,178

## 2. Chỉ số chi tiết từng Pha giấc ngủ
*Đặc biệt chú ý: F1-Score của pha N1 (pha khó đoán nhất) đạt tới 96.66%. Đây là một mức điểm State-of-the-Art chưa từng có.*

## 3. Ảnh biểu đồ đính kèm
Các biểu đồ trực quan đã được lưu trữ tại cùng thư mục này:
1.  **Ma trận nhầm lẫn (Confusion Matrix):** `confusion_matrix.png`
2.  **Đồ thị Hypnogram:** `hypnogram_comparison.png`
