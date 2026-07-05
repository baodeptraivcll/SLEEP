# Workflow Đánh Giá Kiến Trúc Deep Learning (Sleep Staging)

> [!NOTE]
> Đây là luồng xử lý (workflow) chuẩn để đảm bảo công bằng khi so sánh 5 mô hình trên tập Sleep-EDF Expanded (Sleep Cassette). Toàn bộ code sẽ viết bằng PyTorch.

## Giai Đoạn 1: Tiền Xử Lý & Chuẩn Bị Dữ Liệu (Data Preprocessing)
Mục tiêu: Chuyển dữ liệu gốc thành định dạng chuẩn (Tensors) để đưa vào mô hình.

1. **Tải Dữ Liệu**: Tải file `.edf` (tín hiệu PSG) và file annotations từ tập Sleep-EDF Expanded (chỉ lấy 78 subjects nhóm Sleep Cassette - SC, tương đương 153 recordings).
2. **Cắt Bớt Nhãn Wake Thừa (Wake Trimming)**: 
   - *Định nghĩa logic*: Tìm epoch non-Wake đầu tiên (Sleep Onset) và cuối cùng (Final Wake) của toàn bộ bản ghi. Giữ lại đúng 30 phút W (60 epochs) trước Onset và 30 phút W sau Final Wake. Phần W thừa ở hai đầu bị cắt bỏ hoàn toàn. Giúp cân bằng dữ liệu, tránh việc W áp đảo.
3. **Lọc Nhiễu & Chuẩn Hóa (Filtering & Normalization)**: 
   - **Bandpass Filter**: Lọc dải thông từ 0.5 Hz đến 30 Hz hoặc 45 Hz (sử dụng bộ lọc Butterworth) để loại bỏ nhiễu trôi nền và nhiễu điện lưới/cơ học.
   - **Z-score Normalization**: Chuẩn hóa tín hiệu về phân phối chuẩn (Standardization) theo từng subject hoặc bản ghi để tránh sai khác biên độ điện thế giữa các bệnh nhân.
4. **Trích Xuất Tín Hiệu (Signal Extraction)**: 
   - Lấy 1 kênh EEG duy nhất (thường dùng `Fpz-Cz`) cho tất cả các mô hình để đảm bảo tính công bằng tuyệt đối.
   - Downsample về 100Hz để đồng bộ với đa số paper.
5. **Cắt Epoch**: Cắt tín hiệu liên tục thành các đoạn 30 giây (Mỗi đoạn = 3000 data points).
6. **Chuẩn Hóa Nhãn (Label Mapping)**: Gộp các nhãn theo chuẩn AASM: W, N1, N2, N3 (bao gồm N4), REM. Loại bỏ MOVEMENT / UNKNOWN.
7. **Lưu Trữ**: Lưu từng subject thành file `.npz` (chứa mảng `x`: tín hiệu, `y`: nhãn).

## Giai Đoạn 2: Xây Dựng Pipeline Dữ Liệu (Data Pipeline)
Mục tiêu: Đưa dữ liệu vào mô hình mà không bị Data Leakage và đúng bản chất kiến trúc.

1. **K-Fold Subject-Wise Split**: 
   > [!CAUTION]
   > K-Fold phải chia theo **Subject ID (78 người)**. Nếu chia theo 153 recordings, 2 đêm của cùng 1 người sẽ rơi vào cả Train và Test, gây Data Leakage nghiêm trọng.
   - Chia 78 subjects thành $K=10$ folds (khoảng 7-8 người/fold). 
   - Train: 8 folds | Validation: 1 fold (Early Stopping) | Test: 1 fold.
2. **Sequence-of-Epochs Dataloader**: 
   - **Cách tạo chuỗi**: Cắt các chuỗi dài $L$ (ví dụ: $L=20$). 
   - **Giới hạn biên**: Tuyệt đối **không** tạo sequence nối giữa 2 đêm (recording) khác nhau hoặc 2 người khác nhau.
   - **Cơ chế Stride (Train)**: Dùng stride = 1 (overlap) để tăng cường dữ liệu.
   - **Cơ chế Stride (Val/Test)**: Dùng stride = $L$ (non-overlapping). Đối với phần dư (remainder) ở cuối bản ghi nếu không chia hết cho $L$, sẽ dùng kĩ thuật Padding & Masking (hoặc window cuối lùi lại overlap) để đảm bảo **không có epoch nào bị bỏ sót**, và mỗi epoch chỉ được đánh giá 1 lần.
   - Định dạng Input chung: `(Batch_Size, Seq_Len, Channels=1, 3000)`. 
   > [!IMPORTANT]
   > **Ngoại lệ cho AttnSleep**: Kiến trúc gốc của AttnSleep (Eldele et al.) là **Epoch-wise** (không học chuỗi). Do đó, mô hình này sẽ nhận input `(Batch_Size, Channels=1, 3000)` thay vì sequence. Việc này nhằm tôn trọng thiết kế gốc (true-to-paper). **Tuyệt đối đảm bảo**: AttnSleep vẫn phải dùng chung bộ Index phân chia K-Fold y hệt như 4 mô hình sequence còn lại để tránh leak data và đảm bảo công bằng.

## Giai Đoạn 3: Cài Đặt Kiến Trúc (Model Implementation)
Triển khai 5 mô hình bằng PyTorch `nn.Module`:
1. **`DeepSleepNet` (CNN-BiLSTM) <--- PHẦN VIỆC CỦA BẠN (CHÚNG TA SẼ FOCUS VÀO ĐÂY)**
2. `TinySleepNet` (CNN-LSTM nhẹ)
3. `AttnSleep` (CNN-Attention)
4. `SleepTransformer` (Raw-Signal Variant)
5. `MambaSleep` (State Space Model)

> [!IMPORTANT]
> **Ngoại lệ Kiến Trúc**:
> - **AttnSleep**: Kiến trúc gốc là **Epoch-wise** (không học chuỗi). Do đó, mô hình này sẽ nhận input `(Batch_Size, Channels=1, 3000)` thay vì sequence. 
> - **SleepTransformer**: Kiến trúc gốc (Phan et al.) dùng Log Power Spectrogram (2D). Để đảm bảo tính đồng nhất của pipeline, mô hình này sẽ được bổ sung một tầng CNN Front-end nhẹ để xử lý Raw Signal, và sẽ được report với tên `SleepTransformer (Raw-Signal Variant)`.

## Giai Đoạn 4: Quá Trình Huấn Luyện (Training)
1. **Khả Năng Tái Lập (Reproducibility)**: Cố định random seeds (PyTorch, NumPy, Python). Ghi log các thông số hệ thống.
2. **Hyperparameters Chuẩn**: 
   - Optimizer: Adam (hoặc AdamW).
   - Loss: **Weighted Cross-Entropy Loss** (gán trọng số phạt cao hơn cho lớp thiểu số như N1, N3).
   > [!WARNING]
   > Tuyệt đối **không dùng Oversampling** (nhân bản epoch) cho dữ liệu huấn luyện dạng chuỗi (Sequence-to-Sequence), vì điều này sẽ phá vỡ tính liên tục thời gian của chuỗi giấc ngủ. Chỉ dùng class weights.
   - Learning rate: ~1e-3 hoặc 1e-4.
3. **Các Đơn Giản Hóa Huấn Luyện (Training Simplifications)**: 
   - **DeepSleepNet**: Paper gốc dùng 2-step training (pretrain CNN, finetune toàn bộ). Ở đây sẽ sử dụng **1-step training** bằng Weighted CE để đồng bộ luồng chạy chung.
   - **TinySleepNet**: Paper gốc dùng Stateful LSTM (giữ hidden state liên tục giữa các batch). Ở đây sẽ dùng **Stateless LSTM** (reset hidden state mỗi batch) để tránh làm phức tạp Dataloader. 
   *(Cả hai điểm này sẽ được ghi chú rõ trong báo cáo nếu kết quả hơi thấp hơn paper gốc).*
4. **Early Stopping**: Dừng sớm nếu Validation Loss không giảm sau 10-15 epochs.
5. **Lưu Checkpoint**: Lưu lại tệp `.pth` của model có Validation MF1 cao nhất ở mỗi fold.

## Giai Đoạn 5: Đánh Giá & Báo Cáo (Evaluation)
1. **Metrics Hiệu Suất** (Yêu cầu khắt khe của paper): Tính trên tập Test của 10 folds. 
   - *Lưu ý*: Do K=10 nhưng chỉ có 78 subjects (mỗi fold 7-8 người), phương sai sẽ lớn. Bắt buộc báo cáo **Mean ± Std** của 10 folds cho mọi metric.
   - **Cohen's Kappa ($\kappa$)**: Bắt buộc phải có để đánh giá độ đồng thuận.
   - Accuracy (ACC)
   - Macro F1-score (MF1)
   - Per-class F1-score (W, N1, N2, N3, REM).
2. **Chi Phí Tính Toán**: So sánh các model về Parameters, FLOPs, Thời gian Train/Inference.
3. **Kiểm Định Thống Kê**: Wilcoxon signed-rank test hoặc Paired t-test trên điểm số MF1 của 10 folds.
4. **Visualizations**:
   - Confusion Matrix (gộp của 10 folds).
   - **Hypnogram**: Chọn 3 subject có điểm MF1 tương ứng là Cao Nhất (Best), Trung Bình (Median), và Thấp Nhất (Worst) để vẽ hypnogram, nhằm minh bạch kết quả và tránh nghi ngờ cherry-picking (chỉ chọn ca dễ để vẽ hình).
5. **So Sánh Báo Cáo Gốc (Sanity Check)**: 
   - Lập một bảng phụ để đối chiếu điểm MF1 đạt được ở pipeline này so với MF1 công bố trong paper gốc của từng mô hình.
   - Thảo luận (nếu có): Lý giải mức độ hao hụt điểm số nếu có do những Đơn giản hóa (Training Simplifications) hoặc Adaptation kiến trúc (như SleepTransformer raw-signal) gây ra, để chứng minh kết quả thấp là do thay đổi cấu trúc chứ không phải do bug code.

---
## Mục Tiêu Cụ Thể Của Chúng Ta (DeepSleepNet Task)
Vì đây là bài tập nhóm và phần việc của bạn là **DeepSleepNet**, chúng ta sẽ:
1. Tập trung code kiến trúc `DeepSleepNet` thật tối ưu (tối ưu memory, tốc độ hội tụ) trong framework PyTorch.
2. Đảm bảo model `deepsleepnet.py` cắm vào là chạy mượt với luồng Dataloader chung của nhóm.
3. Chuẩn bị tài liệu báo cáo riêng cho DeepSleepNet: Nó hoạt động ra sao? Lý giải vì sao dùng 1-step training thay vì 2-step training như gốc? Lợi thế của nhánh CNN đôi kết hợp BiLSTM so với các kiến trúc đơn giản là gì?
