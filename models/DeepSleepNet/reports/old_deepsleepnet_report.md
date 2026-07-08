# Báo Cáo Cài Đặt: DeepSleepNet (Baseline Model)

Đây là tài liệu phân tích kỹ thuật về file `deepsleepnet.py` mà chúng ta vừa code, được thiết kế để khớp 100% với kịch bản (narrative) của bài tập nhóm: Đóng vai trò là **"Kẻ Tiên Phong Lỗi Thời"** - thế hệ End-to-End đầu tiên, làm nền (baseline) cho sự tối ưu của TinySleepNet và sức mạnh của Transformer/Mamba.

## 1. Thiết Kế Đặc Trưng: Sự Cồng Kềnh Mang Tính Lịch Sử
Thay vì tối ưu hóa dung lượng như các mô hình hiện đại, chúng ta đã giữ nguyên sự khổng lồ của DeepSleepNet gốc (Supratak et al., 2017) với tổng cộng **~22.7 triệu tham số**:

### 1.1. Representation Learning (Nhánh CNN Đôi Khổng Lồ)
Mô hình dùng 2 nhánh song song với kích thước filter chênh lệch cực lớn:
- **Nhánh nhỏ (Small Filter)**: `kernel_size = 50`, `stride = 6`. Nhánh này quét qua tín hiệu để bắt các đặc trưng **tần số cao** (Beta, Sleep Spindles).
- **Nhánh lớn (Large Filter)**: `kernel_size = 400`, `stride = 50`. Nhánh này nhìn các cửa sổ tín hiệu rất rộng để bắt các đặc trưng **tần số thấp** (Delta, K-Complexes).
- *Điểm phê phán (Dùng để viết Paper)*: Kích thước Kernel khổng lồ (400) khiến mô hình tiêu tốn quá nhiều VRAM và số lượng phép tính (FLOPs) cực cao so với các mô hình CNN nhúng hiện đại.

### 1.2. Sequence Learning (BiLSTM)
- Dữ liệu giấc ngủ có tính liên tục cao. BiLSTM gồm 2 lớp với 512 hidden units giúp mô hình dự đoán trạng thái dựa trên cả Quá khứ và Tương lai.
- *Điểm phê phán (Dùng để viết Paper)*: Việc dùng **Bidirectional LSTM (Hai chiều)** đồng nghĩa với việc mô hình bắt buộc phải "chờ" dữ liệu tương lai mới có thể đưa ra kết quả cho epoch hiện tại. Nó triệt tiêu hoàn toàn khả năng chạy Real-time (Thời gian thực) trên các thiết bị đeo tay theo dõi giấc ngủ. Hơn nữa, nó chỉ lưu được ngữ cảnh (context) tầm ngắn, thua xa Self-Attention của Transformer hay State-Space của Mamba.

## 2. Kỹ Thuật Tối Ưu Code (Hỗ Trợ 2-Step Training)

Điểm trừ lớn nhất của DeepSleepNet gốc chính là quy trình **Train 2 bước (2-step training)** vô cùng rườm rà. Để tái hiện chân thực nhất điều này phục vụ đánh giá, code mô hình đã được tách ra 2 chế độ rõ rệt thông qua tham số `mode`:

### Bước 1: Pre-training (Chế độ `mode="pretrain"`)
- **Cách hoạt động**: Data loader sẽ phải truyền vào từng epoch đơn lẻ `(Batch, Channels=1, Length=3000)`, áp dụng kỹ thuật *Oversampling* để cân bằng nhãn (N1, REM).
- Mạch tín hiệu chỉ đi qua CNN rồi rẽ ngang vào `self.pretrain_classifier`.
- Mục tiêu: Ép nhánh CNN học cách nhận diện sóng não thô. (Lúc này BiLSTM đứng im).

### Bước 2: Fine-tuning (Chế độ `mode="finetune"`)
- **Cách hoạt động**: Model chuyển sang nhận 4D Tensor dạng Sequence `(Batch, Seq_Len, Channels=1, Length=3000)`. Dữ liệu sẽ đi qua toàn bộ mạng (CNN $\rightarrow$ BiLSTM $\rightarrow$ Shortcut $\rightarrow$ Classifier).
- Người code `train.py` sẽ sử dụng hàm `model.freeze_cnn()` được thiết kế sẵn trong class để khóa cứng toàn bộ trọng số của CNN, dồn toàn lực gradient cho BiLSTM học đặc trưng chuỗi thời gian, và không dùng Oversampling nữa.

---
*Tóm lại, bản implementation này là một "bảo tàng sống" hoàn hảo: Chạy cực kỳ ổn định, code sạch sẽ tối đa, nhưng mang đầy đủ những "khuyết điểm cấu trúc" cần thiết để nhóm bạn có thể mổ xẻ, phê phán và làm bật lên giá trị của TinySleepNet hay MambaSleep trong bài Paper IEEE!*
