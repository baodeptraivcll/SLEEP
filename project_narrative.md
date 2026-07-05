# Đề cương Đồ án: Evaluating Deep Learning Architectures for Sleep Staging

> Tài liệu này phác thảo toàn bộ câu chuyện (narrative), hướng đánh giá và cách phân bổ vai trò cho 5 mô hình trong đồ án môn học. Hãy dùng tài liệu này làm kim chỉ nam để phân công task và viết Paper cuối kỳ.

## 1. Bức Tranh Toàn Cảnh (The Big Picture)

Bài toán Sleep Staging (Phân loại giấc ngủ) dựa trên tín hiệu sóng não (EEG) đã chuyển từ kỷ nguyên "trích xuất đặc trưng bằng tay" (Manual Feature Engineering) sang kỷ nguyên "Deep Learning End-to-End". 

Tuy nhiên, câu hỏi nghiên cứu lớn nhất hiện nay là: **"Việc ứng dụng các kiến trúc cực kỳ phức tạp (Transformer, Mamba) có thực sự mang lại lợi ích đột phá so với các kiến trúc nhỏ gọn (CNN-LSTM) hay không, khi xét trên sự cân bằng giữa Độ chính xác (Accuracy) và Chi phí phần cứng (Efficiency)?"**

Đồ án này sẽ tái hiện lại sự tiến hóa của Deep Learning trong Sleep Staging thông qua 5 cột mốc kiến trúc tiêu biểu.

---

## 2. Dàn Diễn Viên (5 Models & Vai trò)

Mỗi thành viên trong nhóm sẽ phụ trách một mô hình, nhưng khi viết paper, mỗi mô hình phải mang một ý nghĩa đại diện riêng biệt:

### 🌟 1. DeepSleepNet (CNN-BiLSTM) - "Kẻ Tiên Phong Lỗi Thời"
- **Citation:** *Supratak et al., "DeepSleepNet: A Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG," IEEE TNSRE, 2017.*
- **Đại diện cho:** Thế hệ End-to-End đầu tiên.
- **Bản chất:** Dùng 2 nhánh CNN lớn/nhỏ để bắt đặc trưng tần số, sau đó nhét qua BiLSTM để học chuỗi thời gian. Dùng Oversampling để bù đắp nhãn thiếu (N1, REM).
- **Hướng viết Paper:** Dùng làm **Baseline (Hệ quy chiếu)**. Phê phán sự cồng kềnh (21 triệu params), việc không thể chạy realtime (do dùng BiLSTM chờ data tương lai), và quy trình train 2 bước rườm rà.

### 🌟 2. TinySleepNet (CNN-LSTM) - "Kẻ Tối Ưu Hóa Dành Cho Edge Computing"
- **Citation:** *Supratak & Dong, "TinySleepNet: An Efficient Deep Learning Model for Sleep Stage Scoring based on Raw Single-Channel EEG," IEEE EMBC, 2020.*
- **Đại diện cho:** Tính hiệu quả, phần cứng giới hạn (Wearables, Smartwatch).
- **Bản chất:** Đập bỏ nhánh CNN thừa, đổi BiLSTM thành LSTM đơn hướng, loại bỏ Oversampling và thay bằng Data Augmentation (Random Shift & Sequence Skip). Kích thước thu nhỏ 15 lần (1.3 triệu params).
- **Hướng viết Paper:** Vũ khí dùng để phản biện các mô hình khổng lồ. Chứng minh rằng một mô hình nhỏ xíu, nếu dùng thủ thuật Augmentation chuẩn xác, hoàn toàn có thể đạt F1-Score ngang ngửa các "quái vật" ở dưới.

### 🌟 3. AttnSleep (CNN-Attention) - "Sự Tinh Tế Của Attention"
- **Citation:** *Eldele et al., "An Attention-Based Deep Learning Approach for Sleep Stage Classification With Single-Channel EEG," IEEE TNSRE, 2021.*
- **Đại diện cho:** Nỗ lực giải quyết điểm mù (Vanishing Gradient) của RNN/LSTM.
- **Bản chất:** Thay vì nhớ một cách tuyến tính như LSTM, AttnSleep dùng cơ chế Attention để model tự "đánh giá" xem đoạn sóng não nào (K-complex, Spindles) là quan trọng nhất để đưa ra quyết định.
- **Hướng viết Paper:** Phân tích khả năng bắt các pha giao thời của giấc ngủ. Attention giúp mô hình nhạy bén hơn khi chuyển từ Wake sang N1 hoặc N2 sang N3.

### 4. SleepTransformer - "Gã Khổng Lồ Toàn Cảnh"
- **Citation:** *Phan et al., "Sleeptransformer: Automatic sleep staging with sequence to sequence initialization and attention," IEEE JBHI, 2022.*
- **Đại diện cho:** Kỷ kỷ nguyên Transformer và Global Context.
- **Bản chất:** Dùng Self-Attention để mô hình có thể "nhìn" thấy toàn bộ vĩ mô của cả một đêm ngủ cùng lúc. Bỏ qua hoàn toàn cấu trúc tuần tự.
- **Hướng viết Paper:** Phô diễn sức mạnh ở các bộ Dataset cực lớn, độ chính xác có thể là cao nhất. Nhưng điểm trừ chí mạng là Complexity $O(N^2)$, khát RAM/VRAM và không thể chạy trên các thiết bị y tế nhỏ gọn.

### 5. WaveMamba (State Space Model) - "Kẻ Thách Thức Tối Thượng"
- **Citation:** *P. Guo and A. Sano, "WaveMamba: Efficient Full-Night Sleep Staging via Wavelets and State Space Models," TechRxiv, 2025.*
- **Đại diện cho:** Công nghệ Next-gen (SSM) sinh ra để thay thế Transformer.
- **Bản chất:** Giữ được khả năng hiểu Global Context vô cực của Transformer, nhưng ép được chi phí tính toán về Linear $O(N)$ giống hệt LSTM. (Đặc biệt kết hợp thêm biến đổi Wavelet để bắt đặc trưng tần số tốt hơn).
- **Hướng viết Paper:** Đóng vai trò là "Trùm cuối". Kết luận của Paper sẽ xoay quanh việc chứng minh WaveMamba chính là sự giao thoa hoàn hảo: Tốc độ và nhẹ như TinySleepNet, nhưng thông minh và toàn cảnh như SleepTransformer.

---

## 3. Tiêu Chí Đánh Giá Trong Paper (Benchmarking Metrics)

Không chỉ so sánh điểm số, Paper của nhóm phải có các bảng/biểu đồ so sánh chéo 4 chiều sau:

1. **Hiệu suất phân loại (Performance):** 
   - Không chỉ nhìn Accuracy tổng. Hãy soi kỹ **F1-Score của nhãn N1** (nhãn khó phân loại nhất). Mô hình nào ăn được N1, mô hình đó thực sự xịn.
   - Sử dụng *Cohen's Kappa* để đánh giá độ đồng thuận với chuyên gia y tế.

2. **Chi phí tính toán (Efficiency):** 
   - Lập bảng so sánh **Model Parameters** (Số triệu tham số).
   - Đo **Inference Time** (Thời gian suy luận 1 chuỗi 15 Epochs mất bao nhiêu mili-giây).

3. **Khả năng nắm bắt ngữ cảnh (Temporal Context):** 
   - Vẽ biểu đồ Hypnogram (Biểu đồ chu kỳ giấc ngủ thực tế vs dự đoán) để xem LSTM, Transformer hay Mamba bám sát thực tế tốt hơn ở các đoạn ngủ chập chờn.

4. **Nghiên cứu cắt bỏ (Ablation Study) - Đặc quyền của TinySleepNet:**
   - Tắt Data Augmentation (Random Shift) của TinySleepNet và show ra biểu đồ cho thấy F1 sụp đổ như thế nào. Qua đó chứng minh Tầm quan trọng của Data Augmentation đối với các mô hình nhỏ.

---

## 4. Workflow 7 Bước Thực Chiến & Luật Chuẩn Hóa (Standardization)

Để đồ án đạt chuẩn "Benchmarking" học thuật (Apple-to-Apple), nhóm phải tuân thủ luật: **Khung sườn Training dùng chung, Ruột DataLoader tùy biến theo kiến trúc**. Dưới đây là 7 bước triển khai:

**Bước 1: Chuẩn hóa Dữ Liệu (Làm chung)**
- Tải file EDF từ nguồn Sleep-EDF. Cắt toàn bộ tín hiệu thành các file `.npz` (mỗi epoch 30 giây, 3000 điểm tín hiệu).
- *Luật bắt buộc:* Cả 5 người xài chung đúng 1 thư mục data `.npz` này, không ai được làm sai lệch.

**Bước 2: Xây dựng Model (Mạnh ai nấy code)**
- 5 người tự code 5 kiến trúc lõi (DeepSleepNet, TinySleepNet, AttnSleep, Transformer, Mamba) bằng PyTorch.
- Ép toàn bộ các model phải nhận được cùng 1 định dạng Input chung: `(Batch, Seq_Len, 1, 3000)`.

**Bước 3: Chuẩn hóa Training Pipeline (Apple-to-Apple)**
- *Cái bắt buộc chung:* Cùng 1 phép chia K-Fold (Random Seed = 42) để đảm bảo tập Train/Validation giống nhau. Cùng dùng Optimizer (Adam), cùng Learning Rate cơ bản, cùng hàm Loss (CrossEntropy).
- *Cái được phép khác:* Kỹ thuật Data Augmentation (ví dụ TinySleepNet dùng Random Shift, DeepSleepNet dùng Oversampling) và Sequence Length (Tùy sức chịu đựng của từng model).

**Bước 4: Ném lên Cloud Training (Mạnh ai nấy chạy)**
- Từng người tự nén code và file `.npz` đẩy lên Google Drive.
- Bật Google Colab (GPU) và bấm Train. Ai xong trước thì lưu cái mô hình tốt nhất (`best_model.pth`) và file log kết quả lại.

**Bước 5: Thống Kê Điểm Số (Đánh giá độc lập)**
- Từng người dùng chung file `evaluate.py`.
- Xuất ra 3 con số quan trọng nhất: **Accuracy, Macro-F1 (đặc biệt soi F1 của N1), Cohen's Kappa**.
- Ghi nhận Thời gian suy luận (Inference Time) và Kích thước model (Params).

**Bước 6: Gộp Kết Quả & Vẽ Biểu Đồ (Trưởng nhóm tổng hợp)**
- Ném tất cả kết quả của 5 người vào 1 đoạn code Python chung để vẽ biểu đồ.
- Sinh ra: Biểu đồ cột so sánh F1-Score, Bảng Confusion Matrix chéo, và Đỉnh cao nhất là **Biểu đồ Hypnogram** đè 5 model lên sóng não thực tế.

**Bước 7: Viết Paper (Lắp ghép kịch bản)**
- Lắp kết quả Bước 6 vào kịch bản ở Mục 2.
- Chốt hạ bằng kết luận: Model nào xịn nhất về tổng thể (chắc chắn là Mamba hoặc Transformer), và model nào là vua của Edge Computing thực chiến (TinySleepNet). Đóng PDF. Nộp!

---

## 5. Yêu Cầu Bắt Buộc Từ Giảng Viên (Nộp Bài & Chấm Điểm)

Dựa trên rubric môn học (Research Based Learning), nhóm BẮT BUỘC phải tuân thủ các quy định khắt khe sau để không bị trừ điểm hoặc rớt môn:

### 5.1. Công Cụ & Định Dạng Nộp Bài
- **Bắt buộc xài WandB (Weights & Biases):** Mọi quá trình Train của 5 ông phải được log lên hệ thống WandB để minh chứng thời gian thực hiện trong học kỳ. (Sẽ update code `train.py` để tự động đẩy log lên).
- **Format Paper:** Chuẩn **IEEE conference**, gõ bằng **LaTeX**. (Nghiêm cấm dùng Word).
- **Độ dài:** Tối thiểu 4 trang.
- **Tài liệu tham khảo (References):** Bắt buộc phải có **ÍT NHẤT 30 bài báo** được trích dẫn (và phải có nhắc đến trong bài).
- **Sản phẩm nộp:** Training logs (link WandB) + PDF Report (từ LaTeX).

### 5.2. Lộ Trình 2 Buổi Thuyết Trình (Presentations)
- **Present 1 (Báo cáo tiến độ):** Giới thiệu bài toán, khảo sát đánh giá (Related Works), đưa ra đề xuất (Methodology), và show kết quả sơ bộ (EDA, EDA sóng não, kết quả chạy thử vài fold).
- **Present 2 (Báo cáo cuối kỳ):** Hoàn thành full paper, full biểu đồ so sánh 5 models.
- *(Điểm cộng):* Nếu Paper mang đi nộp hội nghị/tạp chí và được Accepted / Minor Revision / Major Revision thì sẽ có điểm thưởng tùy rank hội nghị.

### 5.3. Cấu Trúc Bắt Buộc Của Report (Full Paper)
Nhóm chia nhau viết theo đúng sườn này:
1. **Introduction:** Tổng quan bài toán Sleep Staging, tại sao chọn (khó khăn thực tế), nhóm đã giải quyết gì (đánh giá đa chiều 5 models), đóng góp của nhóm.
2. **Related works:** Tóm tắt 30+ bài báo. Nêu bật ưu/khuyết điểm của các phương pháp cũ (RNN/CNN) và mới (Attention/SSM).
3. **Methodology:** Trình bày chi tiết phương pháp, luồng DataLoader chung, và kiến trúc của 5 model tham gia thi đấu.
4. **Experimental and Results:** 
   - **EDA (Exploratory Data Analysis):** Phân tích dữ liệu Sleep-EDF, phân bố nhãn (imbalance), vẽ phổ sóng não.
   - **Metrics:** F1, Kappa, Accuracy.
   - **Training:** Quá trình Finetune, hyperparameters chung (Adam, CrossEntropy).
   - **Kết quả:** Bảng so sánh 5 models. Thảo luận (Discussion) điểm mạnh/yếu từng thằng. **Ablation Study** (bắt buộc cho TinySleepNet).
5. **Conclusion:** Tóm tắt achievements.
6. **References:** Trích dẫn chuẩn IEEE (>30 bài).
