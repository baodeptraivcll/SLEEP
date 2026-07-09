# Implementation Plan — Sleep Staging 4 Models with Kaggle + WandB

## 0. Mục tiêu

Dự án đánh giá công bằng 4 kiến trúc deep learning cho bài toán sleep staging trên Sleep-EDF Expanded, subset Sleep Cassette SC.

4 model so sánh:

1. `DeepSleepNet` — CNN + BiLSTM
2. `TinySleepNet` — Lightweight CNN + LSTM
3. `SleepTransformer (Raw-Signal Variant)` — CNN Front-end + Transformer Encoder
4. `MambaSleep` — CNN Front-end + Mamba/State Space Model

Tất cả model phải dùng chung:

- Cùng dữ liệu `.npz` đã preprocessing.
- Cùng kênh EEG `Fpz-Cz`.
- Cùng subject-wise split.
- Cùng `seq_len = 20`.
- Cùng `train_stride = 5`.
- Cùng `val_stride = 20` và `test_stride = 20`.
- Cùng `epochs = 50`.
- Cùng `dtype = float32` cho input.
- Cùng WandB project và cùng tên key log bắt buộc.
- Cùng evaluation protocol.

Mục tiêu là kết quả đủ sạch để viết paper/report so sánh 4 kiến trúc.

---

## 1. Official Experiment Config

Cấu hình chốt cho toàn nhóm:

```text
K-Fold: 10 subject-wise folds
Train/Val/Test: 8/1/1 subject folds
Epochs: 50
Seq length: 20
Train stride: 5
Validation stride: 20
Test stride: 20
Input dtype: float32
Label dtype: int64
Loss: Weighted Cross Entropy
Class weights: train split only
Checkpoint selection: best validation Macro-F1
Final result: test split only
WandB: bắt buộc
Platform: Kaggle Notebook
Recommended mode: mỗi Kaggle run/version chạy đúng 1 fold
```

Lý do chọn `train_stride = 5`:

- Nhẹ hơn `stride = 1` khoảng 5 lần về số training sequences.
- Vẫn giữ overlap để model học transition giữa các sleep stages.
- Thực tế hơn khi train 4 model trên Kaggle.

Lý do giữ `val_stride = test_stride = 20`:

- Validation/test không bị đánh giá trùng epoch.
- Mỗi epoch thật chỉ có đúng 1 prediction.
- Chunk cuối nếu thiếu thì dùng padding + mask.

---

## 2. Kaggle Running Strategy

## 2.1 Vấn đề

Train 4 model × 10 folds × 50 epochs rất dễ gặp:

- Kaggle timeout.
- Mất session.
- Crash GPU/RAM.
- Notebook dừng giữa chừng.
- Không kịp lưu kết quả local.

Vì vậy không nên chạy toàn bộ 10 folds trong một Kaggle session duy nhất.

---

## 2.2 Chiến lược chốt

Mỗi Kaggle run/version chỉ chạy **1 fold** của **1 model**.

Ví dụ:

```text
DeepSleepNet_fold0
DeepSleepNet_fold1
...
DeepSleepNet_fold9

TinySleepNet_fold0
TinySleepNet_fold1
...
TinySleepNet_fold9
```

Cách này giúp:

- Nếu crash thì chỉ mất 1 fold, không mất toàn bộ 10 folds.
- WandB vẫn lưu log từng epoch.
- Best checkpoint được lưu và upload lên WandB artifact.
- Dễ kiểm tra model/fold nào lỗi.
- Dễ chạy song song hoặc chia việc cho 4 người.

Lưu ý thực tế:

- Trong một Kaggle notebook đang chạy, không nên phụ thuộc vào việc notebook tự tạo Kaggle version mới cho fold tiếp theo.
- Cách an toàn hơn là thiết kế notebook/script nhận `FOLD_ID` và `ARCHITECTURE`, rồi mỗi Kaggle run chạy đúng 1 fold.
- Nếu muốn tự động hóa submit nhiều Kaggle versions, có thể làm bằng Kaggle API từ máy local hoặc môi trường ngoài Kaggle. Nhưng trong project này, yêu cầu tối thiểu là code phải hỗ trợ chạy từng fold độc lập.

---

## 2.3 Biến cấu hình cho Kaggle

Mỗi notebook/script phải có phần config đầu file:

```python
ARCHITECTURE = "DeepSleepNet"  # DeepSleepNet, TinySleepNet, SleepTransformer, MambaSleep
FOLD_ID = 0                    # 0..9
RUN_ALL_FOLDS = False          # Kaggle official run nên để False

EPOCHS = 50
SEQ_LEN = 20
TRAIN_STRIDE = 5
VAL_STRIDE = 20
TEST_STRIDE = 20
DTYPE = "float32"
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10
SEED = 42
```

Nếu muốn truyền qua environment variable:

```python
import os

ARCHITECTURE = os.getenv("ARCHITECTURE", "DeepSleepNet")
FOLD_ID = int(os.getenv("FOLD_ID", "0"))
```

---

## 2.4 Output folder trên Kaggle

Tất cả output của mỗi fold lưu vào `/kaggle/working`:

```text
/kaggle/working/results/
└── DeepSleepNet/
    └── fold_0/
        ├── best_model.pth
        ├── metrics.json
        ├── confusion_matrix.npy
        ├── predictions.npz
        └── config.json
```

Không lưu kết quả quan trọng chỉ trong RAM.

Sau mỗi epoch tốt nhất, phải:

1. Save checkpoint local.
2. Update `wandb.summary`.
3. Log best checkpoint lên WandB artifact nếu có thể.

---

## 3. WandB Requirements

## 3.1 WandB là bắt buộc

Mọi quá trình train của 4 model phải log lên WandB để minh chứng quá trình thực hiện trong học kỳ.

Tất cả thành viên phải dùng cùng:

```text
entity = "giabao240806-fpt-university"
project = "Đồng cam mất ngủ"
```

---

## 3.2 Không commit API key lên GitHub

Không hard-code API key thật vào repo public.

Sai:

```python
os.environ["WANDB_API_KEY"] = "wandb_xxx_key_that_is_committed_to_github"
```

Đúng hơn:

```python
import os

os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
os.environ["WANDB_SILENT"] = "true"
```

Trên Kaggle:

- Lưu key bằng Kaggle Secrets nếu có thể.
- Hoặc nhập key ở cell riêng không commit lên GitHub.
- Nếu bắt buộc dùng trực tiếp trong notebook nộp nội bộ, tuyệt đối không push notebook chứa key lên repo public.

---

## 3.3 WandB init chuẩn

Mỗi fold nên là một WandB run riêng.

```python
import os
import wandb

os.environ["WANDB_SILENT"] = "true"

run_name = f"{ARCHITECTURE}_fold{FOLD_ID}_seq{SEQ_LEN}_stride{TRAIN_STRIDE}"

run = wandb.init(
    entity="giabao240806-fpt-university",
    project="Đồng cam mất ngủ",
    name=run_name,
    group=ARCHITECTURE,
    job_type=f"fold_{FOLD_ID}",
    save_code=True,
    resume="allow",
    id=run_name,
    config={
        "architecture": ARCHITECTURE,
        "dataset": "Sleep-EDF-SC",
        "channel": "Fpz-Cz",
        "epochs": 50,
        "seq_len": 20,
        "train_stride": 5,
        "val_stride": 20,
        "test_stride": 20,
        "dtype": "float32",
        "batch_size": BATCH_SIZE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "split": "subject-wise 10-fold",
        "fold_id": FOLD_ID,
        "seed": SEED,
        "loss": "weighted_cross_entropy",
        "class_weights": "train_split_only",
        "checkpoint_metric": "best_val_f1",
        "platform": "kaggle",
        "run_policy": "one_fold_per_kaggle_run"
    }
)
```

Lý do dùng `id=run_name` và `resume="allow"`:

- Nếu Kaggle crash và chạy lại cùng model/fold, WandB có thể nối tiếp run cũ thay vì tạo run rác.
- Giúp bảo toàn log theo fold.

---

## 3.4 WandB keys bắt buộc

Để vẽ gộp 4 model trên cùng biểu đồ, tất cả thành viên phải dùng chính xác các key sau trong `wandb.log()`:

```text
"epoch"
"train_loss"
"val_f1"
"val_kappa"
"val_acc"
"final_fold_acc"
"final_fold_f1"
"final_fold_kappa"
```

Không đổi tên thành:

```text
val_macro_f1
validation_f1
test_f1
fold_f1
accuracy
```

Nếu muốn log thêm metric khác thì được, nhưng các key bắt buộc trên phải giữ nguyên.

---

## 3.5 Log mỗi epoch

Trong mỗi epoch:

```python
wandb.log({
    "epoch": epoch,
    "train_loss": float(train_loss),
    "val_f1": float(val_f1),
    "val_kappa": float(val_kappa),
    "val_acc": float(val_acc),
})
```

Trong đó:

```text
val_f1 = validation Macro-F1
val_kappa = validation Cohen's Kappa
val_acc = validation Accuracy
```

---

## 3.6 Log best-so-far để phòng crash

Khi validation F1 tốt nhất được cải thiện:

```python
if val_f1 > best_val_f1:
    best_val_f1 = val_f1
    best_epoch = epoch
    best_val_acc = val_acc
    best_val_kappa = val_kappa

    torch.save(model.state_dict(), best_model_path)

    wandb.summary["best_epoch"] = int(best_epoch)
    wandb.summary["best_val_f1"] = float(best_val_f1)
    wandb.summary["best_val_acc"] = float(best_val_acc)
    wandb.summary["best_val_kappa"] = float(best_val_kappa)
    wandb.summary["best_train_loss"] = float(train_loss)
    wandb.summary["best_model_path"] = str(best_model_path)
```

Mục tiêu:

- Nếu Kaggle crash trước khi train xong, WandB vẫn giữ best-so-far.
- Dễ biết epoch nào tốt nhất.
- Dễ chứng minh quá trình train thật.

---

## 3.7 Log checkpoint lên WandB Artifact

Sau khi có best checkpoint, nên upload artifact:

```python
artifact = wandb.Artifact(
    name=f"{ARCHITECTURE}_fold{FOLD_ID}_best_model",
    type="model"
)
artifact.add_file(best_model_path)
wandb.log_artifact(artifact)
```

Có thể log artifact khi:

- Mỗi lần best model được cải thiện.
- Hoặc cuối fold, sau khi train xong.

Nếu log mỗi lần cải thiện quá nặng, ít nhất phải log artifact cuối fold.

---

## 3.8 Log final test fold metrics

Sau khi train xong:

1. Load best checkpoint theo validation Macro-F1.
2. Evaluate trên test set của fold đó.
3. Log final test metrics bằng đúng keys bắt buộc.

```python
model.load_state_dict(torch.load(best_model_path, map_location=device))

test_acc, test_f1, test_kappa, test_cm = evaluate_model(
    model=model,
    loader=test_loader,
    device=device
)

wandb.log({
    "epoch": EPOCHS,
    "final_fold_acc": float(test_acc),
    "final_fold_f1": float(test_f1),
    "final_fold_kappa": float(test_kappa),
})

wandb.summary["final_fold_acc"] = float(test_acc)
wandb.summary["final_fold_f1"] = float(test_f1)
wandb.summary["final_fold_kappa"] = float(test_kappa)
```

Không dùng validation metrics làm final result.

---

## 4. Subject-Wise Split

## 4.1 Lỗi cần tránh

Dữ liệu `.npz` đang lưu theo recording/night:

```text
SC4001.npz
SC4002.npz
SC4011.npz
SC4012.npz
```

Trong đó:

```text
SC4001 và SC4002 cùng subject SC400
SC4011 và SC4012 cùng subject SC401
```

Không được chia theo file `.npz`, vì sẽ gây data leakage.

Sai:

```text
Train: SC4001.npz
Test:  SC4002.npz
```

Đúng:

```text
SC4001.npz và SC4002.npz luôn cùng split
```

---

## 4.2 Extract subject ID

```python
import os
import re


def extract_subject_id(file_path):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    match = re.match(r"^(SC\d{3})", stem)
    if match is None:
        raise ValueError(f"Cannot extract subject_id from filename: {stem}")
    return match.group(1)
```

Ví dụ:

```text
SC4001 -> SC400
SC4002 -> SC400
SC4011 -> SC401
SC4012 -> SC401
```

---

## 4.3 K-Fold protocol

Dùng `K = 10`.

Mỗi fold:

```text
Train: 8 subject folds
Val:   1 subject fold
Test:  1 subject fold
```

Logic:

```text
For fold i:
    test_subjects = subject_fold[i]
    val_subjects  = subject_fold[(i + 1) % 10]
    train_subjects = remaining 8 folds
```

Mỗi subject xuất hiện trong test đúng 1 lần qua 10 folds.

---

## 4.4 Leakage check bắt buộc

Mỗi fold phải check:

```python
def check_no_subject_leakage(train_files, val_files, test_files):
    train_subjects = {extract_subject_id(f) for f in train_files}
    val_subjects = {extract_subject_id(f) for f in val_files}
    test_subjects = {extract_subject_id(f) for f in test_files}

    assert train_subjects.isdisjoint(val_subjects), "Leak between train and val"
    assert train_subjects.isdisjoint(test_subjects), "Leak between train and test"
    assert val_subjects.isdisjoint(test_subjects), "Leak between val and test"
```

Log bắt buộc:

```text
Fold 0
Train subjects: ... | Train recordings: ...
Val subjects: ...   | Val recordings: ...
Test subjects: ...  | Test recordings: ...
No subject leakage detected.
```

---

## 5. Dataset Loader & Sequence Construction

## 5.1 Input dtype

Trong dataset loader:

```python
x = x.astype("float32")
y = y.astype("int64")
```

Nếu `x` có shape `(num_epochs, 3000)`, thêm channel dimension:

```python
if x.ndim == 2:
    x = x[:, None, :]
```

Output tensor:

```python
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).long()
mask = torch.from_numpy(mask).bool()
```

---

## 5.2 Sequence shape

Tất cả model dùng cùng input:

```text
x_seq:    (B, 20, 1, 3000)
y_seq:    (B, 20)
mask_seq: (B, 20)
```

Model output:

```text
logits: (B, 20, 5)
```

---

## 5.3 Không nối sequence giữa recordings

Không được concatenate tất cả files rồi mới chunk sequence.

Sai:

```text
all_x = concat(recording_1, recording_2, ..., recording_n)
create_sequences(all_x)
```

Đúng:

```text
for each .npz recording:
    load x, y
    create sequences only inside this recording
```

Lý do:

- Không nối đêm 1 sang đêm 2.
- Không nối subject này sang subject khác.
- Không tạo transition giả.

---

## 5.4 Train stride

Train dùng:

```text
train_stride = 5
```

Ví dụ với `seq_len = 20`:

```text
seq 1: epoch 0–19
seq 2: epoch 5–24
seq 3: epoch 10–29
```

---

## 5.5 Validation/Test stride

Validation/test dùng:

```text
val_stride = 20
test_stride = 20
```

Ví dụ:

```text
seq 1: epoch 0–19
seq 2: epoch 20–39
seq 3: epoch 40–59
```

Không dùng `stride = 5` cho val/test vì sẽ đánh giá trùng epoch.

---

## 5.6 Padding + mask cho val/test

Nếu chunk cuối không đủ 20 epochs:

```text
real epochs = 7
seq_len = 20
mask = [1,1,1,1,1,1,1,0,0,...,0]
```

Loss và metrics chỉ tính vị trí `mask == 1`.

Không được bỏ phần dư cuối recording ở validation/test.

---

## 6. Training Protocol

## 6.1 Seed

```python
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 6.2 Loss with mask

```python
logits = model(x, mask=mask)  # (B, L, 5)

logits_flat = logits.reshape(-1, 5)
y_flat = y.reshape(-1)
mask_flat = mask.reshape(-1).bool()

loss = criterion(logits_flat[mask_flat], y_flat[mask_flat])
```

---

## 6.3 Class weights

Weighted Cross Entropy dùng class weights tính từ train split only.

Đúng:

```text
class_weights = count labels from train_files only
```

Sai:

```text
class_weights = count labels from all files
class_weights = count labels from train + val + test
```

---

## 6.4 Checkpoint

Checkpoint chọn theo validation Macro-F1:

```text
best_model.pth = model with highest val_f1
```

Không chọn checkpoint theo test score.

---

## 6.5 Early stopping

Official config:

```text
epochs = 50
early_stopping_patience = 10
```

Nếu dùng early stopping:

- Vẫn log mỗi epoch đã chạy.
- Nếu stop trước 50 epochs, ghi rõ trong WandB summary.
- Final evaluation vẫn phải dùng best checkpoint trên test split.

---

## 7. Evaluation Protocol

Mỗi fold test tính:

```text
Accuracy
Macro-F1
Cohen's Kappa
Per-class F1: W, N1, N2, N3, REM
Confusion Matrix
```

Sau 10 folds, report:

```text
Mean ± Std
```

Bảng paper chính:

| Model | ACC | MF1 | Kappa | F1-W | F1-N1 | F1-N2 | F1-N3 | F1-REM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSleepNet | mean ± std | mean ± std | mean ± std | | | | | |
| TinySleepNet | mean ± std | mean ± std | mean ± std | | | | | |
| SleepTransformer Raw | mean ± std | mean ± std | mean ± std | | | | | |
| MambaSleep | mean ± std | mean ± std | mean ± std | | | | | |

---

## 8. Model-Specific Notes

## 8.1 DeepSleepNet

Input:

```text
(B, 20, 1, 3000)
```

Output:

```text
(B, 20, 5)
```

Architecture:

```text
Small-filter CNN branch
Large-filter CNN branch
Feature concatenation
BiLSTM
Classifier
```

Report note:

```text
We used one-step end-to-end training instead of the original two-step training strategy to maintain a consistent training protocol across all compared models.
```

---

## 8.2 TinySleepNet

Input:

```text
(B, 20, 1, 3000)
```

Output:

```text
(B, 20, 5)
```

Report note:

```text
We used stateless LSTM training to simplify batching and maintain consistency with the shared sequence dataloader.
```

---

## 8.3 SleepTransformer Raw-Signal Variant

Input:

```text
(B, 20, 1, 3000)
```

Output:

```text
(B, 20, 5)
```

Architecture:

```text
CNN front-end
Positional encoding
Transformer encoder
Classifier
```

Report note:

```text
We report this model as SleepTransformer (Raw-Signal Variant), because the shared pipeline uses raw EEG epochs instead of spectrogram features.
```

---

## 8.4 MambaSleep

Input:

```text
(B, 20, 1, 3000)
```

Output:

```text
(B, 20, 5)
```

Architecture:

```text
CNN front-end
Mamba/SSM sequence block
Classifier
```

Report note:

```text
We implemented a Mamba-based sleep staging model adapted to raw EEG epoch sequences.
```

---

## 9. Required Output Files Per Fold

Mỗi Kaggle fold run phải tạo:

```text
results/{model_name}/fold_{fold}/best_model.pth
results/{model_name}/fold_{fold}/metrics.json
results/{model_name}/fold_{fold}/confusion_matrix.npy
results/{model_name}/fold_{fold}/predictions.npz
results/{model_name}/fold_{fold}/config.json
```

`predictions.npz` nên có:

```text
y_true
y_pred
subject_id
recording_id
mask
```

---

## 10. Aggregation After All Kaggle Versions

Sau khi chạy đủ 10 folds cho một model:

1. Download hoặc collect `metrics.json` từ 10 fold outputs.
2. Tạo `summary.csv`.
3. Tính mean ± std.
4. Gộp confusion matrix.
5. Lấy MF1 theo fold để làm statistical test.

`summary.csv` format:

```text
fold,accuracy,macro_f1,kappa,f1_W,f1_N1,f1_N2,f1_N3,f1_REM,train_time,inference_time,num_params
```

---

## 11. Statistical Test

Dùng MF1 của 10 folds:

```text
DeepSleepNet:     [fold0, ..., fold9]
TinySleepNet:     [fold0, ..., fold9]
SleepTransformer: [fold0, ..., fold9]
MambaSleep:       [fold0, ..., fold9]
```

Khuyến nghị:

```text
Wilcoxon signed-rank test
```

Vì chỉ có 10 folds, Wilcoxon an toàn hơn paired t-test.

---

## 12. Agent Task Checklist

## Data Pipeline Agent

Phải đảm bảo:

```text
[ ] Subject-wise split, không split theo npz file
[ ] SC4001 và SC4002 luôn cùng split
[ ] No leakage check
[ ] seq_len = 20
[ ] train_stride = 5
[ ] val_stride = 20
[ ] test_stride = 20
[ ] padding + mask cho val/test
[ ] x float32, y int64
[ ] không nối sequence giữa recordings
```

## Model Agents

Mỗi model phải đảm bảo:

```text
[ ] Input shape: (B, 20, 1, 3000)
[ ] Output shape: (B, 20, 5)
[ ] Loss chạy được với mask
[ ] Metrics bỏ qua padded epochs
[ ] WandB log đúng keys bắt buộc
[ ] Best checkpoint theo val_f1
[ ] Final test metrics log bằng final_fold_acc/f1/kappa
```

## WandB/Kaggle Agent

Phải đảm bảo:

```text
[ ] Mỗi fold là một WandB run riêng
[ ] Run name chứa architecture + fold + seq20 + stride5
[ ] resume="allow"
[ ] Log mỗi epoch
[ ] Update wandb.summary khi có best_val_f1
[ ] Log final fold metrics
[ ] Save checkpoint vào /kaggle/working
[ ] Log best checkpoint artifact
[ ] Không commit API key lên GitHub public
```

---

## 13. Final Acceptance Criteria

Project đạt chuẩn khi:

1. 4 model dùng cùng subject-wise 10-fold split.
2. Không có subject leakage.
3. 4 model dùng cùng `seq_len = 20`.
4. 4 model dùng cùng `train_stride = 5`.
5. Validation/test dùng stride 20 + padding + mask.
6. Input dùng `float32`.
7. Epoch official là 50.
8. Class weights tính từ train only.
9. Checkpoint chọn theo validation Macro-F1.
10. Final result tính trên test fold only.
11. Mỗi Kaggle run/version chạy 1 model + 1 fold.
12. Mỗi fold có WandB run riêng.
13. WandB log đúng keys bắt buộc.
14. WandB summary lưu best-so-far để phòng crash.
15. Best checkpoint được lưu local và log artifact.
16. Có `metrics.json` cho từng fold.
17. Có `summary.csv` sau 10 folds.
18. Paper report mean ± std qua 10 folds.

---

## 14. Notes For Paper

Có thể viết trong report:

```text
To avoid subject leakage, we performed 10-fold cross-validation at the subject level. All recordings from the same subject were assigned to the same split. Each model used the same sequence length of 20 epochs. Training sequences were generated with a stride of 5 to reduce computational cost while preserving temporal overlap, whereas validation and test sequences used non-overlapping windows with a stride of 20. Padding and masking were applied to the final incomplete sequence of each recording to ensure that no validation/test epoch was discarded.
```

Về Kaggle + WandB:

```text
Due to computational limits on Kaggle, each fold was trained as an independent Kaggle run/version and logged as a separate WandB run. This allowed intermediate best validation results, checkpoints, and final test metrics to be preserved even if a session crashed.
```

Về kết quả thấp hơn paper gốc:

```text
Any performance gap compared with the original papers may be due to strict subject-wise evaluation, shared single-channel raw EEG input, simplified training protocols, and architecture adaptations required for a unified comparison pipeline.
```

---

## 15. Do Not Do

Không được:

- Split theo `.npz` file.
- Dùng validation làm final test.
- Dùng test score để chọn checkpoint.
- Tính class weights từ toàn dataset.
- Dùng train_stride khác nhau giữa 4 model.
- Dùng val/test stride = 5.
- Drop phần dư cuối recording ở val/test.
- Tính metric trên padded epochs.
- Nối sequence giữa 2 recording.
- Commit WandB API key lên GitHub public.
