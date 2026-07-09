import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

class SleepEDFDataset(Dataset):
    def __init__(self, npz_files, augment=False, seq_length=15):
        """npz_files: list of paths to patient .npz files"""
        self.seq_length = seq_length
        self.augment = augment
        
        self.x_seqs = []
        self.y_seqs = []
        
        for f in npz_files:
            data = np.load(f)
            x_subject = data['x'].astype(np.float32) # (N, 3000) or (N, 1, 3000)
            y_subject = data['y'].astype(np.int64)   # (N,)
            
            if len(x_subject.shape) == 2:
                x_subject = np.expand_dims(x_subject, axis=1) # (N, 1, 3000)
                
            # Random Shift Data Augmentation (cho cả đêm) - Tương đương tác giả np.roll toàn chuỗi
            if self.augment:
                # Dịch chuyển tối đa 10% của 3000 = 300
                shift = np.random.randint(-300, 300) 
                
                # Zero-padding shift (Tránh cuộn tín hiệu từ cuối lên đầu)
                shifted_x = np.zeros_like(x_subject)
                if shift > 0:
                    shifted_x[..., shift:] = x_subject[..., :-shift]
                elif shift < 0:
                    shifted_x[..., :shift] = x_subject[..., -shift:]
                else:
                    shifted_x = x_subject.copy()
                    
                x_subject = shifted_x
                
                if shift < 0:
                    x_subject = x_subject[:-1]
                    y_subject = y_subject[:-1]
                elif shift > 0:
                    x_subject = x_subject[1:]
                    y_subject = y_subject[1:]
                    
            # Sequence Augmentation (Skip một số epoch đầu)
            if self.augment:
                n_skips = np.random.randint(0, 5) # Bỏ qua 0-4 epochs
                x_subject = x_subject[n_skips:]
                y_subject = y_subject[n_skips:]
                
            # Chia thành các sequence độ dài seq_length
            # Bỏ qua phần lẻ ở cuối
            n_seqs = len(x_subject) // seq_length
            if n_seqs > 0:
                x_seq = x_subject[:n_seqs * seq_length].reshape(n_seqs, seq_length, 1, 3000)
                y_seq = y_subject[:n_seqs * seq_length].reshape(n_seqs, seq_length)
                self.x_seqs.append(x_seq)
                self.y_seqs.append(y_seq)
        
        if len(self.x_seqs) > 0:
            self.x_seqs = np.concatenate(self.x_seqs, axis=0) # (Total_Seqs, 15, 1, 3000)
            self.y_seqs = np.concatenate(self.y_seqs, axis=0) # (Total_Seqs, 15)
        else:
            self.x_seqs = np.array([])
            self.y_seqs = np.array([])

    def __len__(self):
        return len(self.y_seqs)

    def __getitem__(self, idx):
        x_val = self.x_seqs[idx].copy()
        y_val = self.y_seqs[idx].copy()
        
        return torch.FloatTensor(x_val), torch.LongTensor(y_val)

def extract_subject_id(file_path):
    # Lấy 5 ký tự đầu tên file (e.g. SC400)
    return os.path.basename(file_path)[:5]

def check_no_subject_leakage(train_files, val_files, test_files):
    train_subjects = {extract_subject_id(f) for f in train_files}
    val_subjects = {extract_subject_id(f) for f in val_files}
    test_subjects = {extract_subject_id(f) for f in test_files}

    assert train_subjects.isdisjoint(val_subjects), f"Leakage train-val: {train_subjects & val_subjects}"
    assert train_subjects.isdisjoint(test_subjects), f"Leakage train-test: {train_subjects & test_subjects}"
    assert val_subjects.isdisjoint(test_subjects), f"Leakage val-test: {val_subjects & test_subjects}"
    return True

def get_kfold_splits(data_dir, k=10):
    """Returns a list of (train_files, val_files, test_files) for strict subject-wise k-fold"""
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    files = np.array(sorted(files))
    
    # Gom nhóm các file theo Bệnh nhân (Subject)
    subject_files = {}
    for f in files:
        subject_id = extract_subject_id(f)
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(f)
        
    subject_ids = np.array(sorted(list(subject_files.keys())))
    
    # Chia K-Fold trên danh sách Bệnh nhân
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    subject_folds = []
    for _, test_idx in kf.split(subject_ids):
        subject_folds.append(set(subject_ids[test_idx]))
        
    splits = []
    for fold_idx in range(k):
        test_subjects = subject_folds[fold_idx]
        val_subjects = subject_folds[(fold_idx + 1) % k]
        train_subjects = set(subject_ids) - test_subjects - val_subjects
        
        train_files, val_files, test_files = [], [], []
        
        for subject_id in sorted(train_subjects):
            train_files.extend(subject_files[subject_id])
        for subject_id in sorted(val_subjects):
            val_files.extend(subject_files[subject_id])
        for subject_id in sorted(test_subjects):
            test_files.extend(subject_files[subject_id])
            
        train_files = np.array(sorted(train_files))
        val_files = np.array(sorted(val_files))
        test_files = np.array(sorted(test_files))
        
        # Kiểm tra Data Leakage
        check_no_subject_leakage(train_files, val_files, test_files)
        
        splits.append((train_files, val_files, test_files))
        
    return splits
