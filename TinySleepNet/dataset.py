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
                # Hàm roll sẽ nối đuôi mảng. Nếu ta coi N epoch là 1 chuỗi dài
                # np.roll sẽ tự wrap around
                x_subject = np.roll(x_subject, shift)
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

def get_kfold_splits(data_dir, k=10):
    """Returns a list of (train_files, val_files) for subject-wise k-fold"""
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    files = np.array(sorted(files)) # Sorted for reproducibility
    
    # ponytail: Subject-wise split. Data leakage prevented because one file = one subject
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    splits = []
    
    for train_idx, val_idx in kf.split(files):
        train_files = files[train_idx]
        val_files = files[val_idx]
        splits.append((train_files, val_files))
        
    return splits
