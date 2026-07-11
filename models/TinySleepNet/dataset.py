import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

def temporal_shift_no_wrap(x, shift):
    if shift == 0:
        return x
    shifted = np.zeros_like(x)
    if shift > 0:
        shifted[..., shift:] = x[..., :-shift]
    elif shift < 0:
        shifted[..., :shift] = x[..., -shift:]
    return shifted

def extract_subject_id(file_path):
    """
    SC4001.npz -> SC400
    SC4002.npz -> SC400
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    return stem[:5]

def get_kfold_splits(data_dir, k=10, seed=42):
    """
    Returns a list of (train_files, val_files, test_files) for subject-wise k-fold.
    """
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    files = sorted(files)
    
    unique_subjects = sorted(list(set(extract_subject_id(f) for f in files)))
    
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [[] for _ in range(k)]
    for fold_idx, (_, val_idx) in enumerate(kf.split(unique_subjects)):
        for idx in val_idx:
            folds[fold_idx].append(unique_subjects[idx])
            
    splits = []
    for i in range(k):
        test_subjects = folds[i]
        val_subjects = folds[(i + 1) % k]
        
        train_subjects = []
        for f_idx in range(k):
            if f_idx != i and f_idx != (i + 1) % k:
                train_subjects.extend(folds[f_idx])
                
        train_files = sorted([f for f in files if extract_subject_id(f) in train_subjects])
        val_files = sorted([f for f in files if extract_subject_id(f) in val_subjects])
        test_files = sorted([f for f in files if extract_subject_id(f) in test_subjects])
        
        splits.append((train_files, val_files, test_files))
        
    return splits

def check_no_subject_leakage(train_files, val_files, test_files):
    train_subjects = {extract_subject_id(f) for f in train_files}
    val_subjects = {extract_subject_id(f) for f in val_files}
    test_subjects = {extract_subject_id(f) for f in test_files}

    assert train_subjects.isdisjoint(val_subjects), "Train and Val overlap!"
    assert train_subjects.isdisjoint(test_subjects), "Train and Test overlap!"
    assert val_subjects.isdisjoint(test_subjects), "Val and Test overlap!"
    print("No subject leakage detected.")

class SleepEDFDataset(Dataset):
    def __init__(self, npz_files, seq_len=20, stride=5, split="train", pad_last=True, dtype=np.float32):
        self.seq_len = seq_len
        self.stride = stride
        self.pad_last = pad_last
        self.dtype = dtype
        self.split = split
        
        self.x_seqs = []
        self.y_seqs = []
        self.masks = []
        
        # Apply augmentation only in train split
        augment = (split == "train")
        
        for f in npz_files:
            data = np.load(f)
            x_subject = data['x'].astype(self.dtype) # (N, 3000) or (N, 1, 3000)
            y_subject = data['y'].astype(np.int64)   # (N,)
            
            if len(x_subject.shape) == 2:
                x_subject = np.expand_dims(x_subject, axis=1) # (N, 1, 3000)
                
            # Random Shift Data Augmentation (whole night)
            if augment:
                shift = np.random.randint(-300, 300) 
                x_subject = temporal_shift_no_wrap(x_subject, shift)
                    
            # Sequence Augmentation (Skip some initial epochs)
            if augment:
                n_skips = np.random.randint(0, 5) # skip 0-4 epochs
                x_subject = x_subject[n_skips:]
                y_subject = y_subject[n_skips:]
                
            N = len(x_subject)
            if N == 0:
                continue
                
            # Generate sequences inside this recording only
            for i in range(0, N, stride):
                if i + seq_len <= N:
                    self.x_seqs.append(x_subject[i : i + seq_len])
                    self.y_seqs.append(y_subject[i : i + seq_len])
                    self.masks.append(np.ones(seq_len, dtype=np.bool_))
                elif pad_last:
                    # Incomplete last window
                    rem = N - i
                    x_pad = np.zeros((seq_len, x_subject.shape[1], x_subject.shape[2]), dtype=self.dtype)
                    x_pad[:rem] = x_subject[i:]
                    
                    y_pad = np.zeros(seq_len, dtype=np.int64)
                    y_pad[:rem] = y_subject[i:]
                    
                    mask = np.zeros(seq_len, dtype=np.bool_)
                    mask[:rem] = True
                    
                    self.x_seqs.append(x_pad)
                    self.y_seqs.append(y_pad)
                    self.masks.append(mask)

    def __len__(self):
        return len(self.x_seqs)

    def __getitem__(self, idx):
        x_seq = torch.from_numpy(self.x_seqs[idx]).float()
        y_seq = torch.from_numpy(self.y_seqs[idx]).long()
        mask_seq = torch.from_numpy(self.masks[idx]).bool()
        return x_seq, y_seq, mask_seq
