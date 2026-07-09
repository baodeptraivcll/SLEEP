import os
import numpy as np
import torch
from torch.utils.data import Dataset

def temporal_shift_no_wrap(x, shift):
    if shift == 0:
        return x
    shifted = np.zeros_like(x)
    if shift > 0:
        shifted[..., shift:] = x[..., :-shift]
    elif shift < 0:
        shifted[..., :shift] = x[..., -shift:]
    return shifted

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
        
        # Apply augmentation only during train split
        augment = (split == "train")
        
        for f in npz_files:
            data = np.load(f)
            x_subject = data['x'].astype(self.dtype) # (N, 3000) or (N, 1, 3000)
            y_subject = data['y'].astype(np.int64)   # (N,)
            
            if len(x_subject.shape) == 2:
                x_subject = np.expand_dims(x_subject, axis=1) # (N, 1, 3000)
                
            # Random Shift Data Augmentation (whole night)
            if augment:
                # Max shift of 10% of 3000 = 300
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
