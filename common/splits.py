import os
import numpy as np
from sklearn.model_selection import KFold

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
    Logic fold:
        For fold i:
            test subjects = subject fold i
            val subjects  = subject fold (i + 1) % K
            train subjects = all remaining subject folds
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
