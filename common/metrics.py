import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

def compute_metrics(y_true, y_pred):
    """
    y_true: 1D numpy array of true labels (only valid mask entries)
    y_pred: 1D numpy array of predicted labels (only valid mask entries)
    """
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "kappa": 0.0,
            "per_class_f1": {"W": 0.0, "N1": 0.0, "N2": 0.0, "N3": 0.0, "REM": 0.0},
            "confusion_matrix": np.zeros((5, 5), dtype=int).tolist()
        }
        
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    
    classes = ['W', 'N1', 'N2', 'N3', 'REM']
    per_class_f1_vals = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    per_class_f1 = {classes[i]: float(per_class_f1_vals[i]) for i in range(len(classes))}
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "kappa": float(kappa),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist()
    }
