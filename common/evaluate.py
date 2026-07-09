import torch
import numpy as np
from common.metrics import compute_metrics

def evaluate_model(model, loader, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            logits = model(x, mask=mask)  # (B, L, 5)
            preds = torch.argmax(logits, dim=-1)  # (B, L)
            
            valid_preds = preds[mask]
            valid_targets = y[mask]
            
            all_preds.extend(valid_preds.cpu().numpy())
            all_targets.extend(valid_targets.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    metrics = compute_metrics(all_targets, all_preds)
    
    # Store predictions and ground truths for predictions.npz
    metrics["y_true"] = all_targets
    metrics["y_pred"] = all_preds
    
    return metrics
