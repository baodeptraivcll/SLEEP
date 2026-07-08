import torch
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model, loader, device='cuda', verbose=True):
    model.eval()
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x) # (B, Seq_Len, 5)
            
            # Reshape để tính metric
            out = out.view(-1, out.size(-1)) # (B * Seq_Len, 5)
            y = y.view(-1) # (B * Seq_Len)
            
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    inf_time = time.time() - start_time
    
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    params = count_parameters(model)
    
    if verbose:
        print(f"Model Params: {params:,}")
        print(f"Inference Time (Total): {inf_time:.4f}s")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print("Confusion Matrix:")
        print(cm)
    
    return acc, macro_f1, kappa, cm

