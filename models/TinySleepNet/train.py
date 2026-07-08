import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import f1_score
from evaluate import evaluate_model

def train_model(model, train_loader, val_loader, class_weights=None, epochs=50, device='cuda', save_path='best_model.pth'):
    model = model.to(device)
    
    # ponytail: Weighted CrossEntropy for handling N1 minority class
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Bản gốc dùng lr 10^-4 cho RNN
    
    best_val_f1 = 0.0
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x) # out shape: (B, Seq_Len, 5)
            
            # Reshape để tính loss vì nn.CrossEntropyLoss yêu cầu (N, C)
            out = out.view(-1, out.size(-1)) # (B * Seq_Len, 5)
            y = y.view(-1) # (B * Seq_Len)
            
            loss = criterion(out, y)
            loss.backward()
            
            # Tác giả kẹp Gradient Clipping threshold=5.0 chống nổ gradient RNN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_acc, val_f1, val_kappa, _ = evaluate_model(model, val_loader, device=device, verbose=False)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val F1: {val_f1:.4f} - Val Kappa: {val_kappa:.4f}")
        
        # ponytail: Log metrics to WandB as required by the teacher
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_f1": val_f1,
            "val_kappa": val_kappa,
            "val_acc": val_acc
        })
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print("  -> Saved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


