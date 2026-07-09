import os
import json
import numpy as np
import torch
import torch.nn as nn
import wandb
from common.evaluate import evaluate_model

def compute_class_weights_from_files(train_files, num_classes=5):
    """
    Computes class weights from train_files only.
    Safely handles missing classes with epsilon.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for f in train_files:
        data = np.load(f)
        y = data['y']
        for c in range(num_classes):
            counts[c] += np.sum(y == c)
            
    total = np.sum(counts)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c] + 1e-6)
        else:
            weights[c] = 1.0
            
    return torch.from_numpy(weights).float()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        logits = model(x, mask=mask)  # (B, L, C)
        
        B, L, C = logits.shape
        logits_flat = logits.reshape(B * L, C)
        y_flat = y.reshape(B * L)
        mask_flat = mask.reshape(B * L).bool()
        
        # Only compute loss on unmasked positions
        if mask_flat.sum() > 0:
            loss = criterion(logits_flat[mask_flat], y_flat[mask_flat])
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            total_loss += loss.item()
            
    return total_loss / len(loader) if len(loader) > 0 else 0.0

def run_training(model, train_loader, val_loader, test_loader, train_files, args, device):
    # For local/Colab:
    out_dir = os.path.join("results", args.architecture, f"fold_{args.fold_id}")
    # For Kaggle (uncomment if running on Kaggle):
    # out_dir = f"/kaggle/working/results/{args.architecture}/fold_{args.fold_id}"
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, "best_model.pth")
    
    # Class weights calculation
    class_weights = compute_class_weights_from_files(train_files, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )
    
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device=device)
        
        val_f1 = val_metrics["macro_f1"]
        val_kappa = val_metrics["kappa"]
        val_acc = val_metrics["accuracy"]
        
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - Val F1: {val_f1:.4f} - Val Kappa: {val_kappa:.4f}")
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_f1": val_f1,
                "val_kappa": val_kappa,
                "val_acc": val_acc
            })
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_kappa = val_kappa
            best_train_loss = train_loss
            patience_counter = 0
            
            torch.save(model.state_dict(), best_model_path)
            
            if args.use_wandb:
                wandb.summary["best_epoch"] = int(best_epoch)
                wandb.summary["best_val_f1"] = float(best_val_f1)
                wandb.summary["best_val_acc"] = float(best_val_acc)
                wandb.summary["best_val_kappa"] = float(best_val_kappa)
                wandb.summary["best_train_loss"] = float(best_train_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
    # Load best checkpoint and evaluate on Test loader
    print("\nTraining complete. Evaluating best model on test set...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
    test_metrics = evaluate_model(model, test_loader, device=device)
    test_acc = test_metrics["accuracy"]
    test_f1 = test_metrics["macro_f1"]
    test_kappa = test_metrics["kappa"]
    
    print(f"Test Results: ACC={test_acc:.4f} | Macro-F1={test_f1:.4f} | Kappa={test_kappa:.4f}")
    
    if args.use_wandb:
        # Log final test evaluation metrics
        wandb.log({
            "epoch": best_epoch,
            "final_fold_acc": test_acc,
            "final_fold_f1": test_f1,
            "final_fold_kappa": test_kappa,
        })
        wandb.summary["final_fold_acc"] = test_acc
        wandb.summary["final_fold_f1"] = test_f1
        wandb.summary["final_fold_kappa"] = test_kappa
        
        # Save model checkpoint to WandB Artifacts
        artifact = wandb.Artifact(
            name=f"{args.architecture}_fold{args.fold_id}_best_model",
            type="model"
        )
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)
        
    # Save outputs to results directory
    # 1. metrics.json
    metrics_json_path = os.path.join(out_dir, "metrics.json")
    metrics_to_save = {
        "fold_id": args.fold_id,
        "architecture": args.architecture,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "test_kappa": test_kappa,
        "per_class_f1": test_metrics["per_class_f1"],
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1
    }
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
        
    # 2. confusion_matrix.npy
    cm_path = os.path.join(out_dir, "confusion_matrix.npy")
    np.save(cm_path, np.array(test_metrics["confusion_matrix"]))
    
    # 3. predictions.npz
    preds_path = os.path.join(out_dir, "predictions.npz")
    np.savez(preds_path, y_true=test_metrics["y_true"], y_pred=test_metrics["y_pred"])
    
    # 4. config.json
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
        
    print(f"Results saved to {out_dir}")
