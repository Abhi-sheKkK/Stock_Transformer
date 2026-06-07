import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import sys

def train_model(model, train_loader, val_loader, epochs=50, learning_rate=2e-4, close_scaler=None,
                feature_scaler=None, time_scaler=None):
    """
    Production-grade PyTorch training loop for StockTransformer.
    
    Features:
      - Chronological validation split evaluation
      - HuberLoss to mitigate extreme market outliers
      - AdamW with weight decay
      - Linear Warmup (3 epochs) + Cosine Annealing decay scheduler
      - Gradient clipping at 1.0 to handle explosive volatility
      - 7-epoch patience early stopping
      - Graceful exception & KeyboardInterrupt handling: saves current progress before exit
      - Hit Rate (Directional Accuracy) tracking across the 5-day horizon
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # AdamW with explicit weight decay regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    # Warmup and decay scheduler
    def lr_lambda(epoch):
        warmup_epochs = 3
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            # Cosine decay to near-zero
            progress = float(epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
            
    scheduler = LambdaLR(optimizer, lr_lambda)
    criterion = nn.HuberLoss(delta=1.0)
    
    writer = SummaryWriter(log_dir="runs/stock_transformer")
    
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0
    train_losses, val_losses = [], []
    
    epoch_idx = 0
    try:
        for epoch in range(epochs):
            epoch_idx = epoch
            model.train()
            epoch_loss = 0.0
            train_hits = 0.0
            train_total = 0
            
            for batch in train_loader:
                x_temporal = batch['X_temporal'].to(device)
                stock_id = batch['X_stock_id'].to(device)
                sector_id = batch['X_sector_id'].to(device)
                y_target = batch['Y_target'].to(device)
                
                optimizer.zero_grad()
                output = model(x_temporal, stock_id, sector_id)  # [batch, horizon]
                loss = criterion(output, y_target)
                
                loss.backward()
                # Gradient clipping to handle market volatility
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Directional Accuracy (Hit Rate) tracking
                # Checks if the predicted sign matches target sign
                correct_dirs = (torch.sign(output) == torch.sign(y_target)).float()
                train_hits += correct_dirs.sum().item()
                train_total += correct_dirs.numel()
                
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            train_hit_rate = (train_hits / train_total) * 100 if train_total > 0 else 0.0
            
            # Step the scheduler
            scheduler.step()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_hits = 0.0
            val_total = 0
            
            val_epoch_preds, val_epoch_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x_temporal = batch['X_temporal'].to(device)
                    stock_id = batch['X_stock_id'].to(device)
                    sector_id = batch['X_sector_id'].to(device)
                    y_target = batch['Y_target'].to(device)
                    
                    output = model(x_temporal, stock_id, sector_id)
                    loss = criterion(output, y_target)
                    val_loss += loss.item()
                    
                    correct_dirs = (torch.sign(output) == torch.sign(y_target)).float()
                    val_hits += correct_dirs.sum().item()
                    val_total += correct_dirs.numel()
                    
                    val_epoch_preds.append(output.cpu().numpy())
                    val_epoch_targets.append(y_target.cpu().numpy())
                    
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_hit_rate = (val_hits / val_total) * 100 if val_total > 0 else 0.0
            
            val_epoch_preds = np.concatenate(val_epoch_preds, axis=0)
            val_epoch_targets = np.concatenate(val_epoch_targets, axis=0)
            day_wise_hits = []
            for d in range(val_epoch_preds.shape[1]):
                day_acc = np.mean(np.sign(val_epoch_preds[:, d]) == np.sign(val_epoch_targets[:, d])) * 100
                day_wise_hits.append(f"Day {d+1}: {day_acc:.1f}%")
            day_wise_str = ", ".join(day_wise_hits)
            
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                  f"Train Hit Rate: {train_hit_rate:.2f}% | Val Hit Rate: {val_hit_rate:.2f}% | LR: {current_lr:.2e}")
            print(f"  Val Day-wise Hit Rate -> {day_wise_str}")
            
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('HitRate/train', train_hit_rate, epoch)
            writer.add_scalar('HitRate/val', val_hit_rate, epoch)
            writer.add_scalar('LR', current_lr, epoch)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered: no validation improvement for {patience} epochs.")
                    break
                    
    except (Exception, KeyboardInterrupt) as e:
        print(f"\n[CRITICAL] Training interrupted or crashed: {e}")
        print("Saving current state to 'best_model.pth' and 'interrupted_model.pth' to prevent loss of progress...")
        torch.save(model.state_dict(), 'best_model.pth')
        torch.save({
            'epoch': epoch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'interrupted_model.pth')
        # If interrupted by user, let's proceed to evaluate with what we have instead of hard failing
        if isinstance(e, KeyboardInterrupt):
            print("Proceeding to final evaluation using the best weights saved so far...")
        else:
            raise e

    # Load best weights for validation reporting
    try:
        model.load_state_dict(torch.load('best_model.pth'))
    except Exception:
        print("Warning: Could not load 'best_model.pth'. Evaluating using current weights.")
        
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            x_temporal = batch['X_temporal'].to(device)
            stock_id = batch['X_stock_id'].to(device)
            sector_id = batch['X_sector_id'].to(device)
            y_target = batch['Y_target'].to(device)
            
            output = model(x_temporal, stock_id, sector_id)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y_target.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    day1_preds = all_preds[:, 0]
    day1_targets = all_targets[:, 0]
    
    mse = np.mean((day1_preds - day1_targets) ** 2)
    rmse = np.sqrt(mse)
    
    dir_acc = np.mean(np.sign(day1_preds) == np.sign(day1_targets)) * 100
    all_dir_acc = np.mean(np.sign(all_preds) == np.sign(all_targets)) * 100
    
    print(f"\nFinal Validation Summary:")
    print(f"  Day-1 Return MSE: {mse:.6f} | RMSE: {rmse:.6f}")
    print(f"  All-Horizon Directional Accuracy (Hit Rate): {all_dir_acc:.2f}%")
    print(f"  Day-wise Directional Accuracy (Hit Rate):")
    for d in range(all_preds.shape[1]):
        day_pred = all_preds[:, d]
        day_targ = all_targets[:, d]
        day_acc = np.mean(np.sign(day_pred) == np.sign(day_targ)) * 100
        print(f"    Day {d+1}: {day_acc:.2f}%")
    
    if feature_scaler is not None and time_scaler is not None and close_scaler is not None:
        from src.features import save_scalers
        save_scalers(feature_scaler, time_scaler, close_scaler, save_dir='models')
        
    writer.close()
    
    test_indices = list(range(len(day1_preds)))
    return day1_preds, day1_targets, test_indices, train_losses, val_losses

def train_and_evaluate(model, train_loader, val_loader, close_scaler, num_epochs=10, learning_rate=2e-4,
                       feature_scaler=None, time_scaler=None):
    """
    Backward-compatible wrapper mapping to the production-grade train_model function.
    """
    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=num_epochs,
        learning_rate=learning_rate,
        close_scaler=close_scaler,
        feature_scaler=feature_scaler,
        time_scaler=time_scaler
    )
