import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def train_and_evaluate(model, train_loader, test_loader, close_scaler, num_epochs=10, learning_rate=1e-4,
                       feature_scaler=None, time_scaler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Cosine annealing: smoothly decays LR to near-zero, prevents late-stage oscillation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # HuberLoss delta=1.0 (standard) — delta=4.0 was too high, made it behave like MSE
    criterion = nn.HuberLoss(delta=1.0)
    
    writer = SummaryWriter(log_dir="runs/stock_transformer")
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    train_losses, val_losses = [], []
    all_predictions, all_targets, test_indices = [], [], []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            src, tgt = batch
            src, tgt = src.to(device).to(dtype=torch.float32), tgt.to(device).to(dtype=torch.float32)
            tgt = tgt.unsqueeze(-1)
            tgt_input = torch.zeros((tgt.shape[0], tgt.shape[1], src.shape[-1]), device=device)
            tgt_input[:, :, 0] = tgt.squeeze(-1)
            
            optimizer.zero_grad()
            output = model(src, tgt_input[:, :-1])
            
            # Supervise ALL timesteps, not just the last one
            # This gives 5x more gradient signal per batch and teaches the model
            # to produce good intermediate predictions (critical for autoregressive inference)
            loss = criterion(output, tgt.squeeze(-1)[:, :-1])
            
            loss.backward()
            
            # Gradient clipping — prevents exploding gradients (common in Transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        
        # Step the scheduler
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                src, tgt = batch
                src, tgt = src.to(device).to(dtype=torch.float32), tgt.to(device).to(dtype=torch.float32)
                tgt = tgt.unsqueeze(-1)
                tgt_input = torch.zeros((tgt.shape[0], tgt.shape[1], src.shape[-1]), device=device)
                tgt_input[:, :, 0] = tgt.squeeze(-1)
                
                output = model(src, tgt_input[:, :-1])
                loss = criterion(output, tgt.squeeze(-1)[:, :-1])
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch + 1}/{num_epochs}, Train: {avg_epoch_loss:.6f}, Val: {avg_val_loss:.6f}, LR: {current_lr:.2e}')
        
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)')
                break
            
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            src, tgt = batch
            src, tgt = src.to(device).to(dtype=torch.float32), tgt.to(device).to(dtype=torch.float32)
            tgt = tgt.unsqueeze(-1)
            tgt_input = torch.zeros((tgt.shape[0], tgt.shape[1], src.shape[-1]), device=device)
            tgt_input[:, :, 0] = tgt.squeeze(-1)
            
            output = model(src, tgt_input[:, :-1])
            preds = output[:, -1].cpu().numpy()
            targets = tgt[:, -1].cpu().numpy()
            
            all_predictions.extend(preds)
            all_targets.extend(targets)
            test_indices.extend(list(range(i * test_loader.batch_size, min((i + 1) * test_loader.batch_size, len(test_loader.dataset)))))
            
    all_predictions = close_scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1)).flatten()
    all_targets = close_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()
    
    mse = np.mean((all_predictions - all_targets)**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((all_targets - all_predictions) / all_targets)) * 100
    
    dir_acc = 0.0
    if len(all_targets) > 1:
        actual_diff = np.diff(all_targets)
        pred_diff = all_predictions[1:] - all_targets[:-1]
        correct_directions = np.sign(actual_diff) == np.sign(pred_diff)
        dir_acc = np.mean(correct_directions) * 100
    
    print(f'Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, Directional Accuracy: {dir_acc:.2f}%')
    
    # Save scalers alongside model so inference uses the same distribution
    if feature_scaler is not None and time_scaler is not None:
        from src.features import save_scalers
        save_scalers(feature_scaler, time_scaler, close_scaler, save_dir='models')
    
    writer.close()
    return all_predictions, all_targets, test_indices, train_losses, val_losses


