import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def train_and_evaluate(model, train_loader, test_loader, close_scaler, num_epochs=10, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=4.0)
    
    writer = SummaryWriter(log_dir="runs/stock_transformer")
    
    best_val_loss = float('inf')
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
            loss = criterion(output[:, -1], tgt.squeeze(-1)[:, -1])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        
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
                loss = criterion(output[:, -1], tgt.squeeze(-1)[:, -1])
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_epoch_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
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
    
    writer.close()
    return all_predictions, all_targets, test_indices, train_losses, val_losses
