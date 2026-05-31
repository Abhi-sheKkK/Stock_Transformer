# Forecast Pipeline — Diagnosis & Fixes

> Reference document for all changes made to the forecasting pipeline (May 31, 2026).

---

## Part 1: Scaler Persistence (Inference Bug)

### Problem
Every call to `create_input()` was **refitting** the `MinMaxScaler` and `QuantileTransformer` from scratch. The model was trained on one distribution but received a completely different one at inference time.

- `MinMaxScaler`: With new data, the min/max ranges change → same raw price maps to different scaled value.
- `QuantileTransformer`: Maps values to normal distribution based on quantile ranks. With more data or different price levels, the quantile mapping produces **completely different scaled values** for the same raw price.
- If current price is outside the training range, `QuantileTransformer` clips to distribution edges → predictions collapse.

### Fix
Scalers are now **persisted** at training time and **reloaded** at inference time.

**Files changed:**

| File | Change |
|---|---|
| `src/features.py` | Added `scalers_path` param to `create_input()`, plus `save_scalers()` / `load_scalers()` |
| `src/train.py` | Accepts and passes scalers, saves them to `models/` after training |
| `main.py` | Passes scalers to `train_and_evaluate()` |
| `predict.py` | Loads saved scalers from `models/` (with fallback) |
| `api/routes/predict.py` | Same — loads saved scalers |
| `app.py` | Inference uses saved scalers, training saves them |

**Scaler files saved to `models/`:**
```
models/feature_scaler.joblib
models/time_scaler.joblib
models/close_scaler.joblib
```

---

## Part 2: Training Instability (Zigzag Val Loss)

### Problem: Val loss oscillates and doesn't track train loss

7 root causes identified:

### Issue 1: `dim_feedforward=2048` (THE dominant issue)

PyTorch's `TransformerEncoderLayer` defaults `dim_feedforward=2048` regardless of `d_model`. With `d_model=64`:

```
Expansion ratio: 64 → 2048 → 64  (32x — should be 4x)
FFN parameters:  1,057,024 out of 1,169,889 total (90.4%)
Params:Samples:  137:1 (should be <10:1)
```

The FFN layers had so much capacity they memorized training patterns. Val loss can't follow because those patterns don't generalize.

**Fix:** `dim_feedforward = d_model * 4 = 256`

```
Before: 1,169,889 params
After:  245,345 params (79% reduction)
```

### Issue 2: No learning rate scheduler

Fixed LR=1e-4 overshoots optimal minima repeatedly. The model improves on training data but each overshot sends val loss up.

**Fix:** `CosineAnnealingLR` — starts at 1e-4, decays smoothly to 1e-6.

### Issue 3: No gradient clipping

Transformers are prone to gradient spikes. A single bad batch can explode gradients, causing a sudden val loss spike.

**Fix:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`.

### Issue 4: HuberLoss `delta=4.0` too large

Data is QuantileTransformer'd to N(0,1), so most values are in [-3, 3]. With `delta=4.0`, HuberLoss behaves as pure MSE — its robustness to outliers never activates.

**Fix:** `delta=1.0` (the standard default).

### Issue 5: Only supervising the last timestep

```python
# BEFORE: Only last step supervised
loss = criterion(output[:, -1], tgt[:, -1])

# AFTER: All steps supervised (5x more gradient signal)
loss = criterion(output, tgt[:, :-1])
```

The decoder outputs predictions for all 5 timesteps but only the last one was supervised. Steps 1-4 got zero gradient signal, making the autoregressive chain unreliable at inference.

### Issue 6: No early stopping

Training ran for all epochs even if val loss hadn't improved for 20+ epochs.

**Fix:** Patience-based early stopping (patience=15 epochs).

### Issue 7: Dropout inconsistency

Transformer layers defaulted to `dropout=0.1` while FC head used `0.3`.

**Fix:** Consistent `dropout=0.2` everywhere.

---

## Summary of All Changes

### `src/model.py` — Architecture

```diff
- def __init__(self, ..., dropout_rate=0.3):
+ def __init__(self, ..., dropout_rate=0.2):

+ self.input_norm = nn.LayerNorm(d_model)
+ dim_ff = d_model * 4  # was PyTorch default 2048

  self.transformer_encoder = nn.TransformerEncoder(
-     nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
+     nn.TransformerEncoderLayer(
+         d_model=d_model, nhead=nhead,
+         dim_feedforward=dim_ff, dropout=dropout_rate
+     ),
      num_layers=num_encoder_layers
  )
  # Same for decoder...

- src = self.fc_in(src).permute(1, 0, 2)
+ src = self.input_norm(self.fc_in(src)).permute(1, 0, 2)
```

### `src/train.py` — Training Loop

```diff
+ scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
+     optimizer, T_max=num_epochs, eta_min=1e-6
+ )

- criterion = nn.HuberLoss(delta=4.0)
+ criterion = nn.HuberLoss(delta=1.0)

- loss = criterion(output[:, -1], tgt.squeeze(-1)[:, -1])
+ loss = criterion(output, tgt.squeeze(-1)[:, :-1])  # all timesteps

  loss.backward()
+ torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()

+ scheduler.step()  # after each epoch

+ # Early stopping (patience=15)
+ if avg_val_loss < best_val_loss:
+     patience_counter = 0
+ else:
+     patience_counter += 1
+     if patience_counter >= 15: break
```

### `src/features.py` — Scaler Persistence

```python
# Training path (fits fresh scalers):
create_input('AAPL')

# Inference path (loads saved scalers):
create_input('AAPL', scalers_path='models')

# Save after training:
save_scalers(feature_scaler, time_scaler, close_scaler, save_dir='models')

# Load for inference:
feature_scaler, time_scaler, close_scaler = load_scalers(load_dir='models')
```

---

## How to Retrain

```bash
# From project root
python main.py --ticker AAPL --epochs 100 --batch_size 64

# This will:
# 1. Fit scalers on full AAPL history
# 2. Train with all fixes (cosine LR, grad clip, full-seq loss, early stop)
# 3. Save best_model.pth
# 4. Save scalers to models/*.joblib
# 5. Print MSE, RMSE, MAPE, Directional Accuracy
```

---

## Key Metrics to Watch

| Metric | What it tells you | Good range |
|---|---|---|
| Train/Val loss gap | Overfitting degree | Should be <2x |
| Val loss trend | Generalization | Should decrease, not zigzag |
| MAPE | Average % error | <5% is good, <3% is excellent |
| Directional Accuracy | "Did it predict up/down correctly?" | >55% is useful, >60% is strong |
| RMSE | Absolute dollar error | Context-dependent (% of price) |
