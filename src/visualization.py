import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_results(train_losses, val_losses, actual, predicted, test_indices, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Styling
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Plot Training vs Validation Losses
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, color="#1f77b4", label='Training Loss', linewidth=2)
    plt.plot(val_losses, color="#ff7f0e", label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Huber Loss', fontsize=12)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=300)
    plt.close()
    
    # 2. Plot Predictions vs Actual
    # Convert dummy indices to a range for plotting
    x_axis = range(len(actual))
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, actual, color="#2ca02c", label='Actual Price', linewidth=1.5, alpha=0.9)
    # Predicted is usually smoother; we use a dashed line with high opacity
    plt.plot(x_axis, predicted, color="#d62728", label='Predicted Price', linewidth=2, linestyle='--')
    
    plt.title('Predicted vs Actual Closing Prices', fontsize=16, fontweight='bold')
    plt.xlabel('Time (Test Set Steps)', fontsize=12)
    plt.ylabel('Stock Price', fontsize=12)
    plt.legend(frameon=True, loc="upper left", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/predictions_vs_actual.png", dpi=300)
    plt.close()
    
    print(f"✅ Enhanced plots saved to the '{save_dir}' directory.")
