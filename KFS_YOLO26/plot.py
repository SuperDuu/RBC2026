import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_PATH = 'models/training_log_3.csv'
PLOT_ACC_PATH = 'models/figure_v2.3_accuracy.png'
PLOT_LOSS_PATH = 'models/figure_v2.3_loss.png'

os.makedirs('models', exist_ok=True)

try:
    df = pd.read_csv(LOG_PATH)
    print(f"[+] Đã nạp thành công {len(df)} epoch dữ liệu.")
except Exception as e:
    print(f"[-] Lỗi đọc file CSV: {e}")
    exit()

plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
plt.plot(df['epoch'], df['val_accuracy'], label='Val Accuracy', color='#d62728', linewidth=2)

plt.title('CNN v2.3 Training History - Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(PLOT_ACC_PATH, dpi=300)
print(f"[+] Đã lưu biểu đồ Accuracy tại: {PLOT_ACC_PATH}")

plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#d62728', linewidth=2)

plt.title('CNN v2.3 Training History - Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(PLOT_LOSS_PATH, dpi=300)
print(f"[+] Đã lưu biểu đồ Loss tại: {PLOT_LOSS_PATH}")