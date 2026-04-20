import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV dosyasını oku
df = pd.read_csv('runs/thermal_yolo/results.csv')
df.columns = df.columns.str.strip()

# YOLO tarzı results.png oluştur
fig, axes = plt.subplots(2, 5, figsize=(22, 10))
fig.suptitle('YOLO Training Results (338 Epochs - Early Stopped)', fontsize=16, fontweight='bold')

plots = [
    ('epoch', 'train/box_loss', 'Train Box Loss'),
    ('epoch', 'train/cls_loss', 'Train Cls Loss'),
    ('epoch', 'train/dfl_loss', 'Train DFL Loss'),
    ('epoch', 'val/box_loss', 'Val Box Loss'),
    ('epoch', 'val/cls_loss', 'Val Cls Loss'),
    ('epoch', 'val/dfl_loss', 'Val DFL Loss'),
    ('epoch', 'metrics/precision(B)', 'Precision'),
    ('epoch', 'metrics/recall(B)', 'Recall'),
    ('epoch', 'metrics/mAP50(B)', 'mAP50'),
    ('epoch', 'metrics/mAP50-95(B)', 'mAP50-95'),
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for idx, (x_col, y_col, title) in enumerate(plots):
    ax = axes[idx // 5, idx % 5]
    x = df[x_col].values
    y = df[y_col].values

    ax.plot(x, y, color=colors[idx], linewidth=1.2, alpha=0.7)

    # Smoothed line (moving average)
    if len(y) > 10:
        window = min(20, len(y) // 5)
        if window > 1:
            smooth = pd.Series(y).rolling(window=window, center=True).mean()
            ax.plot(x, smooth, color=colors[idx], linewidth=2.5, alpha=1.0)

    # Loss grafikleri için en iyi değeri göster
    if 'loss' in y_col.lower():
        best_idx = np.argmin(y)
        best_val = y[best_idx]
        ax.axhline(y=best_val, color='gray', linestyle='--', alpha=0.3)
        ax.text(0.98, 0.95, f'Min: {best_val:.4f}\nEpoch: {int(x[best_idx])}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    else:
        # Metrik grafikleri için en iyi değeri göster
        best_idx = np.argmax(y)
        best_val = y[best_idx]
        ax.axhline(y=best_val, color='gray', linestyle='--', alpha=0.3)
        ax.text(0.98, 0.05, f'Max: {best_val:.4f}\nEpoch: {int(x[best_idx])}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('runs/thermal_yolo/results.png', dpi=200, bbox_inches='tight')
plt.close()

# Learning rate grafiği de ekleyelim
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle('Learning Rate Schedule', fontsize=14, fontweight='bold')

lr_cols = ['lr/pg0', 'lr/pg1', 'lr/pg2']
lr_names = ['LR pg0', 'LR pg1', 'LR pg2']

for i, (col, name) in enumerate(zip(lr_cols, lr_names)):
    axes2[i].plot(df['epoch'], df[col], color=colors[i], linewidth=1.5)
    axes2[i].set_title(name, fontsize=11)
    axes2[i].set_xlabel('Epoch')
    axes2[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('runs/thermal_yolo/lr_schedule.png', dpi=200, bbox_inches='tight')
plt.close()

# Son epoch istatistiklerini yazdır
print("=" * 60)
print("YOLO EĞİTİM SONUÇLARI (338 Epoch - Erken Durduruldu)")
print("=" * 60)

last = df.iloc[-1]
best_map50 = df['metrics/mAP50(B)'].max()
best_map50_epoch = df.loc[df['metrics/mAP50(B)'].idxmax(), 'epoch']
best_map5095 = df['metrics/mAP50-95(B)'].max()
best_map5095_epoch = df.loc[df['metrics/mAP50-95(B)'].idxmax(), 'epoch']

print(f"\nSon Epoch ({int(last['epoch'])}) Değerleri:")
print(f"  Train Box Loss:  {last['train/box_loss']:.4f}")
print(f"  Train Cls Loss:  {last['train/cls_loss']:.4f}")
print(f"  Train DFL Loss:  {last['train/dfl_loss']:.4f}")
print(f"  Val Box Loss:    {last['val/box_loss']:.4f}")
print(f"  Val Cls Loss:    {last['val/cls_loss']:.4f}")
print(f"  Val DFL Loss:    {last['val/dfl_loss']:.4f}")
print(f"  Precision:       {last['metrics/precision(B)']:.4f}")
print(f"  Recall:          {last['metrics/recall(B)']:.4f}")
print(f"  mAP50:           {last['metrics/mAP50(B)']:.4f}")
print(f"  mAP50-95:        {last['metrics/mAP50-95(B)']:.4f}")

print(f"\nEn İyi Değerler:")
print(f"  En İyi mAP50:    {best_map50:.4f} (Epoch {int(best_map50_epoch)})")
print(f"  En İyi mAP50-95: {best_map5095:.4f} (Epoch {int(best_map5095_epoch)})")
print(f"  En İyi Precision:{df['metrics/precision(B)'].max():.4f} (Epoch {int(df.loc[df['metrics/precision(B)'].idxmax(), 'epoch'])})")
print(f"  En İyi Recall:   {df['metrics/recall(B)'].max():.4f} (Epoch {int(df.loc[df['metrics/recall(B)'].idxmax(), 'epoch'])})")

print(f"\nGrafikler kaydedildi:")
print(f"  - runs/thermal_yolo/results.png")
print(f"  - runs/thermal_yolo/lr_schedule.png")
