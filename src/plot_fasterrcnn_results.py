from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_XLSX = Path("runs/faster_cnn/fasterrcnn_mobilenet_training.xlsx")
DEFAULT_OUTDIR = Path("runs/faster_cnn")
METRICS_SHEET = "Training Metrics"
CM_SHEET = "Confusion Matrix"

COLUMNS = [
    "epoch", "lr",
    "box_train", "box_val",
    "cls_train", "cls_val",
    "dfl_train", "dfl_val",
    "precision_val", "precision_train",
    "recall_val", "recall_train",
    "f1_val", "f1_train",
    "map50", "map50_95",
    "accuracy_val",
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def load_metrics(xlsx_path: Path = DEFAULT_XLSX) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=METRICS_SHEET, header=None, skiprows=2)
    raw = raw.iloc[:, : len(COLUMNS)]
    raw.columns = COLUMNS
    df = raw.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["epoch"]).reset_index(drop=True)

    if df["f1_val"].isna().all():
        p = df["precision_val"]
        r = df["recall_val"]
        df["f1_val"] = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    return df


def load_confusion_matrix(xlsx_path: Path = DEFAULT_XLSX):
    raw = pd.read_excel(xlsx_path, sheet_name=CM_SHEET, header=None)
    labels = raw.iloc[6, 3:6].tolist()
    matrix = raw.iloc[7:10, 3:6].to_numpy(dtype=float)
    return np.nan_to_num(matrix), labels


def _annotate_best(ax, x, y, minimize: bool):
    y_arr = np.asarray(y, dtype=float)
    if np.all(np.isnan(y_arr)):
        return
    idx = int(np.nanargmin(y_arr) if minimize else np.nanargmax(y_arr))
    val = y_arr[idx]
    ax.axhline(y=val, color="gray", linestyle="--", alpha=0.3)
    label = f"{'Min' if minimize else 'Max'}: {val:.4f}\nEpoch: {int(x[idx])}"
    va, y_pos = ("top", 0.95) if minimize else ("bottom", 0.05)
    ax.text(0.98, y_pos, label, transform=ax.transAxes, ha="right", va=va, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))


def _plot_pair(ax, x, df, train_col, val_col, title, color_idx, minimize=True):
    ax.plot(x, df[train_col], color=COLORS[color_idx], linewidth=1.6, alpha=0.85, label="Train")
    ax.plot(x, df[val_col], color=COLORS[color_idx + 1], linewidth=1.6, alpha=0.85, label="Val")
    _annotate_best(ax, x, df[val_col], minimize=minimize)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.tick_params(labelsize=8)


def _plot_single(ax, x, y, title, color_idx, minimize=False, ylabel=None):
    ax.plot(x, y, color=COLORS[color_idx], linewidth=1.8)
    _annotate_best(ax, x, y, minimize=minimize)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


def plot_training_results(df: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("Faster R-CNN MobileNet Training Results", fontsize=16, fontweight="bold")
    x = df["epoch"].values

    _plot_pair(axes[0, 0], x, df, "box_train", "box_val", "Box Loss", 0, minimize=True)
    _plot_pair(axes[0, 1], x, df, "cls_train", "cls_val", "Class Loss", 2, minimize=True)
    _plot_pair(axes[0, 2], x, df, "dfl_train", "dfl_val", "DFL Loss", 4, minimize=True)
    _plot_single(axes[0, 3], x, df["precision_val"], "Precision (Val)", 6, minimize=False)

    _plot_single(axes[1, 0], x, df["recall_val"], "Recall (Val)", 7, minimize=False)
    _plot_single(axes[1, 1], x, df["f1_val"], "F1 Score (Val)", 8, minimize=False)

    ax_map = axes[1, 2]
    ax_map.plot(x, df["map50"], color=COLORS[0], linewidth=1.8, label="mAP@0.5")
    ax_map.plot(x, df["map50_95"], color=COLORS[3], linewidth=1.8, label="mAP@0.5:0.95")
    _annotate_best(ax_map, x, df["map50"], minimize=False)
    ax_map.set_title("mAP", fontsize=11, fontweight="bold")
    ax_map.set_xlabel("Epoch", fontsize=9)
    ax_map.grid(True, alpha=0.3)
    ax_map.legend(fontsize=8, loc="best")
    ax_map.tick_params(labelsize=8)

    _plot_single(axes[1, 3], x, df["accuracy_val"], "Accuracy (Val)", 9, minimize=False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_learning_rate(df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Learning Rate Schedule", fontsize=13, fontweight="bold")
    ax.plot(df["epoch"], df["lr"], color=COLORS[5], linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(matrix: np.ndarray, labels, out_path: Path, normalize: bool = True) -> Path:
    m = matrix.astype(float)
    if normalize:
        row_sums = m.sum(axis=1, keepdims=True)
        m = np.divide(m, row_sums, out=np.zeros_like(m), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() if m.max() > 0 else 1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""),
                 fontsize=12, fontweight="bold")

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = m[i, j]
            ax.text(j, i, f"{val:.2f}" if normalize else f"{int(val)}",
                    ha="center", va="center",
                    color="white" if val > m.max() / 2 else "black", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_summary(df: pd.DataFrame) -> None:
    last = df.iloc[-1]
    print("=" * 60)
    print("FASTER R-CNN MOBILENET TRAINING RESULTS")
    print("=" * 60)
    print(f"\nLast Epoch ({int(last['epoch'])}):")
    print(f"  Box Loss    Train/Val: {last['box_train']:.4f} / {last['box_val']:.4f}")
    print(f"  Class Loss  Train/Val: {last['cls_train']:.4f} / {last['cls_val']:.4f}")
    print(f"  DFL Loss    Train/Val: {last['dfl_train']:.4f} / {last['dfl_val']:.4f}")
    print(f"  Precision (Val): {last['precision_val']:.4f}")
    print(f"  Recall    (Val): {last['recall_val']:.4f}")
    print(f"  F1 Score  (Val): {last['f1_val']:.4f}")
    print(f"  mAP@0.5        : {last['map50']:.4f}")
    print(f"  mAP@0.5:0.95   : {last['map50_95']:.4f}")
    print(f"  Accuracy  (Val): {last['accuracy_val']:.4f}")

    print("\nBest Values:")
    for col, name, fn in [
        ("map50", "mAP@0.5", "max"),
        ("map50_95", "mAP@0.5:0.95", "max"),
        ("precision_val", "Precision", "max"),
        ("recall_val", "Recall", "max"),
        ("f1_val", "F1", "max"),
        ("box_val", "Val Box Loss", "min"),
    ]:
        idx = df[col].idxmax() if fn == "max" else df[col].idxmin()
        print(f"  {name:15s}: {df.loc[idx, col]:.4f} (Epoch {int(df.loc[idx, 'epoch'])})")


def generate_all(xlsx_path: Path = DEFAULT_XLSX, out_dir: Path = DEFAULT_OUTDIR) -> dict:
    xlsx_path = Path(xlsx_path)
    out_dir = Path(out_dir)
    df = load_metrics(xlsx_path)
    matrix, labels = load_confusion_matrix(xlsx_path)

    outputs = {
        "results": plot_training_results(df, out_dir / "results.png"),
        "lr": plot_learning_rate(df, out_dir / "lr_schedule.png"),
        "cm": plot_confusion_matrix(matrix, labels, out_dir / "confusion_matrix.png"),
    }
    print_summary(df)
    print("\nSaved:")
    for key, path in outputs.items():
        print(f"  - {path}")
    return outputs


if __name__ == "__main__":
    generate_all()
