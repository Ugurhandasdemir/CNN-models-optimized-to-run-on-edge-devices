"""
=============================================================================
  TERMAL VERİ SETİ - DETAYLI KEŞİFSEL VERİ ANALİZİ (EDA)
=============================================================================
  Veri Seti: Birleştirilmiş Termal COCO Dataset
  Sınıflar:  0-Person, 1-Car, 2-OtherVehicle

  Bu script, nesne tespit veri setimizin temel istatistiklerini ve
  dağılımlarını görselleştirerek veri setini daha iyi anlamamızı sağlar.
  Model eğitimi öncesinde veri setindeki olası dengesizlikleri,
  anomalileri ve genel yapıyı keşfetmek amacıyla hazırlanmıştır.
=============================================================================
"""

import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

# ----- Ayarlar -----
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

DATASET_DIR = Path(__file__).parent
SPLITS = ["train", "val", "test"]
CLASS_NAMES = {0: "Person", 1: "Car", 2: "OtherVehicle"}
CLASS_COLORS = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}
OUTPUT_DIR = DATASET_DIR / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =====================================================================
#  1. ADIM: VERİYİ YÜKLEME
# =====================================================================
print("=" * 60)
print("  VERİ SETİ YÜKLENIYOR...")
print("=" * 60)

all_data = {}
for split in SPLITS:
    json_path = DATASET_DIR / split / "_annotations.coco.json"
    with open(json_path, "r") as f:
        all_data[split] = json.load(f)
    print(f"  {split:>5}: {len(all_data[split]['images']):>6} görüntü, "
          f"{len(all_data[split]['annotations']):>7} annotation yüklendi")

# Tüm split'leri birleştir (genel analiz için)
all_images = []
all_annotations = []
for split in SPLITS:
    for img in all_data[split]["images"]:
        img_copy = img.copy()
        img_copy["_split"] = split
        all_images.append(img_copy)
    for ann in all_data[split]["annotations"]:
        ann_copy = ann.copy()
        ann_copy["_split"] = split
        all_annotations.append(ann_copy)

print(f"\n  TOPLAM: {len(all_images)} görüntü, {len(all_annotations)} annotation")
print("=" * 60)


# =====================================================================
#  2. GRAFIK 1: SPLIT BAZINDA GÖRÜNTÜ VE ANNOTATION SAYILARI
# =====================================================================
#
#  AÇIKLAMA: Bu bar chart, veri setimizin train/val/test split'lerine
#  nasıl dağıldığını gösterir. İdeal bir veri setinde train seti en
#  büyük olmalıdır (genellikle %70-80). Val ve test setleri daha küçük
#  olur. Eğer val veya test çok küçükse, model değerlendirmesi
#  güvenilir olmayabilir.
# =====================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Grafik 1: Train / Val / Test Dağılımı", fontsize=15, fontweight="bold")

split_img_counts = [len(all_data[s]["images"]) for s in SPLITS]
split_ann_counts = [len(all_data[s]["annotations"]) for s in SPLITS]
colors = ["#2196F3", "#FF9800", "#4CAF50"]

# Görüntü sayıları
bars1 = ax1.bar(SPLITS, split_img_counts, color=colors, edgecolor="black", linewidth=0.5)
ax1.set_title("Görüntü Sayıları")
ax1.set_ylabel("Görüntü Sayısı")
for bar, count in zip(bars1, split_img_counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
             f"{count:,}", ha="center", va="bottom", fontweight="bold")

# Annotation sayıları
bars2 = ax2.bar(SPLITS, split_ann_counts, color=colors, edgecolor="black", linewidth=0.5)
ax2.set_title("Annotation (Etiket) Sayıları")
ax2.set_ylabel("Annotation Sayısı")
for bar, count in zip(bars2, split_ann_counts):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2000,
             f"{count:,}", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_split_dagilimi.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 1: Split dağılımı kaydedildi")


# =====================================================================
#  3. GRAFIK 2: SINIF BAZINDA ANNOTATION DAĞILIMI (PIE + BAR)
# =====================================================================
#
#  AÇIKLAMA: Nesne tespit problemlerinde sınıf dengesizliği (class
#  imbalance) en yaygın sorunlardan biridir. Eğer bir sınıf diğerlerine
#  göre çok az temsil ediliyorsa, model o sınıfı öğrenmekte zorlanır.
#  Bu grafik, her sınıfın veri setinde ne kadar temsil edildiğini
#  gösterir. Dengesizlik varsa, veri artırma (augmentation), sınıf
#  ağırlıklandırma veya oversampling gibi teknikler düşünülmelidir.
# =====================================================================

class_counts = Counter(ann["category_id"] for ann in all_annotations)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Grafik 2: Sınıf Dağılımı (Class Distribution)", fontsize=15, fontweight="bold")

labels = [CLASS_NAMES[i] for i in sorted(class_counts.keys())]
values = [class_counts[i] for i in sorted(class_counts.keys())]
cls_colors = [CLASS_COLORS[i] for i in sorted(class_counts.keys())]

# Pie chart
wedges, texts, autotexts = ax1.pie(
    values, labels=labels, autopct="%1.1f%%", colors=cls_colors,
    startangle=90, textprops={"fontsize": 11},
    wedgeprops={"edgecolor": "black", "linewidth": 0.5},
)
for at in autotexts:
    at.set_fontweight("bold")
ax1.set_title("Oransal Dağılım")

# Bar chart
bars = ax2.bar(labels, values, color=cls_colors, edgecolor="black", linewidth=0.5)
ax2.set_title("Sayısal Dağılım")
ax2.set_ylabel("Annotation Sayısı")
for bar, v in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3000,
             f"{v:,}", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_sinif_dagilimi.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 2: Sınıf dağılımı kaydedildi")


# =====================================================================
#  4. GRAFIK 3: SPLIT BAZINDA SINIF DAĞILIMI (STACKED BAR)
# =====================================================================
#
#  AÇIKLAMA: Her split içinde sınıf dağılımının tutarlı olması önemlidir.
#  Eğer train setinde çok Car varken test setinde hiç yoksa, modelin
#  test performansı yanıltıcı olur. Bu grafik, her split'teki sınıf
#  oranlarının birbirine benzeyip benzemediğini kontrol etmemizi sağlar.
# =====================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Grafik 3: Split Bazında Sınıf Dağılımı", fontsize=15, fontweight="bold")

split_class_counts = {}
for split in SPLITS:
    counter = Counter(a["category_id"] for a in all_data[split]["annotations"])
    split_class_counts[split] = counter

# Stacked bar (absolut)
x = np.arange(len(SPLITS))
width = 0.5
bottoms = np.zeros(len(SPLITS))
for cls_id in sorted(CLASS_NAMES.keys()):
    vals = [split_class_counts[s].get(cls_id, 0) for s in SPLITS]
    ax1.bar(x, vals, width, bottom=bottoms, label=CLASS_NAMES[cls_id],
            color=CLASS_COLORS[cls_id], edgecolor="black", linewidth=0.3)
    bottoms += np.array(vals)
ax1.set_xticks(x)
ax1.set_xticklabels(SPLITS)
ax1.set_ylabel("Annotation Sayısı")
ax1.set_title("Mutlak Değerler")
ax1.legend()

# Stacked bar (yüzde)
bottoms = np.zeros(len(SPLITS))
for cls_id in sorted(CLASS_NAMES.keys()):
    vals = [split_class_counts[s].get(cls_id, 0) for s in SPLITS]
    totals = [sum(split_class_counts[s].values()) for s in SPLITS]
    pcts = [v / t * 100 if t > 0 else 0 for v, t in zip(vals, totals)]
    ax2.bar(x, pcts, width, bottom=bottoms, label=CLASS_NAMES[cls_id],
            color=CLASS_COLORS[cls_id], edgecolor="black", linewidth=0.3)
    # Yüzde etiketleri
    for i, (p, b) in enumerate(zip(pcts, bottoms)):
        if p > 3:
            ax2.text(x[i], b + p / 2, f"{p:.1f}%", ha="center", va="center",
                     fontsize=9, fontweight="bold")
    bottoms += np.array(pcts)
ax2.set_xticks(x)
ax2.set_xticklabels(SPLITS)
ax2.set_ylabel("Oran (%)")
ax2.set_title("Yüzdesel Dağılım")
ax2.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_split_sinif_dagilimi.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 3: Split bazında sınıf dağılımı kaydedildi")


# =====================================================================
#  5. GRAFIK 4: GÖRÜNTÜ BOYUTLARI DAĞILIMI
# =====================================================================
#
#  AÇIKLAMA: Farklı veri setlerini birleştirdiğimiz için görüntü
#  boyutları farklılık gösterebilir. Model eğitiminde tüm görüntüler
#  aynı boyuta resize edilir, bu nedenle orijinal boyut dağılımını
#  bilmek önemlidir. Çok farklı boyutlardaki görüntüler resize
#  sonrası bozulabilir veya bilgi kaybına yol açabilir.
# =====================================================================

widths = [img["width"] for img in all_images]
heights = [img["height"] for img in all_images]
aspects = [w / h for w, h in zip(widths, heights)]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Grafik 4: Görüntü Boyutları Dağılımı", fontsize=15, fontweight="bold")

# Genişlik dağılımı
axes[0].hist(widths, bins=50, color="#9b59b6", edgecolor="black", linewidth=0.3, alpha=0.8)
axes[0].set_title("Genişlik (Width) Dağılımı")
axes[0].set_xlabel("Piksel")
axes[0].set_ylabel("Görüntü Sayısı")
axes[0].axvline(np.mean(widths), color="red", linestyle="--", label=f"Ortalama: {np.mean(widths):.0f}")
axes[0].legend()

# Yükseklik dağılımı
axes[1].hist(heights, bins=50, color="#e67e22", edgecolor="black", linewidth=0.3, alpha=0.8)
axes[1].set_title("Yükseklik (Height) Dağılımı")
axes[1].set_xlabel("Piksel")
axes[1].axvline(np.mean(heights), color="red", linestyle="--", label=f"Ortalama: {np.mean(heights):.0f}")
axes[1].legend()

# En-boy oranı (aspect ratio)
axes[2].hist(aspects, bins=50, color="#1abc9c", edgecolor="black", linewidth=0.3, alpha=0.8)
axes[2].set_title("En-Boy Oranı (Aspect Ratio)")
axes[2].set_xlabel("Width / Height")
axes[2].axvline(np.mean(aspects), color="red", linestyle="--", label=f"Ortalama: {np.mean(aspects):.2f}")
axes[2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_goruntu_boyutlari.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 4: Görüntü boyutları dağılımı kaydedildi")


# =====================================================================
#  6. GRAFIK 5: BOUNDING BOX BOYUTLARI (SINIF BAZINDA)
# =====================================================================
#
#  AÇIKLAMA: Bounding box boyutları, tespit edilecek nesnelerin
#  görüntüdeki büyüklüğünü gösterir. Küçük nesneler (small objects)
#  tespit etmesi en zor olanlardır. COCO standardına göre:
#    - Small:  area < 32x32 = 1024 px²
#    - Medium: 1024 < area < 96x96 = 9216 px²
#    - Large:  area > 9216 px²
#  Eğer veri setimizde çok fazla küçük nesne varsa, modelin anchor
#  boyutları ve feature pyramid yapısı buna göre ayarlanmalıdır.
# =====================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Grafik 5: Bounding Box Boyutları (Sınıf Bazında)", fontsize=15, fontweight="bold")

for cls_id in sorted(CLASS_NAMES.keys()):
    cls_anns = [a for a in all_annotations if a["category_id"] == cls_id]
    bb_widths = [a["bbox"][2] for a in cls_anns]
    bb_heights = [a["bbox"][3] for a in cls_anns]
    bb_areas = [a["area"] for a in cls_anns]

    ax = axes[cls_id]
    ax.hist2d(bb_widths, bb_heights, bins=80, cmap="YlOrRd", cmin=1)
    ax.set_title(f"{CLASS_NAMES[cls_id]} (n={len(cls_anns):,})")
    ax.set_xlabel("BBox Genişlik (px)")
    ax.set_ylabel("BBox Yükseklik (px)")
    ax.set_xlim(0, min(np.percentile(bb_widths, 99), 500))
    ax.set_ylim(0, min(np.percentile(bb_heights, 99), 500))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_bbox_boyutlari.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 5: BBox boyutları kaydedildi")


# =====================================================================
#  7. GRAFIK 6: NESNE BOYUT KATEGORİLERİ (SMALL / MEDIUM / LARGE)
# =====================================================================
#
#  AÇIKLAMA: COCO metriklerinde small/medium/large ayrımı yapılır.
#  Bu grafik, her sınıfta kaç tane küçük, orta ve büyük nesne
#  olduğunu gösterir. Eğer çoğu nesne "small" ise, model küçük
#  nesneleri tespit edebilecek şekilde yapılandırılmalıdır (örn.
#  daha yüksek çözünürlükte input, multi-scale feature maps).
# =====================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Grafik 6: COCO Nesne Boyut Kategorileri", fontsize=15, fontweight="bold")

size_labels = ["Small\n(<32²)", "Medium\n(32²-96²)", "Large\n(>96²)"]
size_colors = ["#e74c3c", "#f39c12", "#27ae60"]

for cls_id in sorted(CLASS_NAMES.keys()):
    cls_anns = [a for a in all_annotations if a["category_id"] == cls_id]
    areas = [a["area"] for a in cls_anns]
    small = sum(1 for a in areas if a < 1024)
    medium = sum(1 for a in areas if 1024 <= a < 9216)
    large = sum(1 for a in areas if a >= 9216)

    ax = axes[cls_id]
    bars = ax.bar(size_labels, [small, medium, large], color=size_colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_title(f"{CLASS_NAMES[cls_id]}")
    ax.set_ylabel("Annotation Sayısı")
    for bar, v in zip(bars, [small, medium, large]):
        pct = v / len(areas) * 100 if areas else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_boyut_kategorileri.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 6: Boyut kategorileri kaydedildi")


# =====================================================================
#  8. GRAFIK 7: GÖRÜNTÜ BAŞINA NESNE SAYISI DAĞILIMI
# =====================================================================
#
#  AÇIKLAMA: Bir görüntüde kaç nesne olduğu, modelin ne kadar
#  "kalabalık" sahnelerle karşılaşacağını gösterir. Eğer çoğu
#  görüntüde 1-2 nesne varsa basit bir sahnedir. Eğer 50+ nesne
#  varsa, NMS (Non-Maximum Suppression) parametreleri ve modelin
#  capacity'si buna göre ayarlanmalıdır.
# =====================================================================

img_obj_counts = Counter()
for ann in all_annotations:
    img_obj_counts[ann["image_id"]] += 1

counts_list = list(img_obj_counts.values())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Grafik 7: Görüntü Başına Nesne Sayısı", fontsize=15, fontweight="bold")

# Histogram
ax1.hist(counts_list, bins=range(0, min(max(counts_list) + 2, 102), 1),
         color="#8e44ad", edgecolor="black", linewidth=0.2, alpha=0.8)
ax1.set_title("Dağılım Histogramı")
ax1.set_xlabel("Nesne Sayısı / Görüntü")
ax1.set_ylabel("Görüntü Sayısı")
ax1.set_xlim(0, min(np.percentile(counts_list, 99) + 5, 100))
ax1.axvline(np.mean(counts_list), color="red", linestyle="--",
            label=f"Ortalama: {np.mean(counts_list):.1f}")
ax1.axvline(np.median(counts_list), color="blue", linestyle="--",
            label=f"Medyan: {np.median(counts_list):.0f}")
ax1.legend()

# Box plot - split bazında
split_counts = {}
for split in SPLITS:
    img_ids_in_split = {img["id"] for img in all_data[split]["images"]}
    split_obj_counts = Counter()
    for ann in all_data[split]["annotations"]:
        split_obj_counts[ann["image_id"]] += 1
    split_counts[split] = list(split_obj_counts.values())

bp = ax2.boxplot([split_counts[s] for s in SPLITS], labels=SPLITS,
                 patch_artist=True, showfliers=False)
for patch, color in zip(bp["boxes"], ["#2196F3", "#FF9800", "#4CAF50"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_title("Split Bazında (Box Plot)")
ax2.set_ylabel("Nesne Sayısı / Görüntü")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_goruntu_basina_nesne.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 7: Görüntü başına nesne sayısı kaydedildi")


# =====================================================================
#  9. GRAFIK 8: BOUNDING BOX MERKEZ NOKTALARININ ISITMA HARİTASI
# =====================================================================
#
#  AÇIKLAMA: Nesnelerin görüntü üzerinde nerede konumlandığını
#  gösterir. Eğer nesneler hep görüntünün ortasındaysa, modelin
#  kenarları öğrenmesi zor olabilir. Termal görüntülerde drone
#  perspektifinden bakıldığında nesneler genellikle daha homojen
#  dağılır, ama bazı bias'lar olabilir (örn. yol üzerinde araçlar
#  belli bölgelerde yoğunlaşır).
# =====================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Grafik 8: BBox Merkez Noktaları Isı Haritası (Normalize)", fontsize=15, fontweight="bold")

# Görüntü boyutlarını image_id ile eşleştir
img_dims = {}
for img in all_images:
    img_dims[img["id"]] = (img["width"], img["height"])

for cls_id in sorted(CLASS_NAMES.keys()):
    cls_anns = [a for a in all_annotations if a["category_id"] == cls_id]

    # Merkez noktalarını normalize et (0-1 aralığına)
    cx_norm = []
    cy_norm = []
    for a in cls_anns:
        if a["image_id"] in img_dims:
            w, h = img_dims[a["image_id"]]
            cx = (a["bbox"][0] + a["bbox"][2] / 2) / w
            cy = (a["bbox"][1] + a["bbox"][3] / 2) / h
            cx_norm.append(min(max(cx, 0), 1))
            cy_norm.append(min(max(cy, 0), 1))

    ax = axes[cls_id]
    ax.hist2d(cx_norm, cy_norm, bins=50, cmap="hot", cmin=1)
    ax.set_title(f"{CLASS_NAMES[cls_id]} (n={len(cx_norm):,})")
    ax.set_xlabel("X (normalize)")
    ax.set_ylabel("Y (normalize)")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Y ekseni ters (görüntü koordinatları)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_merkez_isi_haritasi.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 8: Merkez ısı haritası kaydedildi")


# =====================================================================
# 10. GRAFIK 9: BOUNDING BOX EN-BOY ORANI (ASPECT RATIO)
# =====================================================================
#
#  AÇIKLAMA: Anchor-based modellerde (YOLO, Faster R-CNN) doğru
#  anchor oranlarını belirlemek kritiktir. Bu grafik, her sınıf
#  için bounding box en-boy oranlarının dağılımını gösterir.
#  Person genellikle dikey (aspect < 1), Car yatay (aspect > 1)
#  olma eğilimindedir. Bu bilgi, anchor tasarımına rehberlik eder.
# =====================================================================

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Grafik 9: BBox En-Boy Oranı (Width/Height)", fontsize=15, fontweight="bold")

for cls_id in sorted(CLASS_NAMES.keys()):
    cls_anns = [a for a in all_annotations if a["category_id"] == cls_id]
    ratios = [a["bbox"][2] / a["bbox"][3] if a["bbox"][3] > 0 else 0 for a in cls_anns]
    # %1-99 percentile aralığında göster (outlier temizleme)
    p1, p99 = np.percentile(ratios, 1), np.percentile(ratios, 99)
    ratios = [r for r in ratios if p1 <= r <= p99]
    ax.hist(ratios, bins=80, alpha=0.6, color=CLASS_COLORS[cls_id],
            label=f"{CLASS_NAMES[cls_id]} (med={np.median(ratios):.2f})",
            edgecolor="black", linewidth=0.2)

ax.set_xlabel("Aspect Ratio (Width / Height)")
ax.set_ylabel("Annotation Sayısı")
ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="Kare (1:1)")
ax.legend(fontsize=11)
ax.set_title("Sınıf Bazında BBox Aspect Ratio Dağılımı")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_aspect_ratio.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 9: Aspect ratio dağılımı kaydedildi")


# =====================================================================
# 11. GRAFIK 10: BBox ALAN DAĞILIMI (LOG SCALE)
# =====================================================================
#
#  AÇIKLAMA: Nesne alanlarının logaritmik dağılımı, küçük ve büyük
#  nesneler arasındaki farkı daha iyi anlamamızı sağlar. Lineer
#  ölçekte küçük nesneler kaybolurken, log ölçekte tüm boyutlar
#  görünür hale gelir. Bu grafik, modelin hangi ölçeklerde daha
#  çok veriyle eğitileceğini gösterir.
# =====================================================================

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Grafik 10: BBox Alan Dağılımı (Log Ölçek)", fontsize=15, fontweight="bold")

for cls_id in sorted(CLASS_NAMES.keys()):
    cls_anns = [a for a in all_annotations if a["category_id"] == cls_id]
    areas = [a["area"] for a in cls_anns if a["area"] > 0]
    log_areas = np.log10(areas)
    ax.hist(log_areas, bins=80, alpha=0.6, color=CLASS_COLORS[cls_id],
            label=f"{CLASS_NAMES[cls_id]} (med={np.median(areas):.0f} px²)",
            edgecolor="black", linewidth=0.2)

# COCO sınır çizgileri
ax.axvline(np.log10(1024), color="orange", linestyle="--", alpha=0.7, label="Small/Medium sınırı (32²)")
ax.axvline(np.log10(9216), color="green", linestyle="--", alpha=0.7, label="Medium/Large sınırı (96²)")
ax.set_xlabel("log₁₀(Alan px²)")
ax.set_ylabel("Annotation Sayısı")
ax.legend(fontsize=10)
ax.set_title("Sınıf Bazında BBox Alan Dağılımı")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_alan_dagilimi_log.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 10: Alan dağılımı (log) kaydedildi")


# =====================================================================
# 12. GRAFIK 11: CO-OCCURRENCE MATRİSİ (SINIFLAR BİRLİKTE Mİ?)
# =====================================================================
#
#  AÇIKLAMA: Hangi sınıfların aynı görüntüde birlikte bulunduğunu
#  gösterir. Örneğin, Person ve Car sıklıkla birlikte görünüyorsa,
#  model bu bağlamı (context) öğrenebilir. Bu bilgi, veri setinin
#  gerçek dünya senaryolarını ne kadar iyi temsil ettiğini anlamamıza
#  yardımcı olur.
# =====================================================================

# Her görüntüdeki sınıfları bul
img_classes = defaultdict(set)
for ann in all_annotations:
    img_classes[ann["image_id"]].add(ann["category_id"])

n_classes = len(CLASS_NAMES)
cooccurrence = np.zeros((n_classes, n_classes), dtype=int)
for img_id, classes in img_classes.items():
    for c1 in classes:
        for c2 in classes:
            cooccurrence[c1][c2] += 1

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("Grafik 11: Sınıf Birlikte Görünme Matrisi (Co-occurrence)",
             fontsize=15, fontweight="bold")

im = ax.imshow(cooccurrence, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels([CLASS_NAMES[i] for i in range(n_classes)])
ax.set_yticklabels([CLASS_NAMES[i] for i in range(n_classes)])

# Değerleri hücrelere yaz
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(j, i, f"{cooccurrence[i][j]:,}", ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="white" if cooccurrence[i][j] > cooccurrence.max() * 0.6 else "black")

plt.colorbar(im, ax=ax, label="Görüntü Sayısı")
ax.set_title("Diyagonel: sınıfın bulunduğu görüntü sayısı\nDiğer: birlikte bulunma sayısı")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_cooccurrence.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 11: Co-occurrence matrisi kaydedildi")


# =====================================================================
# 13. GRAFIK 12: ÖRNEK GÖRÜNTÜLER VE BOUNDING BOX'LAR
# =====================================================================
#
#  AÇIKLAMA: Veri setindeki gerçek görüntüleri ve üzerlerindeki
#  etiketleri görmek, veri kalitesini ve etiketleme doğruluğunu
#  değerlendirmenin en doğrudan yoludur. Yanlış etiketler, çok
#  küçük/büyük bbox'lar veya eksik etiketler burada fark edilebilir.
#  Bu "sanity check", eğitim öncesinde yapılması gereken en önemli
#  adımlardan biridir.
# =====================================================================

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("Grafik 12: Rastgele Örnek Görüntüler ve BBox'lar", fontsize=15, fontweight="bold")

# train setinden rastgele 12 görüntü seç (annotation'ı olan)
train_img_ids_with_anns = list({a["image_id"] for a in all_data["train"]["annotations"]})
random.seed(42)
sample_ids = random.sample(train_img_ids_with_anns, min(12, len(train_img_ids_with_anns)))

train_img_map = {img["id"]: img for img in all_data["train"]["images"]}
train_ann_by_img = defaultdict(list)
for ann in all_data["train"]["annotations"]:
    train_ann_by_img[ann["image_id"]].append(ann)

for idx, img_id in enumerate(sample_ids):
    ax = axes[idx // 4][idx % 4]
    img_info = train_img_map[img_id]
    img_path = DATASET_DIR / "train" / "images" / img_info["file_name"]

    try:
        img = Image.open(img_path)
        ax.imshow(img, cmap="gray")
    except Exception:
        ax.text(0.5, 0.5, "Goruntu\nyuklenemedi", ha="center", va="center", transform=ax.transAxes)

    # BBox'ları çiz
    for ann in train_ann_by_img[img_id]:
        x, y, w, h = ann["bbox"]
        cls_id = ann["category_id"]
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5,
                                  edgecolor=CLASS_COLORS[cls_id],
                                  facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 2, CLASS_NAMES[cls_id], fontsize=7,
                color=CLASS_COLORS[cls_id], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5))

    ax.set_title(f"ID:{img_id} ({len(train_ann_by_img[img_id])} obj)", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "12_ornek_goruntuler.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 12: Örnek görüntüler kaydedildi")


# =====================================================================
# 14. GRAFIK 13: KAYNAK VERİ SETİ BAZINDA DAĞILIM
# =====================================================================
#
#  AÇIKLAMA: Birleştirilmiş veri setimiz 5 farklı kaynaktan oluşuyor.
#  Her kaynağın ne kadar katkıda bulunduğunu bilmek, potansiyel
#  domain shift sorunlarını anlamamıza yardımcı olur. Eğer bir kaynak
#  çok baskınsa, model ağırlıklı olarak o kaynağın özelliklerini
#  öğrenebilir.
# =====================================================================

# Dosya isimlerinden kaynak veri setini belirle
source_prefixes = {
    "drone_thermal": "Drone Thermal Model",
    "thermal_v1": "thermal.v1i.coco",
    "thermal_v1_alt": "thermal.v1i.coco (alt)",
    "hituav": "HIT-UAV",
    "rgbt_tiny": "RGBT-Tiny",
}

source_img_counts = Counter()
source_ann_counts = Counter()

for img in all_images:
    prefix = img["file_name"].rsplit("_", 1)[0]
    source = source_prefixes.get(prefix, "Bilinmeyen")
    source_img_counts[source] += 1

# Annotation için image_id -> source mapping
img_source = {}
for img in all_images:
    prefix = img["file_name"].rsplit("_", 1)[0]
    img_source[img["id"]] = source_prefixes.get(prefix, "Bilinmeyen")

for ann in all_annotations:
    source = img_source.get(ann["image_id"], "Bilinmeyen")
    source_ann_counts[source] += 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Grafik 13: Kaynak Veri Seti Bazında Dağılım", fontsize=15, fontweight="bold")

src_names = list(source_img_counts.keys())
src_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

# Görüntü
bars = ax1.barh(src_names, [source_img_counts[s] for s in src_names],
                color=src_colors[:len(src_names)], edgecolor="black", linewidth=0.5)
ax1.set_title("Görüntü Sayısı")
ax1.set_xlabel("Görüntü")
for bar, name in zip(bars, src_names):
    ax1.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
             f"{source_img_counts[name]:,}", va="center", fontweight="bold")

# Annotation
bars = ax2.barh(src_names, [source_ann_counts[s] for s in src_names],
                color=src_colors[:len(src_names)], edgecolor="black", linewidth=0.5)
ax2.set_title("Annotation Sayısı")
ax2.set_xlabel("Annotation")
for bar, name in zip(bars, src_names):
    ax2.text(bar.get_width() + 1000, bar.get_y() + bar.get_height() / 2,
             f"{source_ann_counts[name]:,}", va="center", fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "13_kaynak_dagilimi.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 13: Kaynak dağılımı kaydedildi")


# =====================================================================
# 15. GRAFIK 14: SINIF BAZINDA BBox GENİŞLİK VE YÜKSEKLİK VIOLIN PLOT
# =====================================================================
#
#  AÇIKLAMA: Violin plot, box plot'un aksine dağılımın tam şeklini
#  gösterir. Bu sayede, örneğin Person sınıfının bbox yüksekliğinin
#  bimodal (iki tepeli) olup olmadığını görebiliriz. Böyle bir durum,
#  yakın ve uzak nesnelerin farklı boyut dağılımlarına sahip olduğunu
#  gösterir ve multi-scale tespit stratejisi gerektirir.
# =====================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Grafik 14: BBox Genişlik ve Yükseklik Dağılımı (Violin Plot)",
             fontsize=15, fontweight="bold")

# Her sınıf için bbox genişlik ve yükseklik verisi hazırla
bbox_w_data = []
bbox_h_data = []
violin_labels = []
violin_colors = []

for cls_id in sorted(CLASS_NAMES.keys()):
    cls_anns = [a for a in all_annotations if a["category_id"] == cls_id]
    # Aşırı uçları kırp (outlier etkisini azalt)
    ws = [a["bbox"][2] for a in cls_anns]
    hs = [a["bbox"][3] for a in cls_anns]
    w_cap = np.percentile(ws, 98)
    h_cap = np.percentile(hs, 98)
    bbox_w_data.append([w for w in ws if w <= w_cap])
    bbox_h_data.append([h for h in hs if h <= h_cap])
    violin_labels.append(CLASS_NAMES[cls_id])
    violin_colors.append(CLASS_COLORS[cls_id])

# Genişlik violin
vp1 = ax1.violinplot(bbox_w_data, showmedians=True, showextrema=True)
for i, body in enumerate(vp1["bodies"]):
    body.set_facecolor(violin_colors[i])
    body.set_alpha(0.7)
ax1.set_xticks(range(1, len(violin_labels) + 1))
ax1.set_xticklabels(violin_labels)
ax1.set_ylabel("BBox Genişlik (px)")
ax1.set_title("Genişlik Dağılımı")

# Yükseklik violin
vp2 = ax2.violinplot(bbox_h_data, showmedians=True, showextrema=True)
for i, body in enumerate(vp2["bodies"]):
    body.set_facecolor(violin_colors[i])
    body.set_alpha(0.7)
ax2.set_xticks(range(1, len(violin_labels) + 1))
ax2.set_xticklabels(violin_labels)
ax2.set_ylabel("BBox Yükseklik (px)")
ax2.set_title("Yükseklik Dağılımı")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "14_violin_bbox.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 14: Violin plot kaydedildi")


# =====================================================================
# 16. GRAFIK 15: ÖZET İSTATİSTİK TABLOSU
# =====================================================================
#
#  AÇIKLAMA: Tüm önemli istatistikleri tek bir tablo halinde sunmak,
#  veri setinin genel profilini hızlıca kavramak için kullanışlıdır.
#  Bu tablo, raporlarda ve sunumlarda referans olarak kullanılabilir.
# =====================================================================

fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle("Grafik 15: Veri Seti Özet İstatistikleri", fontsize=15, fontweight="bold")
ax.axis("off")

# İstatistikleri hesapla
rows = []

# Genel
rows.append(["GENEL", "", "", ""])
rows.append(["Toplam Goruntu", f"{len(all_images):,}", "", ""])
rows.append(["Toplam Annotation", f"{len(all_annotations):,}", "", ""])
rows.append(["Ort. Nesne/Goruntu", f"{np.mean(counts_list):.1f}", "", ""])
rows.append(["Medyan Nesne/Goruntu", f"{np.median(counts_list):.0f}", "", ""])
rows.append(["", "", "", ""])

# Split bazında
rows.append(["SPLIT BAZINDA", "Train", "Val", "Test"])
rows.append(["Goruntu Sayisi",
             f"{len(all_data['train']['images']):,}",
             f"{len(all_data['val']['images']):,}",
             f"{len(all_data['test']['images']):,}"])
rows.append(["Annotation Sayisi",
             f"{len(all_data['train']['annotations']):,}",
             f"{len(all_data['val']['annotations']):,}",
             f"{len(all_data['test']['annotations']):,}"])
rows.append(["", "", "", ""])

# Sınıf bazında
rows.append(["SINIF BAZINDA", "Person", "Car", "OtherVehicle"])
total_per_class = Counter(a["category_id"] for a in all_annotations)
rows.append(["Toplam Annotation",
             f"{total_per_class[0]:,}", f"{total_per_class[1]:,}", f"{total_per_class[2]:,}"])
rows.append(["Oran (%)",
             f"{total_per_class[0]/len(all_annotations)*100:.1f}%",
             f"{total_per_class[1]/len(all_annotations)*100:.1f}%",
             f"{total_per_class[2]/len(all_annotations)*100:.1f}%"])

# Medyan bbox boyutları
for cls_id, name in CLASS_NAMES.items():
    cls_anns = [a for a in all_annotations if a["category_id"] == cls_id]
    med_w = np.median([a["bbox"][2] for a in cls_anns])
    med_h = np.median([a["bbox"][3] for a in cls_anns])
    med_a = np.median([a["area"] for a in cls_anns])
    if cls_id == 0:
        rows.append(["Medyan BBox (WxH)",
                      f"{med_w:.0f}x{med_h:.0f}", "", ""])
    rows[-1][cls_id + 1] = f"{med_w:.0f}x{med_h:.0f}"

rows.append(["", "", "", ""])
rows.append(["KAYNAK", "Goruntu", "Annotation", ""])
for src_name in source_img_counts:
    rows.append([src_name,
                 f"{source_img_counts[src_name]:,}",
                 f"{source_ann_counts[src_name]:,}", ""])

table = ax.table(cellText=rows, colLabels=["", "Deger 1", "Deger 2", "Deger 3"],
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.4)

# Başlık satırlarını renklendir
for i, row in enumerate(rows):
    if row[0] in ["GENEL", "SPLIT BAZINDA", "SINIF BAZINDA", "KAYNAK"]:
        for j in range(4):
            table[i + 1, j].set_facecolor("#34495e")
            table[i + 1, j].set_text_props(color="white", fontweight="bold")

# Kolon başlıklarını renklendir
for j in range(4):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "15_ozet_istatistikler.png", bbox_inches="tight")
plt.close()
print("[OK] Grafik 15: Özet istatistik tablosu kaydedildi")


# =====================================================================
#  SONUÇ
# =====================================================================
print("\n" + "=" * 60)
print("  EDA TAMAMLANDI!")
print("=" * 60)
print(f"\n  Toplam 15 grafik olusturuldu.")
print(f"  Cikti klasoru: {OUTPUT_DIR}")
print(f"\n  Olusturulan dosyalar:")
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"    - {f.name}")
print("\n" + "=" * 60)
