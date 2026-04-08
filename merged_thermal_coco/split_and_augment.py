"""
==========================================================================================
  TERMAL VERİ SETİ - YENİDEN BÖLME + VERİ ARTTIRMA (DATA AUGMENTATION) SCRİPTİ
==========================================================================================

  Bu script iki ana iş yapar:

    BÖLÜM A  ──  Mevcut veri setini havuzlar, karıştırır ve
                  train=%10 / val=%10 / test=%80 olarak yeniden böler.

    BÖLÜM B  ──  Eğitim (train) setine çeşitli veri arttırma (augmentation)
                  teknikleri uygulayarak yeni bir veri seti oluşturur.

  Uygulanan Augmentation Teknikleri:
    1. Yatay & Dikey Çevirme   (Horizontal / Vertical Flip)
    2. Rastgele Döndürme        (Random Rotation)
    3. Rastgele Kırpma           (Random Crop)
    4. Ölçekleme & Yeniden Boyutlandırma (Random Resized Crop)
    5. Afin Dönüşümü             (Affine Transform)
    6. Elastik Distorsiyon       (Elastic Distortion)
    7. CLAHE                     (Contrast Limited Adaptive Histogram Equalization)
    8. Gauss Gürültüsü           (Gaussian Noise Injection)
    9. Tuz & Biber Gürültüsü     (Salt & Pepper Noise)
   10. Gauss Bulanıklaştırma     (Gaussian Blur / Thermal Blur)

  Neden veri arttırma yapıyoruz?
  ─────────────────────────────
  Derin öğrenme modelleri, eğitim verisini "ezberleyebilir" (overfitting).
  Veri arttırma ile aynı görüntüyü farklı şekillerde modele sunarak:
    - Modelin genelleme yeteneğini artırırız
    - Az veri ile daha iyi sonuçlar elde ederiz
    - Modelin gerçek dünya koşullarına daha dayanıklı olmasını sağlarız

  Termal görüntüler için özel dikkat:
  ────────────────────────────────────
  Termal sensörler RGB kameralardan farklıdır. Renk bilgisi yoktur (tek kanal
  veya gri tonlama). Bu yüzden renk tabanlı augmentation'lar (color jitter,
  hue/saturation) anlamsızdır. Bunun yerine termal görüntülere özgü teknikler
  olan CLAHE, termal gürültü enjeksiyonu ve termal bulanıklaştırma kullanırız.

  Kullanılan Kütüphaneler:
    - PyTorch & Torchvision: Tensor işlemleri ve bazı dönüşümler
    - OpenCV (cv2): CLAHE, afin, elastik distorsiyon gibi gelişmiş işlemler
    - NumPy: Matematiksel işlemler ve gürültü üretimi
    - PIL (Pillow): Görüntü okuma/yazma

  COCO Format Hatırlatması:
    - bbox formatı: [x, y, width, height]  (sol-üst köşe + genişlik/yükseklik)
    - Her augmentation uygulandığında bbox'lar da aynı dönüşümle güncellenir
    - Dönüşüm sonrası bbox'lar görüntü sınırları dışına çıkabilir → kırpılır
    - Çok küçülen veya tamamen çıkan bbox'lar filtrelenir

==========================================================================================
"""

import json
import os
import random
import shutil
import math
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms.functional as TF

# ─────────────────────────────────────────────────────────────
#  AYARLAR (HYPERPARAMETERS)
# ─────────────────────────────────────────────────────────────
#
#  Bu bölümde scriptin davranışını kontrol eden parametreler var.
#  Bir öğrenci olarak, bu değerleri değiştirerek farklı sonuçlar
#  elde edebilir ve augmentation'ların etkisini gözlemleyebilirsiniz.
# ─────────────────────────────────────────────────────────────

SEED = 42                   # Tekrarlanabilirlik için sabit seed
TRAIN_RATIO = 0.720          # Eğitim seti oranı (%10)
VAL_RATIO   = 0.20          # Doğrulama seti oranı (%10)
TEST_RATIO  = 0.10          # Test seti oranı (%80)
MIN_BBOX_AREA = 100         # Augmentation sonrası minimum bbox alanı (px²)
MIN_BBOX_SIDE = 5           # Minimum bbox kenar uzunluğu (px)

# Kaynak ve çıktı klasörleri
SOURCE_DIR = Path(__file__).parent                       # Mevcut veri seti
OUTPUT_DIR = Path(__file__).parent.parent / "merged_thermal_coco_augmented"  # Augmented çıktı

SPLITS = ["train", "val", "test"]
CLASS_NAMES = {0: "Person", 1: "Car", 2: "OtherVehicle"}
TARGET_CATEGORIES = [
    {"id": 0, "name": "Person",       "supercategory": "none"},
    {"id": 1, "name": "Car",          "supercategory": "none"},
    {"id": 2, "name": "OtherVehicle", "supercategory": "none"},
]

# Tekrarlanabilirlik
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────
#  KULLANICI SORUSU: Toplam veri çarpanı
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("  VERİ ARTTIRMA ÇARPANI SEÇİMİ")
print("=" * 70)
while True:
    _cevap = input("  Toplam veri sayısı kaç katına çıksın? [2-10]: ").strip()
    if _cevap.isdigit() and 2 <= int(_cevap) <= 10:
        AUGMENT_MULTIPLIER = int(_cevap)
        break
    print("  Lütfen 2 ile 10 arasında bir tam sayı girin.")
print(f"  Seçilen çarpan: {AUGMENT_MULTIPLIER}x  "
      f"(her görüntüye {AUGMENT_MULTIPLIER - 1} rastgele augmentation uygulanacak)")
print()


# ╔═════════════════════════════════════════════════════════════╗
# ║                                                             ║
# ║   BÖLÜM A: VERİ SETİNİ YENİDEN BÖLME (RE-SPLIT)           ║
# ║                                                             ║
# ╚═════════════════════════════════════════════════════════════╝

print("=" * 70)
print("  BÖLÜM A: VERİ SETİNİ YENİDEN BÖLME")
print("  Train: %10 | Val: %10 | Test: %80")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
#  A.1) Tüm split'lerdeki verileri tek bir havuzda topluyoruz.
#
#  Neden? Mevcut veri seti farklı kaynaklardan geldiği için mevcut
#  split oranları dengesiz olabilir. Hepsini bir araya toplayıp
#  rastgele karıştırmak, her split'te her kaynaktan veri olmasını
#  sağlar ve bias'ı (önyargıyı) azaltır.
# ─────────────────────────────────────────────────────────────

print("\n[A.1] Mevcut veri yükleniyor ve havuzlanıyor...")

pool_images = []           # Tüm görüntü bilgileri
pool_anns_by_img = {}      # image_id -> [annotation listesi]
old_id_to_info = {}        # image_id -> (split, file_name, width, height)

for split in SPLITS:
    json_path = SOURCE_DIR / split / "_annotations.coco.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    for img in data["images"]:
        # Her görüntünün hangi split'ten geldiğini ve dosya yolunu kaydediyoruz
        # Böylece daha sonra dosyayı bulup kopyalayabiliriz
        old_id_to_info[img["id"]] = {
            "split": split,
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }
        pool_images.append(img["id"])

    for ann in data["annotations"]:
        pool_anns_by_img.setdefault(ann["image_id"], []).append(ann)

print(f"    Toplam havuz: {len(pool_images)} görüntü")

# ─────────────────────────────────────────────────────────────
#  A.2) Karıştırma ve bölme
#
#  random.shuffle ile listeyi rastgele sıralıyoruz. SEED sabit
#  olduğu için her çalıştırmada aynı sonucu alırız.
#  Bu, bilimsel tekrarlanabilirlik (reproducibility) için önemlidir.
# ─────────────────────────────────────────────────────────────

print("[A.2] Karıştırılıyor ve bölünüyor...")
random.shuffle(pool_images)

n_total = len(pool_images)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)
n_test  = n_total - n_train - n_val   # Kalan tamamı test'e

split_ids = {
    "train": pool_images[:n_train],
    "val":   pool_images[n_train:n_train + n_val],
    "test":  pool_images[n_train + n_val:],
}

for s, ids in split_ids.items():
    print(f"    {s:>5}: {len(ids):>6} görüntü")

# ─────────────────────────────────────────────────────────────
#  A.3) Yeni klasör yapısı oluştur ve dosyaları kopyala
#
#  Her split için images/ klasörü oluşturuyoruz ve görüntüleri
#  eski konumlarından yeni konumlarına kopyalıyoruz.
# ─────────────────────────────────────────────────────────────

print("[A.3] Klasörler oluşturuluyor ve dosyalar kopyalanıyor...")

new_data = {}   # split -> {"images": [...], "annotations": [...]}

for split in SPLITS:
    out_img_dir = OUTPUT_DIR / split / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    new_data[split] = {"images": [], "annotations": []}

img_counter = 0
ann_counter = 0

for split in SPLITS:
    for old_img_id in split_ids[split]:
        info = old_id_to_info[old_img_id]

        # Kaynak dosya yolu
        src_path = SOURCE_DIR / info["split"] / "images" / info["file_name"]
        if not src_path.exists():
            continue

        # Yeni benzersiz ID ve dosya adı oluştur
        img_counter += 1
        new_img_id = img_counter
        ext = os.path.splitext(info["file_name"])[1]
        new_filename = f"img_{new_img_id:06d}{ext}"

        # Dosyayı kopyala
        dst_path = OUTPUT_DIR / split / "images" / new_filename
        shutil.copy2(str(src_path), str(dst_path))

        # COCO image kaydı
        new_data[split]["images"].append({
            "id": new_img_id,
            "file_name": new_filename,
            "width": info["width"],
            "height": info["height"],
        })

        # Bu görüntüye ait annotation'ları kopyala
        for old_ann in pool_anns_by_img.get(old_img_id, []):
            ann_counter += 1
            new_data[split]["annotations"].append({
                "id": ann_counter,
                "image_id": new_img_id,
                "category_id": old_ann["category_id"],
                "bbox": old_ann["bbox"],
                "area": old_ann["area"],
                "segmentation": old_ann.get("segmentation", []),
                "iscrowd": old_ann.get("iscrowd", 0),
            })

# ─────────────────────────────────────────────────────────────
#  A.4) COCO JSON dosyalarını kaydet
# ─────────────────────────────────────────────────────────────

print("[A.4] COCO JSON'lar kaydediliyor...")

for split in SPLITS:
    coco_out = {
        "images": new_data[split]["images"],
        "annotations": new_data[split]["annotations"],
        "categories": TARGET_CATEGORIES,
    }
    out_path = OUTPUT_DIR / split / "_annotations.coco.json"
    with open(out_path, "w") as f:
        json.dump(coco_out, f)
    n_img = len(new_data[split]["images"])
    n_ann = len(new_data[split]["annotations"])
    print(f"    {split:>5}: {n_img:>6} görüntü, {n_ann:>7} annotation")

print("\n  [OK] Veri seti yeniden bölündü!\n")


# ╔═════════════════════════════════════════════════════════════╗
# ║                                                             ║
# ║   BÖLÜM B: VERİ ARTTIRMA (DATA AUGMENTATION)               ║
# ║                                                             ║
# ╚═════════════════════════════════════════════════════════════╝
#
#  Veri arttırma SADECE eğitim (train) setine uygulanır.
#  Val ve test setleri dokunulmadan kalır çünkü model
#  değerlendirmesi orijinal, bozulmamış veriler üzerinde yapılmalıdır.
#
#  Her augmentation fonksiyonu şu imzaya sahiptir:
#    augment_xxx(image_np, bboxes, img_w, img_h) -> (new_image, new_bboxes)
#
#  Burada:
#    - image_np: NumPy dizisi olarak görüntü (HxWxC veya HxW)
#    - bboxes: [[cat_id, x, y, w, h], ...] listesi
#    - img_w, img_h: görüntü boyutları
#    - Dönüş: (dönüştürülmüş görüntü, dönüştürülmüş bbox'lar)

print("=" * 70)
print("  BÖLÜM B: VERİ ARTTIRMA (AUGMENTATION)")
print("  Sadece TRAIN setine uygulanacak")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────
#  YARDIMCI FONKSİYONLAR (UTILITY FUNCTIONS)
# ─────────────────────────────────────────────────────────────────

def clip_bbox(bbox, img_w, img_h):
    """
    Bounding box'ı görüntü sınırları içine kırpar.

    Augmentation sonrası bbox'lar görüntü dışına taşabilir.
    Örneğin, bir döndürme işlemi sonrası bbox'ın bir köşesi
    negatif koordinatlara düşebilir. Bu fonksiyon, bbox'ı
    görüntü sınırlarına sığacak şekilde kırpar.

    Parametreler:
        bbox: [cat_id, x, y, w, h]
        img_w, img_h: görüntü genişlik ve yüksekliği

    Döndürür:
        Kırpılmış bbox veya None (bbox tamamen dışarıdaysa)
    """
    cat_id, x, y, w, h = bbox

    # Bbox'ın sağ-alt köşesini hesapla
    x2 = x + w
    y2 = y + h

    # Görüntü sınırlarına kırp
    x  = max(0, x)
    y  = max(0, y)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # Yeni genişlik ve yükseklik
    new_w = x2 - x
    new_h = y2 - y

    # Çok küçük veya geçersiz bbox'ları filtrele
    if new_w < MIN_BBOX_SIDE or new_h < MIN_BBOX_SIDE:
        return None
    if new_w * new_h < MIN_BBOX_AREA:
        return None

    return [cat_id, round(x, 2), round(y, 2), round(new_w, 2), round(new_h, 2)]


def ensure_gray(img):
    """
    Görüntüyü gri tonlamaya çevirir (termal görüntüler tek kanaldır).
    Eğer zaten gri ise dokunmaz.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def gray_to_3ch(img):
    """Tek kanallı görüntüyü 3 kanala çevirir (kaydetmek için)."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# ─────────────────────────────────────────────────────────────────
#  AUGMENTATION 1: YATAY ve DİKEY ÇEVİRME (FLIPPING)
# ─────────────────────────────────────────────────────────────────
#
#  Çevirme, en basit ve en etkili augmentation tekniklerinden biridir.
#
#  Yatay çevirme (ayna görüntüsü):
#  ┌──────┐         ┌──────┐
#  │ A  B │   →→→   │ B  A │
#  │ C  D │         │ D  C │
#  └──────┘         └──────┘
#
#  Matematiksel olarak:
#    Yeni_x = Görüntü_Genişliği - Eski_x - BBox_Genişliği
#    Yeni_y değişmez
#
#  Termal görüntülerde çevirme güvenlidir çünkü termal emisyon
#  yönden bağımsızdır (bir insan soldan da sağdan da aynı ısıyı yayar).
# ─────────────────────────────────────────────────────────────────

def augment_horizontal_flip(img, bboxes, img_w, img_h):
    """Görüntüyü yatay eksende çevirir (ayna görüntüsü)."""
    # PyTorch'un fonksiyonel API'sini kullanarak çevirme
    # Önce numpy -> PIL -> torch tensor -> flip -> numpy
    tensor = TF.to_tensor(img)          # (C, H, W) torch tensor
    flipped = TF.hflip(tensor)          # Yatay çevirme
    new_img = (flipped.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    new_bboxes = []
    for cat_id, x, y, w, h in bboxes:
        # Yatay çevirmede x koordinatı ayna'lanır
        new_x = img_w - x - w
        result = clip_bbox([cat_id, new_x, y, w, h], img_w, img_h)
        if result:
            new_bboxes.append(result)

    return new_img, new_bboxes


def augment_vertical_flip(img, bboxes, img_w, img_h):
    """Görüntüyü dikey eksende çevirir (baş aşağı)."""
    tensor = TF.to_tensor(img)
    flipped = TF.vflip(tensor)
    new_img = (flipped.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    new_bboxes = []
    for cat_id, x, y, w, h in bboxes:
        # Dikey çevirmede y koordinatı ayna'lanır
        new_y = img_h - y - h
        result = clip_bbox([cat_id, x, new_y, w, h], img_w, img_h)
        if result:
            new_bboxes.append(result)

    return new_img, new_bboxes


# ─────────────────────────────────────────────────────────────────
#  AUGMENTATION 2: RASTGELE DÖNDÜRME (RANDOM ROTATION)
# ─────────────────────────────────────────────────────────────────
#
#  Döndürme, nesnelerin açısal varyasyonlarını öğretir.
#  Drone termal görüntülerinde drone'un yaw (sapma) açısı
#  değişebildiği için döndürme çok gerçekçi bir augmentation'dır.
#
#  Döndürme matrisi (2D):
#    ┌ cos(θ)  -sin(θ) ┐
#    │ sin(θ)   cos(θ) │
#    └                  ┘
#
#  BBox dönüşümü: Bbox'ın 4 köşesini döndürme matrisiyle çarpıp,
#  yeni axis-aligned (eksenlere paralel) bbox hesaplıyoruz.
#
#  Dikkat: Büyük açılarda (>45°) bbox'lar çok genişleyebilir.
#  Bu yüzden [-30°, +30°] aralığında kalıyoruz.
# ─────────────────────────────────────────────────────────────────

def augment_rotation(img, bboxes, img_w, img_h):
    """Görüntüyü rastgele bir açıyla döndürür (-30° ile +30° arası)."""
    angle = random.uniform(-30, 30)

    # OpenCV ile döndürme (merkez etrafında)
    center = (img_w / 2, img_h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Döndürme matrisi
    new_img = cv2.warpAffine(img, M, (img_w, img_h), borderValue=0)

    # Radyan cinsine çevir (numpy trigonometrik fonksiyonlar radyan kullanır)
    rad = math.radians(-angle)  # OpenCV'nin açı yönü ters
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    new_bboxes = []
    for cat_id, x, y, w, h in bboxes:
        # Bbox'ın 4 köşesini hesapla
        corners = np.array([
            [x,     y],         # Sol üst
            [x + w, y],         # Sağ üst
            [x + w, y + h],     # Sağ alt
            [x,     y + h],     # Sol alt
        ], dtype=np.float64)

        # Her köşeyi döndürme matrisiyle dönüştür
        # M: 2x3 affine matris, köşe: [x, y, 1]
        ones = np.ones((4, 1))
        corners_h = np.hstack([corners, ones])           # Homojen koordinat
        rotated = (M @ corners_h.T).T                    # Matris çarpımı

        # Yeni axis-aligned bbox: döndürülmüş köşelerin min/max'ı
        rx_min = rotated[:, 0].min()
        ry_min = rotated[:, 1].min()
        rx_max = rotated[:, 0].max()
        ry_max = rotated[:, 1].max()

        result = clip_bbox(
            [cat_id, rx_min, ry_min, rx_max - rx_min, ry_max - ry_min],
            img_w, img_h
        )
        if result:
            new_bboxes.append(result)

    return new_img, new_bboxes


# ─────────────────────────────────────────────────────────────────
#  AUGMENTATION 3: RASTGELE KIRPMA (RANDOM CROP)
# ─────────────────────────────────────────────────────────────────
#
#  Kırpma, görüntünün bir bölgesini keser ve yeni görüntü olarak
#  kullanır. Bu teknik:
#    - Modelin nesneleri kısmen görünür durumda tanımasını öğretir
#    - Arka plan çeşitliliğini artırır
#    - Nesne-odaklı öğrenmeyi destekler
#
#  Orijinal boyutun %60-%90'ı kadar bir alan kırpılır.
#  Kırpma sonrası görüntü orijinal boyuta yeniden boyutlandırılır.
# ─────────────────────────────────────────────────────────────────

def augment_random_crop(img, bboxes, img_w, img_h):
    """Görüntüden rastgele bir bölge kırpar ve orijinal boyuta döndürür."""
    # Kırpma oranı: orijinalin %60-%90'ı
    crop_ratio = random.uniform(0.6, 0.9)
    crop_w = int(img_w * crop_ratio)
    crop_h = int(img_h * crop_ratio)

    # Rastgele kırpma başlangıç noktası
    x_start = random.randint(0, img_w - crop_w)
    y_start = random.randint(0, img_h - crop_h)

    # Kırp
    cropped = img[y_start:y_start + crop_h, x_start:x_start + crop_w].copy()

    # Orijinal boyuta geri ölçekle
    new_img = cv2.resize(cropped, (img_w, img_h))

    # Ölçek faktörleri
    scale_x = img_w / crop_w
    scale_y = img_h / crop_h

    new_bboxes = []
    for cat_id, x, y, w, h in bboxes:
        # Bbox'ı kırpma bölgesine göre kaydır
        new_x = (x - x_start) * scale_x
        new_y = (y - y_start) * scale_y
        new_w = w * scale_x
        new_h = h * scale_y

        result = clip_bbox([cat_id, new_x, new_y, new_w, new_h], img_w, img_h)
        if result:
            new_bboxes.append(result)

    return new_img, new_bboxes


# ─────────────────────────────────────────────────────────────────
#  AUGMENTATION 4: ÖLÇEKLEME & RANDOM RESIZED CROP
# ─────────────────────────────────────────────────────────────────
#
#  Bu teknik PyTorch'un RandomResizedCrop dönüşümünün mantığını
#  taklit eder. Görüntüden rastgele bir bölge seçer, rastgele bir
#  aspect ratio ile kırpar ve hedef boyuta yeniden boyutlandırır.
#
#  Scale: Orijinal alanın ne kadarının kullanılacağı (0.5 → %50)
#  Ratio: Kırpma bölgesinin en-boy oranı
#
#  Bu augmentation, modelin farklı ölçeklerdeki nesneleri
#  tanımasını sağlar. Uzaktaki küçük bir araba ile yakındaki
#  büyük bir araba aynı sınıf olmalıdır.
# ─────────────────────────────────────────────────────────────────

def augment_random_resized_crop(img, bboxes, img_w, img_h):
    """PyTorch RandomResizedCrop mantığı: ölçekle + kırp + boyutlandır."""
    # Hedef alan: orijinalin %50-%100'ü
    scale = random.uniform(0.5, 1.0)
    # Aspect ratio: 3/4 ile 4/3 arası
    ratio = random.uniform(0.75, 1.33)

    area = img_w * img_h * scale
    crop_w = int(math.sqrt(area * ratio))
    crop_h = int(math.sqrt(area / ratio))

    # Sınırları aşmamasını sağla
    crop_w = min(crop_w, img_w)
    crop_h = min(crop_h, img_h)

    x_start = random.randint(0, img_w - crop_w)
    y_start = random.randint(0, img_h - crop_h)

    cropped = img[y_start:y_start + crop_h, x_start:x_start + crop_w].copy()
    new_img = cv2.resize(cropped, (img_w, img_h))

    scale_x = img_w / crop_w
    scale_y = img_h / crop_h

    new_bboxes = []
    for cat_id, x, y, w, h in bboxes:
        new_x = (x - x_start) * scale_x
        new_y = (y - y_start) * scale_y
        new_w = w * scale_x
        new_h = h * scale_y
        result = clip_bbox([cat_id, new_x, new_y, new_w, new_h], img_w, img_h)
        if result:
            new_bboxes.append(result)

    return new_img, new_bboxes


# ─────────────────────────────────────────────────────────────────
#  AUGMENTATION 5: AFİN DÖNÜŞÜMÜ (AFFINE TRANSFORM)
# ─────────────────────────────────────────────────────────────────
#
#  Afin dönüşüm, doğrusal bir geometrik dönüşümdür. Döndürme,
#  ölçekleme, kaydırma (translation) ve yamultma (shear) işlemlerini
#  tek bir 2x3 matris ile ifade eder:
#
#    ┌ a  b  tx ┐   →  x' = a·x + b·y + tx
#    │ c  d  ty │   →  y' = c·x + d·y + ty
#    └          ┘
#
#  Burada (tx, ty) kaydırma, (a,b,c,d) döndürme+ölçekleme+yamultma.
#
#  Termal drone görüntülerinde drone'un eğimi (pitch/roll) afin
#  dönüşümlere yol açar, bu yüzden gerçekçi bir augmentation'dır.
# ─────────────────────────────────────────────────────────────────

def augment_affine(img, bboxes, img_w, img_h):
    """Rastgele afin dönüşüm uygular (yamultma + hafif döndürme)."""
    # Kaynak noktalar (görüntünün 3 köşesi)
    src_pts = np.float32([
        [0, 0],
        [img_w - 1, 0],
        [0, img_h - 1],
    ])

    # Hedef noktaları hafifçe kaydır (rastgele pertürbasyon)
    # Maksimum kaydırma: görüntü boyutunun %10'u
    max_shift_x = img_w * 0.10
    max_shift_y = img_h * 0.10

    dst_pts = np.float32([
        [random.uniform(0, max_shift_x),       random.uniform(0, max_shift_y)],
        [img_w - 1 - random.uniform(0, max_shift_x), random.uniform(0, max_shift_y)],
        [random.uniform(0, max_shift_x),       img_h - 1 - random.uniform(0, max_shift_y)],
    ])

    # Afin matrisini hesapla
    M = cv2.getAffineTransform(src_pts, dst_pts)

    # Görüntüyü dönüştür
    new_img = cv2.warpAffine(img, M, (img_w, img_h), borderValue=0)

    # Bbox köşelerini dönüştür
    new_bboxes = []
    for cat_id, x, y, w, h in bboxes:
        corners = np.array([
            [x, y, 1],
            [x + w, y, 1],
            [x + w, y + h, 1],
            [x, y + h, 1],
        ], dtype=np.float64)

        transformed = (M @ corners.T).T

        rx_min = transformed[:, 0].min()
        ry_min = transformed[:, 1].min()
        rx_max = transformed[:, 0].max()
        ry_max = transformed[:, 1].max()

        result = clip_bbox(
            [cat_id, rx_min, ry_min, rx_max - rx_min, ry_max - ry_min],
            img_w, img_h
        )
        if result:
            new_bboxes.append(result)

    return new_img, new_bboxes


# ─────────────────────────────────────────────────────────────────
#  AUGMENTATION 6: ELASTİK DİSTORSİYON (ELASTIC DISTORTION)
# ─────────────────────────────────────────────────────────────────
#
#  Elastik distorsiyon, görüntüyü sanki lastik bir yüzeymiş gibi
#  büker. Her piksel, rastgele bir yönde hafifçe kaydırılır.
#
#  Algoritma (Simard et al., 2003):
#    1. Rastgele bir yer değiştirme alanı (displacement field) üret
#    2. Bu alanı Gauss filtresi ile yumuşat (sigma parametresi)
#    3. Alpha ile ölçekle (distorsiyonun şiddeti)
#    4. Her pikseli bu alana göre kaydır
#
#  Termal görüntülerde atmosferik türbülans benzer distorsiyonlara
#  yol açabildiği için gerçekçi bir augmentation'dır.
#
#  NOT: Elastik distorsiyon genellikle küçük deformasyonlar yapar,
#  bu yüzden bbox'lar üzerindeki etkisi ihmal edilebilir düzeydedir.
#  Bbox'ları değiştirmeden bırakıyoruz (yaygın uygulama).
# ─────────────────────────────────────────────────────────────────

def augment_elastic(img, bboxes, img_w, img_h):
    """Elastik distorsiyon uygular (lastik büküm efekti)."""
    alpha = img_w * 0.08     # Distorsiyon şiddeti
    sigma = img_w * 0.08     # Gauss yumuşatma parametresi

    # Rastgele yer değiştirme alanları (x ve y yönünde)
    dx = np.random.uniform(-1, 1, (img_h, img_w)).astype(np.float32)
    dy = np.random.uniform(-1, 1, (img_h, img_w)).astype(np.float32)

    # Gauss filtresi ile yumuşat (keskin geçişleri önler)
    ksize = int(6 * sigma + 1) | 1   # Çekirdek boyutu (tek sayı olmalı)
    dx = cv2.GaussianBlur(dx, (ksize, ksize), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (ksize, ksize), sigma) * alpha

    # Koordinat ızgarası oluştur
    x_grid, y_grid = np.meshgrid(np.arange(img_w), np.arange(img_h))
    map_x = (x_grid + dx).astype(np.float32)
    map_y = (y_grid + dy).astype(np.float32)

    # Pikselleri yeni konumlarından örnekle (remap)
    new_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)

    # Bbox'lar küçük deformasyon altında değişmez (yaygın varsayım)
    return new_img, bboxes


# ─────────────────────────────────────────────────────────────────
#  AUGMENTATION 7: CLAHE
#  (Contrast Limited Adaptive Histogram Equalization)
# ─────────────────────────────────────────────────────────────────
#
#  TERMAL GÖRÜNTÜLER İÇİN ÇOK ÖNEMLİ BİR TEKNİK!
#
#  Problem: Termal sensörler genellikle dar bir dinamik aralığa sahiptir.
#  Bu, görüntünün düşük kontrastlı (soluk, gri) görünmesine yol açar.
#
#  Standart Histogram Eşitleme:
#    Tüm görüntünün histogramını düzleştirir.
#    Sorun: Termal gürültüyü de abartır (amplify) → kötü sonuç
#
#  CLAHE (Daha İyi Yöntem):
#    - Görüntüyü küçük bloklara (tiles) böler (örn. 8x8)
#    - Her bloğun histogramını ayrı ayrı eşitler
#    - Kontrastı bir eşik değeri (clipLimit) ile sınırlar
#    - Blok sınırlarını bilinear interpolasyon ile yumuşatır
#
#  Parametreler:
#    clipLimit: Kontrast sınırlama eşiği (yüksek = daha fazla kontrast)
#    tileGridSize: Blok boyutu (küçük = daha yerel etki)
#
#  CLAHE sadece piksel değerlerini değiştirir, geometriyi değiştirmez.
#  Bu yüzden bbox'lar aynı kalır.
# ─────────────────────────────────────────────────────────────────

def augment_clahe(img, bboxes, img_w, img_h):
    """
    CLAHE kontrast iyileştirmesi uygular.
    Termal görüntülerde düşük kontrastı iyileştirir ve
    gürültüyü kontrol altında tutar.
    """
    # Gri tonlamaya çevir (CLAHE tek kanal üzerinde çalışır)
    gray = ensure_gray(img)

    # CLAHE nesnesi oluştur
    # clipLimit=2.0-4.0 arası termal görüntüler için iyi çalışır
    clip_limit = random.uniform(2.0, 4.0)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)   # 8x8 piksellik bloklar
    )

    # CLAHE uygula
    enhanced = clahe.apply(gray)

    # 3 kanala geri çevir
    new_img = gray_to_3ch(enhanced)

    # CLAHE geometriyi değiştirmez → bbox'lar aynı kalır
    return new_img, bboxes


# ─────────────────────────────────────────────────────────────────
#  AUGMENTATION 8: GAUSS GÜRÜLTÜSÜ (GAUSSIAN NOISE INJECTION)
# ─────────────────────────────────────────────────────────────────
#
#  Gauss gürültüsü, her piksele normal dağılımlı (Gaussian) rastgele
#  bir değer ekler:
#
#    I_noisy(x,y) = I_original(x,y) + N(μ=0, σ)
#
#  Burada σ (sigma) gürültünün standart sapmasıdır.
#
#  Sinyal-Gürültü Oranı (SNR):
#    SNR(dB) = 10 · log₁₀(P_sinyal / P_gürültü)
#
#    20 dB SNR → düşük kalite, askeri/medikal zorlayıcı senaryolar
#    35 dB SNR → yüksek kalite, normal çalışma koşulları
#
#  Neden gürültü ekliyoruz?
#    - Gerçek termal sensörler her zaman gürültü üretir (termal gürültü)
#    - Model, gürültülü koşullarda da çalışabilmeyi öğrenir
#    - Overfitting'i azaltır (model gürültüyü ezberleyemez)
# ─────────────────────────────────────────────────────────────────

def augment_gaussian_noise(img, bboxes, img_w, img_h):
    """
    Rastgele Gauss gürültüsü ekler.
    SNR seviyesini 20-35 dB aralığında rastgele belirler.
    """
    gray = ensure_gray(img).astype(np.float64)

    # Hedef SNR'ı rastgele seç (dB cinsinden)
    target_snr_db = random.uniform(20, 35)

    # Sinyal gücünü hesapla: P_signal = mean(I²)
    signal_power = np.mean(gray ** 2)

    # SNR formülünden gürültü gücünü hesapla:
    # SNR = 10·log10(P_signal / P_noise)
    # P_noise = P_signal / 10^(SNR/10)
    noise_power = signal_power / (10 ** (target_snr_db / 10))

    # Gauss gürültüsü üret (ortalaması 0, std = sqrt(P_noise))
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, gray.shape)

    # Gürültüyü ekle ve [0, 255] aralığına kırp
    noisy = np.clip(gray + noise, 0, 255).astype(np.uint8)

    new_img = gray_to_3ch(noisy)
    return new_img, bboxes


# ─────────────────────────────────────────────────────────────────
#  AUGMENTATION 9: TUZ & BİBER GÜRÜLTÜSÜ (SALT & PEPPER NOISE)
# ─────────────────────────────────────────────────────────────────
#
#  Tuz & Biber gürültüsü, rastgele pikselleri tamamen beyaz (tuz)
#  veya tamamen siyah (biber) yapar:
#
#    I(x,y) = 255   olasılık p/2 ile  (tuz - beyaz nokta)
#    I(x,y) = 0     olasılık p/2 ile  (biber - siyah nokta)
#    I(x,y) = I(x,y) geri kalanında   (değişmez)
#
#  Bu tür gürültü, termal sensörlerde "ölü piksel" (dead pixel)
#  veya "sıcak piksel" (hot pixel) sorunlarını simüle eder.
#  Ucuz termal kameralarda bu sorun sıkça görülür.
# ─────────────────────────────────────────────────────────────────

def augment_salt_pepper(img, bboxes, img_w, img_h):
    """Tuz & Biber gürültüsü ekler (ölü/sıcak piksel simülasyonu)."""
    gray = ensure_gray(img).copy()

    # Gürültü oranı: %0.5 - %3 arası
    noise_ratio = random.uniform(0.005, 0.03)
    n_pixels = int(img_w * img_h * noise_ratio)

    # Tuz (beyaz pikseller)
    salt_y = np.random.randint(0, img_h, n_pixels)
    salt_x = np.random.randint(0, img_w, n_pixels)
    gray[salt_y, salt_x] = 255

    # Biber (siyah pikseller)
    pepper_y = np.random.randint(0, img_h, n_pixels)
    pepper_x = np.random.randint(0, img_w, n_pixels)
    gray[pepper_y, pepper_x] = 0

    new_img = gray_to_3ch(gray)
    return new_img, bboxes


# ─────────────────────────────────────────────────────────────────
#  AUGMENTATION 10: GAUSS BULANIKLAŞTIRMA / TERMAL BLUR
# ─────────────────────────────────────────────────────────────────
#
#  Termal Bulanıklaştırma (Thermal Blur):
#    I_blur(x,y) = I(x,y) * G(x,y; σ)
#
#  Burada * konvolüsyon operatörü, G Gauss çekirdeğidir:
#    G(x,y; σ) = (1 / 2πσ²) · exp(-(x² + y²) / 2σ²)
#
#  Bu işlem, termal sensörün odak (focus) hatalarını taklit eder.
#  Termal kameralarda odak kayması (focus drift) sıcaklık
#  değişimlerinden dolayı yaygın bir sorundur.
#
#  σ büyüdükçe bulanıklık artar:
#    σ = 1-2  → hafif bulanıklık (normal çalışma)
#    σ = 3-5  → orta bulanıklık (odak kayması)
#    σ = 5+   → ağır bulanıklık (sensör arızası)
# ─────────────────────────────────────────────────────────────────

def augment_thermal_blur(img, bboxes, img_w, img_h):
    """
    Gauss bulanıklaştırma (Termal Blur) uygular.
    Termal sensörün odak hatalarını simüle eder.
    """
    # PyTorch ile Gauss bulanıklaştırma
    sigma = random.uniform(1.0, 3.0)
    # Çekirdek boyutu: sigma'nın 6 katı (tek sayı olmalı)
    kernel_size = int(6 * sigma + 1) | 1

    tensor = TF.to_tensor(img)
    blurred = TF.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size],
                                sigma=[sigma, sigma])
    new_img = (blurred.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Bulanıklaştırma geometriyi değiştirmez → bbox'lar aynı kalır
    return new_img, bboxes


# ─────────────────────────────────────────────────────────────────
#  TÜM AUGMENTATION'LARI BİR LİSTEDE TOPLA
# ─────────────────────────────────────────────────────────────────

AUGMENTATIONS = [
    ("hflip",       "Yatay Cevirme (Horizontal Flip)",   augment_horizontal_flip),
    ("vflip",       "Dikey Cevirme (Vertical Flip)",     augment_vertical_flip),
    ("rotate",      "Rastgele Dondurme (Rotation)",      augment_rotation),
    ("crop",        "Rastgele Kirpma (Random Crop)",      augment_random_crop),
    ("rrc",         "Olcekleme (RandomResizedCrop)",      augment_random_resized_crop),
    ("affine",      "Afin Donusumu (Affine)",             augment_affine),
    ("elastic",     "Elastik Distorsiyon (Elastic)",      augment_elastic),
    ("clahe",       "CLAHE Kontrast Iyilestirme",         augment_clahe),
    ("gnoise",      "Gauss Gurultusu (Gaussian Noise)",   augment_gaussian_noise),
    ("sp_noise",    "Tuz & Biber Gurultusu (Salt&Pepper)",augment_salt_pepper),
    ("tblur",       "Termal Blur (Gaussian Blur)",        augment_thermal_blur),
]


# ─────────────────────────────────────────────────────────────────
#  B.1) TRAIN SETİNE AUGMENTATION UYGULA
# ─────────────────────────────────────────────────────────────────

print(f"\n[B.1] Train setine {AUGMENT_MULTIPLIER}x augmentation uygulanacak "
      f"(görüntü başına {AUGMENT_MULTIPLIER - 1} rastgele teknik)...")
print(f"      Orijinal train: {len(new_data['train']['images'])} görüntü")
print(f"      Beklenen ek görüntü: ~{len(new_data['train']['images']) * (AUGMENT_MULTIPLIER - 1):,}")
print(f"      Beklenen toplam:     ~{len(new_data['train']['images']) * AUGMENT_MULTIPLIER:,}\n")

# Mevcut train verisini yükle
train_json_path = OUTPUT_DIR / "train" / "_annotations.coco.json"
with open(train_json_path, "r") as f:
    train_coco = json.load(f)

# Mevcut image ve annotation bilgileri
existing_images = list(train_coco["images"])
existing_annotations = list(train_coco["annotations"])

# Annotation'ları image_id'ye göre grupla
anns_by_img = defaultdict(list)
for ann in existing_annotations:
    anns_by_img[ann["image_id"]].append(ann)

# Sayaçlar
aug_img_id = max(img["id"] for img in existing_images) + 1
aug_ann_id = max(ann["id"] for ann in existing_annotations) + 1

new_images = []
new_annotations = []

total_images = len(existing_images)
processed = 0

for img_info in existing_images:
    img_id = img_info["id"]
    img_w = img_info["width"]
    img_h = img_info["height"]
    img_path = OUTPUT_DIR / "train" / "images" / img_info["file_name"]

    # Görüntüyü oku
    img_np = cv2.imread(str(img_path))
    if img_np is None:
        continue

    # Bu görüntünün bbox'larını al: [cat_id, x, y, w, h]
    bboxes = []
    for ann in anns_by_img.get(img_id, []):
        bboxes.append([ann["category_id"]] + list(ann["bbox"]))

    if not bboxes:
        continue

    # Seçilen çarpana göre rastgele augmentation teknikleri belirle
    selected_augs = random.sample(AUGMENTATIONS, AUGMENT_MULTIPLIER - 1)

    # Seçilen augmentation'ları uygula
    for aug_key, aug_name, aug_func in selected_augs:
        try:
            aug_img, aug_bboxes = aug_func(img_np.copy(), deepcopy(bboxes), img_w, img_h)
        except Exception:
            continue

        if not aug_bboxes:
            continue  # Tüm bbox'lar kaybolmuşsa bu augmentation'ı atla

        # Augmented görüntüyü kaydet
        ext = os.path.splitext(img_info["file_name"])[1]
        new_filename = f"aug_{aug_key}_{aug_img_id:06d}{ext}"
        save_path = OUTPUT_DIR / "train" / "images" / new_filename
        cv2.imwrite(str(save_path), aug_img)

        # COCO image kaydı
        new_images.append({
            "id": aug_img_id,
            "file_name": new_filename,
            "width": img_w,
            "height": img_h,
        })

        # COCO annotation kayıtları
        for cat_id, bx, by, bw, bh in aug_bboxes:
            new_annotations.append({
                "id": aug_ann_id,
                "image_id": aug_img_id,
                "category_id": cat_id,
                "bbox": [bx, by, bw, bh],
                "area": round(bw * bh, 2),
                "segmentation": [],
                "iscrowd": 0,
            })
            aug_ann_id += 1

        aug_img_id += 1

    processed += 1
    if processed % 200 == 0 or processed == total_images:
        pct = processed / total_images * 100
        total_aug = len(new_images)
        print(f"    [{processed:>5}/{total_images}] ({pct:5.1f}%)  "
              f"Oluşturulan augmented görüntü: {total_aug:,}")


# ─────────────────────────────────────────────────────────────────
#  B.2) AUGMENTED VERİYİ MEVCUT TRAIN VERİSİYLE BİRLEŞTİR
# ─────────────────────────────────────────────────────────────────

print(f"\n[B.2] Orijinal ve augmented veri birleştiriliyor...")

final_images = existing_images + new_images
final_annotations = existing_annotations + new_annotations

final_coco = {
    "images": final_images,
    "annotations": final_annotations,
    "categories": TARGET_CATEGORIES,
}

with open(train_json_path, "w") as f:
    json.dump(final_coco, f)

print(f"    Orijinal train:   {len(existing_images):>7} görüntü, "
      f"{len(existing_annotations):>8} annotation")
print(f"    Augmented eklenen: {len(new_images):>7} görüntü, "
      f"{len(new_annotations):>8} annotation")
print(f"    TOPLAM train:     {len(final_images):>7} görüntü, "
      f"{len(final_annotations):>8} annotation")


# ─────────────────────────────────────────────────────────────────
#  B.3) SINIF BAZINDA DAĞILIM RAPORU
# ─────────────────────────────────────────────────────────────────

print(f"\n[B.3] Sınıf dağılımı raporu:")

for split in SPLITS:
    json_path = OUTPUT_DIR / split / "_annotations.coco.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    n_img = len(data["images"])
    n_ann = len(data["annotations"])

    cls_counts = {}
    for ann in data["annotations"]:
        name = CLASS_NAMES[ann["category_id"]]
        cls_counts[name] = cls_counts.get(name, 0) + 1

    print(f"\n    {split.upper():>5}: {n_img:>7} görüntü, {n_ann:>8} annotation")
    for cls_name in ["Person", "Car", "OtherVehicle"]:
        count = cls_counts.get(cls_name, 0)
        pct = count / n_ann * 100 if n_ann > 0 else 0
        print(f"          {cls_name:<15} {count:>8} ({pct:5.1f}%)")


# ─────────────────────────────────────────────────────────────────
#  SONUÇ
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  İŞLEM TAMAMLANDI!")
print("=" * 70)
print(f"\n  Çıktı klasörü: {OUTPUT_DIR}")
print(f"\n  Yapı:")
print(f"    {OUTPUT_DIR.name}/")
print(f"      train/  → Orijinal + Augmented eğitim verisi")
print(f"      val/    → Doğrulama verisi (augmentation YOK)")
print(f"      test/   → Test verisi (augmentation YOK)")
print(f"\n  Uygulanan augmentation'lar:")
for key, name, _ in AUGMENTATIONS:
    print(f"    • {name}")
print("\n" + "=" * 70)
