import json
from pathlib import Path

NOTEBOOK_PATH = Path(r"d:\SAYZEK\Sayzek_models\DETR_training.ipynb")

CATEGORY_CELL_MARKER = "coco_train = COCO(TRAIN_ANN)"
DATASET_CELL_MARKER = "class DetrCocoDataset"


def _mk(lines: list[str]) -> list[str]:
    # Jupyter allows either raw lines or lines ending with \n; prefer \n for stability.
    out: list[str] = []
    for line in lines:
        if line.endswith("\n"):
            out.append(line)
        else:
            out.append(line + "\n")
    return out


def main() -> None:
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    changed = False

    category_cell_new = _mk(
        [
            "# COCO objeleri ve kategori eşlemeleri",
            "coco_train = COCO(TRAIN_ANN)",
            "coco_val = COCO(VAL_ANN)",
            "",
            "cat_ids = coco_train.getCatIds()",
            "cats = coco_train.loadCats(cat_ids)",
            "cats = sorted(cats, key=lambda x: x['id'])",
            "id2name = {c['id']: c['name'] for c in cats}",
            "cat_id_list = [c['id'] for c in cats]",
            "",
            "# DETR model head'i için sınıf sayısı (labels: 0..num_classes-1)",
            "num_classes = len(cat_id_list)",
            "cat_id_to_idx = {cid: i for i, cid in enumerate(cat_id_list)}",
            "idx_to_cat_id = {i: cid for cid, i in cat_id_to_idx.items()}",
            "",
            "print('num_classes:', num_classes)",
            "print('first categories:', list(id2name.items())[:10])",
        ]
    )

    dataset_cell_new = _mk(
        [
            "# Dataset wrapper (COCO) -> DETR format",
            "from torchvision.datasets import CocoDetection",
            "",
            "image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')",
            "",
            "class DetrCocoDataset(CocoDetection):",
            "    def __init__(self, img_folder, ann_file, processor: DetrImageProcessor, train: bool, cat_id_to_idx: dict[int, int]):",
            "        super().__init__(img_folder, ann_file)",
            "        self.processor = processor",
            "        self.train = train",
            "        self.cat_id_to_idx = cat_id_to_idx",
            "",
            "    def __getitem__(self, idx):",
            "        img, target = super().__getitem__(idx)",
            "        image_id = self.ids[idx]",
            "        # DETR eğitimi için category_id değerlerini 0..num_classes-1 aralığına map'liyoruz",
            "        mapped_target = []",
            "        for ann in target:",
            "            ann2 = dict(ann)",
            "            if 'category_id' in ann2:",
            "                ann2['category_id'] = int(self.cat_id_to_idx[int(ann2['category_id'])])",
            "            mapped_target.append(ann2)",
            "",
            "        encoding = self.processor(",
            "            images=img,",
            "            annotations={'image_id': image_id, 'annotations': mapped_target},",
            "            return_tensors='pt'",
            "        )",
            "        pixel_values = encoding['pixel_values'].squeeze(0)",
            "        labels = encoding['labels'][0]",
            "        return pixel_values, labels",
            "",
            "def detr_collate_fn(batch):",
            "    pixel_values = [item[0] for item in batch]",
            "    labels = [item[1] for item in batch]",
            "    encoding = image_processor.pad(pixel_values, return_tensors='pt')",
            "    return {",
            "        'pixel_values': encoding['pixel_values'],",
            "        'pixel_mask': encoding['pixel_mask'],",
            "        'labels': labels",
            "    }",
            "",
            "train_ds = DetrCocoDataset(TRAIN_IMG_DIR, TRAIN_ANN, image_processor, train=True, cat_id_to_idx=cat_id_to_idx)",
            "val_ds = DetrCocoDataset(VAL_IMG_DIR, VAL_ANN, image_processor, train=False, cat_id_to_idx=cat_id_to_idx)",
            "",
            "BATCH_SIZE = 4",
            "NUM_WORKERS = 2",
            "",
            "train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=detr_collate_fn)",
            "val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=detr_collate_fn)",
            "",
            "print('train batches:', len(train_loader), 'val batches:', len(val_loader))",
        ]
    )

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or []
        src_text = "".join(src)

        if CATEGORY_CELL_MARKER in src_text:
            if "cat_id_to_idx" not in src_text:
                cell["source"] = category_cell_new
                changed = True

        if DATASET_CELL_MARKER in src_text:
            if "cat_id_to_idx" not in src_text or "mapped_target" not in src_text:
                cell["source"] = dataset_cell_new
                changed = True

    if not changed:
        print("No changes needed.")
        return

    NOTEBOOK_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"Patched: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
