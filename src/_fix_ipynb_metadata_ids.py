import json
import uuid
from pathlib import Path


def ensure_metadata_ids(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    changed = False

    for cell in cells:
        meta = cell.get("metadata")
        if meta is None or not isinstance(meta, dict):
            meta = {}
            cell["metadata"] = meta
            changed = True

        # Preserve existing per-cell id if present.
        existing_cell_id = cell.get("id")
        if "id" not in meta:
            if isinstance(existing_cell_id, str) and existing_cell_id:
                meta["id"] = existing_cell_id
            else:
                meta["id"] = uuid.uuid4().hex[:8]
            changed = True

        # Ensure metadata.language exists (some editors move it around)
        if "language" not in meta:
            # Try to infer
            if cell.get("cell_type") == "markdown":
                meta["language"] = "markdown"
            else:
                meta["language"] = "python"
            changed = True

    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"Updated: {path}")
    else:
        print(f"No changes: {path}")


if __name__ == "__main__":
    for p in [
        Path(r"d:\SAYZEK\Sayzek_models\DETR_training.ipynb"),
        Path(r"d:\SAYZEK\Sayzek_models\FasterRCNN_training.ipynb"),
    ]:
        ensure_metadata_ids(p)
