"""
Builds the final dataset/ folder (YOLO detection format) from two raw sources:

  1. A Roboflow export zip (YOLOv8 format) dropped into raw_data/
     -> already has real bounding boxes + train/valid/test split.
  2. The GitHub "Door Classification" (DoorDetect) zip, e.g. from Downloads
     -> classification folders only (no bboxes), so we auto-generate a
        full-frame bounding box per image and use its existing split.
        "Semi" (half-open) images are excluded from training but copied
        into predictions/semi_holdout/ for failure-analysis examples.

Usage:
    python scripts/prepare_dataset.py --roboflow raw_data/closeddoors.zip \
                                       --github "C:/Users/BISWAJEET/Downloads/Door Classification-...zip"

Either source can be omitted; the script works with just one.
Re-run any time — it rebuilds dataset/ from scratch (idempotent).
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
HOLDOUT_DIR = ROOT / "predictions" / "semi_holdout"

# Unified class scheme for this project
CLASS_NAMES = ["door_open", "door_closed"]
OPEN_IDX, CLOSED_IDX = 0, 1


def reset_dataset_dir():
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            d = DATASET_DIR / kind / split
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                f.unlink()


def classify_name(name: str):
    """Map a free-text class/folder name to OPEN_IDX / CLOSED_IDX / None (drop)."""
    n = name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if "semi" in n:
        return None
    if "open" in n:
        return OPEN_IDX
    if "close" in n or "closed" in n:
        return CLOSED_IDX
    return None


# ---------------------------------------------------------------------------
# Source 1: Roboflow YOLOv8 export
# ---------------------------------------------------------------------------
def ingest_roboflow(zip_path: Path, work_dir: Path):
    print(f"[roboflow] extracting {zip_path.name} ...")
    extract_dir = work_dir / "roboflow"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    data_yaml = next(extract_dir.rglob("data.yaml"), None)
    if data_yaml is None:
        print("  ! no data.yaml found, skipping roboflow source")
        return 0

    import yaml
    cfg = yaml.safe_load(data_yaml.read_text())
    src_names = cfg["names"]
    if isinstance(src_names, dict):
        src_names = [src_names[i] for i in range(len(src_names))]
    remap = {i: classify_name(n) for i, n in enumerate(src_names)}
    print(f"  source classes: {src_names}")
    print(f"  remap -> {remap}")

    split_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    count = 0
    for src_split, dst_split in split_map.items():
        img_dir = extract_dir / src_split / "images"
        lbl_dir = extract_dir / src_split / "labels"
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            new_lines = []
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                orig_cls = int(parts[0])
                new_cls = remap.get(orig_cls)
                if new_cls is None:
                    continue  # drop boxes for classes we don't want (e.g. junk/semi)
                new_lines.append(" ".join([str(new_cls)] + parts[1:]))
            if not new_lines:
                continue  # skip images with no usable boxes after remap
            out_stem = f"rf_{img_path.stem}"
            shutil.copy(img_path, DATASET_DIR / "images" / dst_split / f"{out_stem}{img_path.suffix}")
            (DATASET_DIR / "labels" / dst_split / f"{out_stem}.txt").write_text("\n".join(new_lines) + "\n")
            count += 1
    print(f"  ingested {count} images from roboflow")
    return count


# ---------------------------------------------------------------------------
# Source 2: GitHub DoorDetect classification zip -> full-frame boxes
# ---------------------------------------------------------------------------
def ingest_github(zip_path: Path, work_dir: Path):
    print(f"[github] scanning {zip_path.name} ...")
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    semi_count = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if "/Cropped/RGB/" in n and n.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"  found {len(names)} Cropped/RGB images")
        for entry in names:
            # path shape: .../Cropped/RGB/<split>/<Class>/<file>.png
            parts = entry.split("/")
            split_name, class_name, fname = parts[-3], parts[-2], parts[-1]
            split = {"train": "train", "val": "val", "test": "test"}.get(split_name.lower())
            if split is None:
                continue
            cls_idx = classify_name(class_name)
            out_stem = f"gh_{Path(fname).stem}"

            if cls_idx is None:
                # Semi / ambiguous -> not used for training, saved for failure analysis
                with zf.open(entry) as src, open(HOLDOUT_DIR / f"{out_stem}{Path(fname).suffix}", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                semi_count += 1
                continue

            img_out = DATASET_DIR / "images" / split / f"{out_stem}{Path(fname).suffix}"
            with zf.open(entry) as src, open(img_out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            # full-frame bounding box: class x_center y_center width height (normalized)
            (DATASET_DIR / "labels" / split / f"{out_stem}.txt").write_text(f"{cls_idx} 0.5 0.5 1.0 1.0\n")
            count += 1
    print(f"  ingested {count} images from github ({semi_count} 'Semi' images set aside in predictions/semi_holdout/)")
    return count


def write_data_yaml():
    content = (
        f"path: {DATASET_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (DATASET_DIR / "data.yaml").write_text(content)
    print(f"\nwrote {DATASET_DIR / 'data.yaml'}")


def summarize():
    print("\n--- dataset summary ---")
    for split in ("train", "val", "test"):
        n = len(list((DATASET_DIR / "images" / split).iterdir()))
        print(f"  {split}: {n} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roboflow", type=Path, default=None, help="path to Roboflow YOLOv8 export zip")
    ap.add_argument("--github", type=Path, default=None, help="path to GitHub 'Door Classification' zip")
    args = ap.parse_args()

    if not args.roboflow and not args.github:
        auto_rf = list((ROOT / "raw_data").glob("*.zip"))
        if auto_rf:
            args.roboflow = auto_rf[0]
            print(f"auto-detected roboflow zip: {args.roboflow}")
        else:
            print("No sources given and none found in raw_data/. Pass --roboflow and/or --github.")
            sys.exit(1)

    reset_dataset_dir()
    work_dir = ROOT / "raw_data" / "_extracted"
    work_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    if args.roboflow:
        total += ingest_roboflow(args.roboflow, work_dir)
    if args.github:
        total += ingest_github(args.github, work_dir)

    if total == 0:
        print("No images ingested — check the zip paths/structure.")
        sys.exit(1)

    write_data_yaml()
    summarize()


if __name__ == "__main__":
    main()
