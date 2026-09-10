"""
Runs 3 (or more) YOLO training experiments with different hyperparameters,
evaluates each on the test set, times inference latency, and writes a
comparison table to experiments/results.csv (+ prints it).

Each experiment varies ONE axis from the baseline so you can point to a
specific hyperparameter and say what it did:
  exp1_baseline        - default yolov8n, imgsz 640, 50 epochs
  exp2_augmentation    - stronger augmentation (mosaic/hsv/flip) on top of baseline
  exp3_capacity_lr     - yolov8s (bigger) + different learning rate

Edit EXPERIMENTS below to change/add runs.

Usage:
    python scripts/run_experiments.py
"""
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
EXPERIMENTS_DIR = ROOT / "experiments"

EXPERIMENTS = [
    {
        "name": "exp1_baseline",
        "model": "yolov8n.pt",
        "train_args": dict(epochs=50, imgsz=640, batch=16, lr0=0.01, optimizer="auto"),
    },
    {
        "name": "exp2_augmentation",
        "model": "yolov8n.pt",
        "train_args": dict(
            epochs=50, imgsz=640, batch=16, lr0=0.01, optimizer="auto",
            mosaic=1.0, mixup=0.15, hsv_h=0.02, hsv_s=0.8, hsv_v=0.5, fliplr=0.5, degrees=10.0,
        ),
    },
    {
        "name": "exp3_capacity_lr",
        "model": "yolov8s.pt",
        "train_args": dict(epochs=50, imgsz=640, batch=16, lr0=0.005, optimizer="AdamW"),
    },
]


def measure_latency(model: YOLO, imgsz: int, n: int = 50) -> float:
    """Average inference latency in ms/image on CPU using a dummy tensor warm-up + real test images."""
    test_dir = ROOT / "dataset" / "images" / "test"
    imgs = list(test_dir.glob("*"))[:n] or list(test_dir.glob("*"))
    if not imgs:
        return float("nan")
    # warm-up
    for img in imgs[:3]:
        model.predict(source=str(img), imgsz=imgsz, verbose=False)
    start = time.perf_counter()
    for img in imgs:
        model.predict(source=str(img), imgsz=imgsz, verbose=False)
    elapsed = time.perf_counter() - start
    return (elapsed / len(imgs)) * 1000  # ms/image


def run_one(exp: dict) -> dict:
    print(f"\n{'='*60}\nRunning {exp['name']}\n{'='*60}")
    model = YOLO(exp["model"])
    model.train(
        data=str(DATA_YAML),
        project=str(EXPERIMENTS_DIR),
        name=exp["name"],
        exist_ok=True,
        **exp["train_args"],
    )

    metrics = model.val(data=str(DATA_YAML), split="test")
    latency_ms = measure_latency(model, exp["train_args"].get("imgsz", 640))

    precision = float(metrics.box.mp)       # mean precision across classes
    recall = float(metrics.box.mr)          # mean recall across classes
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "experiment": exp["name"],
        "model": exp["model"],
        "hyperparams": str(exp["train_args"]),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "latency_ms_per_image": round(latency_ms, 2),
        "weights_path": str(Path(EXPERIMENTS_DIR / exp["name"] / "weights" / "best.pt")),
    }


def main():
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    results = [run_one(exp) for exp in EXPERIMENTS]

    df = pd.DataFrame(results)
    out_csv = EXPERIMENTS_DIR / "results.csv"
    df.to_csv(out_csv, index=False)

    print("\n\n===== EXPERIMENT COMPARISON =====")
    print(df[["experiment", "precision", "recall", "f1", "mAP50", "mAP50_95", "latency_ms_per_image"]]
          .to_string(index=False))
    print(f"\nSaved full results to {out_csv}")

    best = df.loc[df["mAP50_95"].idxmax()]
    print(f"\nBest by mAP50-95: {best['experiment']} -> {best['weights_path']}")


if __name__ == "__main__":
    main()
