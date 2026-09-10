"""
Step 1: exports a chosen experiment's best.pt to ONNX and verifies it with onnxruntime.
Step 2: runs that model on 5 test images and saves annotated predictions to predictions/.

Usage:
    python scripts/export_and_predict.py --experiment exp1_baseline
"""
import argparse
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
PREDICTIONS_DIR = ROOT / "predictions"


def export_onnx(weights_path: Path) -> Path:
    model = YOLO(str(weights_path))
    onnx_path = model.export(format="onnx", dynamic=False, simplify=True)
    print(f"Exported ONNX model to {onnx_path}")
    return Path(onnx_path)


def verify_onnx(onnx_path: Path, sample_image: Path, imgsz: int = 640):
    import cv2
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    img = cv2.imread(str(sample_image))
    img = cv2.resize(img, (imgsz, imgsz))
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})
    print(f"ONNX sanity check OK — output shape: {outputs[0].shape}")


def save_example_predictions(weights_path: Path, n: int = 5):
    model = YOLO(str(weights_path))
    test_dir = ROOT / "dataset" / "images" / "test"
    all_imgs = list(test_dir.glob("*"))
    sample = random.sample(all_imgs, min(n, len(all_imgs)))
    out_dir = PREDICTIONS_DIR / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sample:
        results = model.predict(source=str(img_path), save=False, verbose=False)
        results[0].save(filename=str(out_dir / f"pred_{img_path.name}"))
    print(f"Saved {len(sample)} example predictions to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="experiment name, e.g. exp1_baseline")
    args = ap.parse_args()

    weights_path = EXPERIMENTS_DIR / args.experiment / "weights" / "best.pt"
    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path} — run run_experiments.py first.")

    onnx_path = export_onnx(weights_path)

    test_dir = ROOT / "dataset" / "images" / "test"
    sample_image = next(test_dir.glob("*"), None)
    if sample_image:
        verify_onnx(onnx_path, sample_image)

    save_example_predictions(weights_path)


if __name__ == "__main__":
    main()
