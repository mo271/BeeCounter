"""Export the PyTorch detector to ONNX for the browser (docs/model/)."""
import shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
out = YOLO(ROOT / "models" / "honey_bee_detector.pt").export(format="onnx", imgsz=640, simplify=True, opset=17)
dest = ROOT / "docs" / "model" / "honey_bee_detector.onnx"
shutil.move(out, dest)
print("wrote", dest, round(dest.stat().st_size / 1e6, 1), "MB")
