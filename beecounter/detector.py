"""Bee detection with the pretrained Apiarist YOLOv8s model.

The model was trained on downscaled frame photos, so whole-image inference at
640px works far better than higher resolutions or tiling (see scripts/).
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "honey_bee_detector.pt"
IMGSZ = 640

# Model class name -> our short class id
CLASS_MAP = {"Worker Bee": "worker", "Drone Bee": "drone", "Queen Bee": "queen", "Varroa Mite": "varroa"}
CLASSES = ["worker", "drone", "queen", "varroa"]

# Per-class confidence thresholds, from the model card (tuned on real frames).
CONF = {"worker": 0.25, "drone": 0.55, "queen": 0.15, "varroa": 0.30}
MIN_CONF = min(CONF.values())
# A varroa mite is a few mm; a "mite" box wider than this fraction of the image is a false positive.
MAX_VARROA_FRAC = 0.05


@dataclass
class Detection:
    x: float  # center, pixels in the EXIF-oriented image
    y: float
    w: float
    h: float
    cls: str
    conf: float


@dataclass
class Result:
    width: int
    height: int
    detections: list[Detection]
    counts: dict[str, int]
    inference_ms: int

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "detections": [asdict(d) for d in self.detections],
            "counts": self.counts,
            "inference_ms": self.inference_ms,
        }


class Detector:
    def __init__(self, model_path: Path = MODEL_PATH, imgsz: int = IMGSZ):
        from ultralytics import YOLO  # slow import, keep it local

        self.model = YOLO(str(model_path))
        self.imgsz = imgsz
        self.names = {i: CLASS_MAP.get(n, n) for i, n in self.model.names.items()}

    def detect(self, image: Image.Image) -> Result:
        image = ImageOps.exif_transpose(image).convert("RGB")
        t0 = time.perf_counter()
        res = self.model(image, conf=MIN_CONF, imgsz=self.imgsz, device="cpu", verbose=False)[0]
        ms = int((time.perf_counter() - t0) * 1000)
        dets: list[Detection] = []
        for box in res.boxes:
            cls = self.names[int(box.cls.item())]
            conf = float(box.conf.item())
            if conf < CONF.get(cls, MIN_CONF):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if cls == "varroa" and max(x2 - x1, y2 - y1) > MAX_VARROA_FRAC * max(image.size):
                continue
            dets.append(Detection((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, cls, round(conf, 3)))
        counts = {c: 0 for c in CLASSES}
        for d in dets:
            counts[d.cls] += 1
        return Result(image.width, image.height, dets, counts, ms)

    def detect_bytes(self, data: bytes) -> Result:
        return self.detect(Image.open(io.BytesIO(data)))
