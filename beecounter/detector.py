"""Bee detection with the pretrained Apiarist YOLOv8s model.

The model was trained on downscaled frame photos, so whole-image inference at
640px works far better than higher resolutions or tiling (see scripts/).

The model distinguishes worker, drone, queen and varroa mite. We only count
bees: worker, drone and queen detections are merged, mites are ignored.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()  # lets Pillow open HEIC/HEIF photos from iPhones

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "honey_bee_detector.pt"
IMGSZ = 640
BEE_CLASSES = {"Worker Bee", "Drone Bee", "Queen Bee"}
CONF = 0.25
IOU = 0.7  # NMS threshold (ultralytics default)
MAX_DET = 1000  # ultralytics caps at 300 by default; a full frame can hold more bees


@dataclass
class Detection:
    x: float  # center, pixels in the EXIF-oriented image
    y: float
    w: float
    h: float
    conf: float


@dataclass
class Result:
    width: int
    height: int
    detections: list[Detection]
    count: int
    inference_ms: int

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "detections": [asdict(d) for d in self.detections],
            "count": self.count,
            "inference_ms": self.inference_ms,
        }


class Detector:
    def __init__(self, model_path: Path = MODEL_PATH, imgsz: int = IMGSZ):
        from ultralytics import YOLO  # slow import, keep it local

        self.model = YOLO(str(model_path))
        self.imgsz = imgsz
        self.bee_ids = {i for i, n in self.model.names.items() if n in BEE_CLASSES}

    def detect(self, image: Image.Image) -> Result:
        image = ImageOps.exif_transpose(image).convert("RGB")
        t0 = time.perf_counter()
        # rect=False pads to a 640x640 square exactly like the ONNX export used in the browser,
        # so this reference and docs/detector.js produce identical results. classes= keeps only
        # bee classes before NMS and agnostic=True suppresses across them, so a bee scored as
        # both worker and drone is counted once.
        res = self.model(image, conf=CONF, iou=IOU, max_det=MAX_DET, imgsz=self.imgsz, rect=False,
                         classes=sorted(self.bee_ids), agnostic_nms=True, device="cpu", verbose=False)[0]
        ms = int((time.perf_counter() - t0) * 1000)
        dets: list[Detection] = []
        for box in res.boxes:
            if int(box.cls.item()) not in self.bee_ids:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append(Detection((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, round(float(box.conf.item()), 3)))
        return Result(image.width, image.height, dets, len(dets), ms)

    def detect_bytes(self, data: bytes) -> Result:
        return self.detect(Image.open(io.BytesIO(data)))
