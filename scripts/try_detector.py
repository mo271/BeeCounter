"""Run the pretrained YOLOv8s honey-bee detector on the sample images and draw
colored dots per detection. Output goes to the directory given as argv[1]."""
import sys, time, pathlib
from PIL import Image, ImageOps, ImageDraw
from ultralytics import YOLO

COLORS = {"Worker": (255, 220, 0), "Drone": (0, 140, 255), "Queen": (255, 0, 60), "Varroa": (200, 0, 200)}
PER_CLASS_CONF = {"Worker": 0.25, "Drone": 0.55, "Varroa": 0.30, "Queen": 0.15}

model = YOLO("models/honey_bee_detector.pt")
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 1280

for p in sorted(pathlib.Path("data/samples").glob("*.[jJ][pP][gG]")):
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    t0 = time.time()
    res = model(im, conf=0.10, imgsz=imgsz, device="cpu", verbose=False)[0]
    dt = time.time() - t0
    counts = {k: 0 for k in COLORS}
    draw = ImageDraw.Draw(im)
    r = max(6, im.width // 250)
    for box in res.boxes:
        cls = model.names[int(box.cls.item())].split()[0]; conf = float(box.conf.item())
        if conf < PER_CLASS_CONF.get(cls, 0.25):
            continue
        counts[cls] += 1
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rr = r * (2 if cls == "Queen" else 1)
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=COLORS[cls], outline=(0, 0, 0), width=2)
    im.thumbnail((1600, 1600))
    im.save(out / f"{p.stem}_dots.jpg", quality=85)
    print(f"{p.name:32s} {dt:5.1f}s  {counts}")
