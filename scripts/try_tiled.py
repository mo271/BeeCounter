"""Tiled inference: cut the full-res image into overlapping tiles, run the
detector per tile at its native scale, merge with NMS. argv: out_dir tile_px imgsz"""
import sys, time, pathlib, torch
from torchvision.ops import nms
from PIL import Image, ImageOps, ImageDraw
from ultralytics import YOLO

COLORS = {"Worker": (255, 220, 0), "Drone": (0, 140, 255), "Queen": (255, 0, 60), "Varroa": (200, 0, 200)}
PER_CLASS_CONF = {"Worker": 0.25, "Drone": 0.55, "Varroa": 0.30, "Queen": 0.15}
model = YOLO("models/honey_bee_detector.pt")
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
tile = int(sys.argv[2]); imgsz = int(sys.argz[3]) if False else int(sys.argv[3]); ov = tile // 5

def detect(im):
    boxes, scores, classes = [], [], []
    W, H = im.size
    xs = list(range(0, max(W - tile, 0) + 1, tile - ov)) or [0]
    ys = list(range(0, max(H - tile, 0) + 1, tile - ov)) or [0]
    if xs[-1] + tile < W: xs.append(W - tile)
    if ys[-1] + tile < H: ys.append(H - tile)
    for y in ys:
        for x in xs:
            crop = im.crop((x, y, x + tile, y + tile))
            r = model(crop, conf=0.10, imgsz=imgsz, device="cpu", verbose=False)[0]
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append([x1 + x, y1 + y, x2 + x, y2 + y]); scores.append(float(b.conf)); classes.append(int(b.cls))
    if not boxes: return []
    B, S = torch.tensor(boxes), torch.tensor(scores)
    keep = nms(B, S, 0.45)
    return [(B[i].tolist(), S[i].item(), classes[i]) for i in keep]

for p in sorted(pathlib.Path("data/samples").glob("*.[jJ][pP][gG]")):
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    t0 = time.time(); dets = detect(im); dt = time.time() - t0
    counts = {k: 0 for k in COLORS}; draw = ImageDraw.Draw(im); r = max(6, im.width // 250)
    for (x1, y1, x2, y2), conf, ci in dets:
        cls = model.names[ci].split()[0]
        if conf < PER_CLASS_CONF[cls]: continue
        counts[cls] += 1; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; rr = r * (2 if cls == "Queen" else 1)
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=COLORS[cls], outline=(0, 0, 0), width=2)
    im.thumbnail((1600, 1600)); im.save(out / f"{p.stem}_dots.jpg", quality=85)
    print(f"{p.name:32s} {dt:5.1f}s  {counts}")
