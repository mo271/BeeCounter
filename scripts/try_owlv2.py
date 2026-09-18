"""Zero-shot OWLv2 with the text query 'a bee' on one image, for comparison."""
import sys, time, torch
from PIL import Image, ImageOps, ImageDraw
from transformers import Owlv2Processor, Owlv2ForObjectDetection
proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval()
for path in sys.argv[2:]:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB"); im.thumbnail((1600, 1600))
    t0 = time.time()
    inputs = proc(text=[["a bee"]], images=im, return_tensors="pt")
    with torch.no_grad(): outs = model(**inputs)
    res = proc.post_process_grounded_object_detection(outs, threshold=0.2, target_sizes=torch.tensor([im.size[::-1]]))[0]
    draw = ImageDraw.Draw(im)
    for b in res["boxes"]:
        x1, y1, x2, y2 = b.tolist(); cx, cy = (x1+x2)/2, (y1+y2)/2
        draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill=(0,255,120), outline=(0,0,0), width=2)
    name = path.split('/')[-1].rsplit('.',1)[0]
    im.save(f"{sys.argv[1]}/{name}_owl.jpg", quality=85)
    print(f"{name}: {len(res['boxes'])} boxes in {time.time()-t0:.1f}s")
