# Models

`honey_bee_detector.pt` is the Apiarist YOLOv8s detector (Apache-2.0) from
https://huggingface.co/maryammeda/apiarist-honey-bee-detector
Classes: Worker Bee, Drone Bee, Queen Bee, Varroa Mite.

Re-download with:

    curl -L -o models/honey_bee_detector.pt \
      https://huggingface.co/maryammeda/apiarist-honey-bee-detector/resolve/main/honey_bee_detector.pt

Best setting found on our samples: whole image, imgsz=640 (see scripts/try_detector.py).
