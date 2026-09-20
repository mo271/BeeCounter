# Third-party notices

BeeCounter's own code is licensed under the Apache License 2.0 (see `LICENSE`).
The repository also redistributes and depends on third-party work:

## Detector weights

`models/honey_bee_detector.pt` and its ONNX export `docs/model/honey_bee_detector.onnx`
are the **Apiarist honey-bee detector** by maryammeda, published at
https://huggingface.co/maryammeda/apiarist-honey-bee-detector under the
Apache License 2.0. The author asks that users credit:

- This work: Apiarist (Build Small Hackathon entry, 2026)
- Training data: hendricks_ricky/bee-project, Roboflow Universe

Provenance of those weights, as stated by the author:

- Fine-tuned from Ultralytics' `yolov8s.pt` using the `ultralytics` package.
  Ultralytics distributes both under the AGPL-3.0 and states that models
  trained with its code are covered by the AGPL-3.0 as well
  (https://www.ultralytics.com/license). The Hugging Face author published
  the fine-tuned weights as Apache-2.0. Whether trained weights are a
  derivative work of the training code is legally unsettled; this repository
  reproduces the author's stated license and, being fully open source,
  satisfies the AGPL's source-availability requirement in any case. If you
  redistribute the weights in a closed-source product, seek your own advice.
- Training dataset: hendricks_ricky/bee-project on Roboflow Universe,
  CC BY 4.0 by the original dataset authors. The dataset itself is not
  included here.

## Software dependencies

- ONNX Runtime Web (MIT), loaded from a CDN by the site; not redistributed.
- libheif-js (LGPL-3.0, a WebAssembly build of libheif), loaded from a CDN
  only when a HEIC photo is added; not redistributed.
- ultralytics (AGPL-3.0), PyTorch (BSD-3), ONNX (Apache-2.0), FastAPI (MIT),
  Pillow (MIT-CMU): Python development dependencies used by the reference
  implementation and tests. They are not part of the published site.

## Assets

- `docs/assets/github-mark.svg` is GitHub's logo, used to link to the source
  repository as permitted by GitHub's logo guidelines.
- `docs/examples/csiro_honeybees_in_hive.jpg` (and its thumbnail) is
  "European honeybees (Apis mellifera) in a hive" by Nick Pitsas, CSIRO,
  CSIRO ScienceImage 7077, licensed CC BY 3.0, via Wikimedia Commons:
  https://commons.wikimedia.org/wiki/File:CSIRO_ScienceImage_7077_European_honeybees_Apis_mellifera_in_a_hive.jpg
  It was re-encoded as JPEG with metadata stripped; no other changes.
- The photos in `data/samples/` were taken by the repository owner.
