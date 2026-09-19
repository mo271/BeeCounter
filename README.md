# BeeCounter

A small web tool for beekeepers: upload photos taken during a hive inspection
and get a bee count per photo and for the whole hive. Every detected bee is
marked with a dot. Detections can be corrected by clicking, and the counts
exported.

Live site: https://firsching.ch/BeeCounter/

Everything runs in the browser. Photos never leave the device, and the site
is plain static files served by GitHub Pages.

Detection uses the pretrained [Apiarist YOLOv8s honey-bee detector](https://huggingface.co/maryammeda/apiarist-honey-bee-detector)
(Apache-2.0), exported to ONNX and run with ONNX Runtime Web. The model also
distinguishes workers, drones, queens and varroa mites, but for now everything
is merged into a single bee count. See [models/README.md](models/README.md).

## Run the site

Any static file server works, for example:

    cd docs && python3 -m http.server 8000

Then open http://127.0.0.1:8000/. The first visit downloads the 43 MB model,
after that the browser caches it. A photo takes about a second on a laptop.

## Layout

- `docs/` is the whole site: `index.html`, `app.js` (UI), `detector.js`
  (preprocessing, ONNX inference, NMS) and `model/honey_bee_detector.onnx`.
  It is called `docs` only because GitHub Pages can publish either the repo
  root or a folder with exactly that name straight from the `main` branch,
  without a build workflow. There is no documentation in it. The empty
  `.nojekyll` file tells Pages to serve the files as they are.
- `beecounter/` is a Python reference implementation of the same detector,
  plus a small FastAPI wrapper. It is used by the tests and for experiments;
  it produces identical results to the browser.
- `models/` holds the original PyTorch weights; `scripts/export_onnx.py`
  regenerates the ONNX file in `docs/model/`.
- `data/samples/` holds test photos; `tests/` checks the detector against them.
- `scripts/` also holds the experiments that picked the model and input size.

## Development

    python -m venv .venv            # or: mamba create -p ./.venv python=3.12
    .venv/bin/pip install -r requirements.txt
    .venv/bin/pytest -q

## Known limitations

- Blurry photos or bees seen through a cover foil are mostly missed.
- Corrections live in the browser only. Reloading the page loses them, so
  export before closing.
- Inference is single-threaded WebAssembly. Old phones will be slow.

## License and credits

BeeCounter's code is under the Apache License 2.0, see [LICENSE](LICENSE).

The detector is the [Apiarist honey-bee detector](https://huggingface.co/maryammeda/apiarist-honey-bee-detector)
by maryammeda (Build Small Hackathon entry, 2026), published under Apache-2.0
and trained on the [hendricks_ricky/bee-project](https://universe.roboflow.com/hendricks_ricky-hotmail-de/bee-project)
dataset from Roboflow Universe (CC BY 4.0). It is a fine-tune of Ultralytics'
YOLOv8s, which is AGPL-3.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for what that means.
