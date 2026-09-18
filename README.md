# BeeCounter

A small web tool for beekeepers: upload photos taken during a hive inspection
and get a bee count per photo and for the whole hive. Every detected bee is
marked with a dot. Detections can be corrected by clicking, and the counts
exported.

Detection uses the pretrained [Apiarist YOLOv8s honey-bee detector](https://huggingface.co/maryammeda/apiarist-honey-bee-detector)
(Apache-2.0). It runs on CPU in well under a second per photo. The model also
distinguishes workers, drones, queens and varroa mites, but for now everything
is merged into a single bee count. See [models/README.md](models/README.md).

## Run it

    python -m venv .venv            # or: mamba create -p ./.venv python=3.12
    .venv/bin/pip install -r requirements.txt
    .venv/bin/uvicorn beecounter.app:app --reload

Then open http://127.0.0.1:8000/.

## Layout

- `beecounter/detector.py` loads the model and turns an image into detections.
- `beecounter/app.py` is the FastAPI app: `POST /api/detect` takes an image
  upload and returns detections plus counts as JSON. It also serves the page.
- `beecounter/static/` is the frontend: plain HTML, CSS and JavaScript.
- `data/samples/` holds test photos; `tests/` checks the detector against them.
- `scripts/` holds the experiments that picked the model and its input size.

## Tests

    .venv/bin/pytest -q

## Known limitations

- Blurry photos or bees seen through a cover foil are mostly missed.
- Corrections live in the browser only. Reloading the page loses them, so
  export before closing.
