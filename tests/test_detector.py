from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from beecounter.app import app
from beecounter.detector import Detector

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture(scope="module")
def detector():
    return Detector()


def test_no_bees_in_portrait(detector):
    assert detector.detect_bytes((SAMPLES / "IMG_8283.JPG").read_bytes()).count == 0


def test_single_bee(detector):
    r = detector.detect_bytes((SAMPLES / "20200509_150918.jpg").read_bytes())
    assert r.count == 1
    # EXIF orientation 6: the oriented image is portrait
    assert r.height > r.width


def test_dense_cluster(detector):
    assert detector.detect_bytes((SAMPLES / "20190729_201604.jpg").read_bytes()).count >= 80


def test_api_roundtrip():
    with TestClient(app) as client:
        with open(SAMPLES / "PXL_20240602_083422333.jpg", "rb") as f:
            resp = client.post("/api/detect", files={"file": ("frame.jpg", f, "image/jpeg")})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 80
        assert body["count"] == len(body["detections"])
        assert {"x", "y", "w", "h", "conf"} <= set(body["detections"][0])
        assert client.post("/api/detect", files={"file": ("x.txt", b"nope", "text/plain")}).status_code == 400
