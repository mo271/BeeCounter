"""Small FastAPI wrapper around the reference detector.

The real frontend lives in web/ and runs the model in the browser; this API is
kept for tests and experiments.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from .detector import Detector

MAX_UPLOAD = 30 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector()
    yield


app = FastAPI(title="BeeCounter reference API", lifespan=lifespan)


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image larger than 30 MB")
    try:
        result = app.state.detector.detect_bytes(data)
    except UnidentifiedImageError:
        raise HTTPException(400, "Not a readable image")
    return {"filename": file.filename, **result.to_dict()}
