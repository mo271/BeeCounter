"""FastAPI app: serves the single-page frontend and the detection endpoint."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from .detector import Detector

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 30 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector()
    yield


app = FastAPI(title="BeeCounter", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><text y="26" font-size="26">\U0001F41D</text></svg>'
    return Response(svg, media_type="image/svg+xml")


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


app.mount("/static", StaticFiles(directory=STATIC), name="static")
