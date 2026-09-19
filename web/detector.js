// Browser-side bee detection with ONNX Runtime Web.
// Mirrors beecounter/detector.py: letterbox to 640, run YOLOv8, per-class NMS,
// merge worker/drone/queen into "bee", drop varroa.
const MODEL_URL = "model/honey_bee_detector.onnx";
const SIZE = 640;
const CONF = 0.25;
const IOU = 0.7;
const BEE_CLASSES = [0, 1, 2]; // worker, drone, queen; 3 = varroa
const PAD = 114;

let sessionPromise = null;

export function loadModel(onProgress) {
  if (!sessionPromise) sessionPromise = fetchWithProgress(MODEL_URL, onProgress).then((buf) =>
    ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }));
  return sessionPromise;
}

async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`model download failed: ${resp.status}`);
  const total = Number(resp.headers.get("content-length")) || 0;
  const reader = resp.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (onProgress) onProgress(got, total);
  }
  const out = new Uint8Array(got); let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out.buffer;
}

// Same letterbox as ultralytics with auto=False: scale long side to 640, pad the rest with gray.
// Resampling is done by hand to reproduce OpenCV's INTER_LINEAR exactly (2x2 bilinear at
// half-pixel centers, no antialiasing), which is what the model was trained with. The
// browser's own canvas downscaling blurs more and loses up to 20% of the bees.
function letterbox(bitmap) {
  const { width: w, height: h } = bitmap;
  const r = Math.min(SIZE / w, SIZE / h);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const left = Math.round((SIZE - nw) / 2 - 0.1), top = Math.round((SIZE - nh) / 2 - 0.1);
  const n = SIZE * SIZE;
  const input = new Float32Array(3 * n).fill(PAD / 255);
  const src = new OffscreenCanvas(w, h).getContext("2d", { willReadFrequently: true });
  src.drawImage(bitmap, 0, 0);
  const s = src.getImageData(0, 0, w, h).data;
  const sx = w / nw, sy = h / nh;
  for (let y = 0; y < nh; y++) {
    let fy = Math.max(0, (y + 0.5) * sy - 0.5);
    let y0 = Math.floor(fy), wy = fy - y0;
    if (y0 >= h - 1) { y0 = h - 1; wy = 0; }
    const y1 = y0 + (wy > 0 ? 1 : 0);
    for (let x = 0; x < nw; x++) {
      let fx = Math.max(0, (x + 0.5) * sx - 0.5);
      let x0 = Math.floor(fx), wx = fx - x0;
      if (x0 >= w - 1) { x0 = w - 1; wx = 0; }
      const x1 = x0 + (wx > 0 ? 1 : 0);
      const i00 = (y0 * w + x0) * 4, i01 = (y0 * w + x1) * 4, i10 = (y1 * w + x0) * 4, i11 = (y1 * w + x1) * 4;
      const o = (top + y) * SIZE + left + x;
      for (let c = 0; c < 3; c++) {
        const v = (1 - wy) * ((1 - wx) * s[i00 + c] + wx * s[i01 + c]) + wy * ((1 - wx) * s[i10 + c] + wx * s[i11 + c]);
        input[c * n + o] = Math.round(v) / 255;
      }
    }
  }
  return { input, r, left, top };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1), x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

// Per-class greedy NMS, like ultralytics (agnostic=False).
function nms(cands) {
  cands.sort((a, b) => b.conf - a.conf);
  const keep = [];
  for (const c of cands) {
    let ok = true;
    for (const k of keep) if (k.cls === c.cls && iou(k, c) > IOU) { ok = false; break; }
    if (ok) keep.push(c);
  }
  return keep;
}

export async function detect(bitmap) {
  const session = await loadModel();
  const { input, r, left, top } = letterbox(bitmap);
  const t0 = performance.now();
  const out = await session.run({ images: new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]) });
  const ms = Math.round(performance.now() - t0);
  const o = out[session.outputNames[0]];         // [1, 4 + nc, 8400]
  const [, rows, n] = o.dims; const nc = rows - 4; const d = o.data;
  const cands = [];
  for (let i = 0; i < n; i++) {
    let cls = 0, conf = 0;
    for (let c = 0; c < nc; c++) { const s = d[(4 + c) * n + i]; if (s > conf) { conf = s; cls = c; } }
    if (conf < CONF) continue;
    const cx = d[i], cy = d[n + i], w = d[2 * n + i], h = d[3 * n + i];
    cands.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, conf, cls });
  }
  const detections = nms(cands).filter((c) => BEE_CLASSES.includes(c.cls)).map((c) => ({
    x: ((c.x1 + c.x2) / 2 - left) / r,
    y: ((c.y1 + c.y2) / 2 - top) / r,
    w: (c.x2 - c.x1) / r,
    h: (c.y2 - c.y1) / r,
    conf: Math.round(c.conf * 1000) / 1000,
  }));
  return { width: bitmap.width, height: bitmap.height, detections, count: detections.length, inference_ms: ms };
}
