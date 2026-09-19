// BeeCounter frontend, fully client-side: upload -> detect in the browser -> dots -> corrections -> hive total.
import { loadModel, detect as detectBitmap, DEFAULT_CONF } from "./detector.js";

const DOT_COLOR = "#ffdc00";
const state = { images: [], active: null };
const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

// ---------- model ----------
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const status = $("model-status");
const modelReady = loadModel((got, total) => {
  const mb = (n) => (n / 1e6).toFixed(0);
  status.textContent = total ? `Loading model… ${mb(got)} / ${mb(total)} MB` : `Loading model… ${mb(got)} MB`;
}).then(() => { status.textContent = "Model ready."; status.classList.add("ready"); })
  .catch((e) => { status.textContent = `Could not load the model: ${e.message}`; status.classList.add("error"); throw e; });

// ---------- upload ----------
$("file").addEventListener("change", (e) => addFiles(e.target.files));
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); addFiles(e.dataTransfer.files); });

// ---------- image decoding ----------
const isHeic = (file) => /\.(heic|heif)$/i.test(file.name) || /image\/hei[cf]/.test(file.type);
const isImage = (file) => file.type.startsWith("image/") || isHeic(file);

// Chrome and Firefox cannot decode HEIC (iPhone photos); Safari can. Try the browser first,
// then fall back to libheif compiled to WebAssembly, loaded only when first needed.
let libheifPromise = null;
async function decodeHeic(file) {
  if (!libheifPromise) libheifPromise = import("https://cdn.jsdelivr.net/npm/libheif-js@1.23.2/libheif-wasm/libheif-bundle.mjs")
    .then(async (m) => { const lib = typeof m.default === "function" ? m.default() : m.default; if (lib.ready) await lib.ready; return lib; });
  const libheif = await libheifPromise;
  if (typeof libheif.HeifDecoder !== "function") throw new Error("HEIC decoder unavailable");
  const images = new libheif.HeifDecoder().decode(new Uint8Array(await file.arrayBuffer()));
  if (!images.length) throw new Error("no image found in HEIC file");
  const image = images[0];
  const width = image.get_width(), height = image.get_height();
  const imageData = new ImageData(width, height);
  await new Promise((resolve, reject) => image.display(imageData, (ok) => (ok ? resolve() : reject(new Error("HEIC decode failed")))));
  return createImageBitmap(imageData);
}

async function loadBitmap(file) {
  try {
    return await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch (e) {
    if (isHeic(file)) return decodeHeic(file);
    throw new Error("could not read this image");
  }
}

// small preview drawn from the decoded bitmap, so it also works for formats the <img> tag cannot show
async function makeThumb(bitmap) {
  const w = 128, h = Math.max(1, Math.round(bitmap.height * w / bitmap.width));
  const c = new OffscreenCanvas(w, h); c.getContext("2d").drawImage(bitmap, 0, 0, w, h);
  return URL.createObjectURL(await c.convertToBlob({ type: "image/jpeg", quality: 0.8 }));
}

let queue = Promise.resolve(); // run detections one after another; the model is single-threaded anyway
async function addFiles(files) {
  for (const file of files) {
    if (!isImage(file)) continue;
    const img = { id: crypto.randomUUID(), name: file.name, file, thumb: null,
                  bitmap: null, result: null, dets: null, threshold: DEFAULT_CONF, status: "waiting…", error: null };
    state.images.push(img);
    renderList();
    queue = queue.then(() => detect(img));
  }
}

async function detect(img) {
  try {
    img.status = "reading…"; renderList();
    img.bitmap = await loadBitmap(img.file);
    img.thumb = await makeThumb(img.bitmap);
    img.status = "detecting…"; renderList();
    await modelReady;
    img.result = await detectBitmap(img.bitmap);
    img.dets = img.result.detections.map((d) => ({ ...d, source: "model", removed: false }));
    img.status = `${img.result.inference_ms} ms`;
  } catch (e) {
    img.error = e.message; img.status = "failed";
  }
  renderList(); renderTotal();
  if (!state.active) select(img.id); else if (state.active === img.id) renderViewer();
}

// ---------- counts ----------
// A dot is shown if the user added it, or the model is at least `threshold` sure and the user did not remove it.
const visible = (img) => img.dets.filter((d) => !d.removed && (d.source === "user" || d.conf >= img.threshold));
const total = () => state.images.reduce((n, i) => n + (i.dets ? visible(i).length : 0), 0);
function renderTotal() { $("total").textContent = total(); }

// ---------- image list ----------
function renderList() {
  $("images").innerHTML = state.images.map((img) => `
    <div class="thumb ${img.id === state.active ? "active" : ""}" data-id="${img.id}">
      ${img.thumb ? `<img src="${img.thumb}" alt="">` : `<span class="thumb-placeholder"></span>`}
      <div><div class="name" title="${img.name}">${img.name}</div>
           <div class="n">${img.dets ? visible(img).length + " bees" : ""}</div>
           <div class="status">${img.error || img.status}</div></div>
      <button data-remove="${img.id}" title="Remove">✕</button>
    </div>`).join("");
}
$("images").addEventListener("click", (e) => {
  const rm = e.target.dataset.remove;
  if (rm) { remove(rm); return; }
  const t = e.target.closest(".thumb");
  if (t) select(t.dataset.id);
});
function remove(id) {
  state.images = state.images.filter((i) => i.id !== id);
  if (state.active === id) state.active = state.images[0]?.id || null;
  renderList(); renderTotal(); renderViewer();
}
function select(id) { state.active = id; renderList(); renderViewer(); }

// ---------- viewer ----------
const activeImg = () => state.images.find((i) => i.id === state.active);
$("show-dots").addEventListener("change", draw);
$("dot-size").addEventListener("input", draw);
$("conf").addEventListener("input", (e) => {
  const img = activeImg(); if (!img) return;
  img.threshold = Number(e.target.value);
  $("conf-val").value = img.threshold.toFixed(2);
  draw(); renderList(); renderTotal();
});

function renderViewer() {
  const img = activeImg();
  const ready = img && img.bitmap && img.dets;
  $("viewer").hidden = !ready;
  $("empty").hidden = !!ready;
  if (!ready) return;
  $("viewer-name").textContent = img.name;
  $("conf").value = img.threshold; $("conf-val").value = img.threshold.toFixed(2);
  canvas.width = img.bitmap.width; canvas.height = img.bitmap.height;
  draw();
}

function draw() {
  const img = activeImg();
  if (!img || !img.bitmap || !img.dets) return;
  ctx.drawImage(img.bitmap, 0, 0);
  const shown = visible(img);
  $("viewer-count").textContent = shown.length;
  if (!$("show-dots").checked) return;
  const r = dotRadius(img);
  for (const d of shown) {
    ctx.beginPath(); ctx.arc(d.x, d.y, r, 0, Math.PI * 2);
    ctx.fillStyle = DOT_COLOR; ctx.fill();
    ctx.lineWidth = Math.max(1, r / 3); ctx.strokeStyle = d.source === "user" ? "#fff" : "#000"; ctx.stroke();
  }
}
// dot radius in image pixels so it looks the same size on screen regardless of image resolution
function dotRadius(img) {
  const scale = img.bitmap.width / canvas.getBoundingClientRect().width;
  return Number($("dot-size").value) * scale;
}

// click: remove the nearest dot, or add a bee
canvas.addEventListener("click", (e) => {
  const img = activeImg();
  if (!img || !img.bitmap || !img.dets) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  const hitR = dotRadius(img) * 2.5;
  let best = null, bestD = hitR;
  for (const d of visible(img)) { const dist = Math.hypot(d.x - x, d.y - y); if (dist < bestD) { best = d; bestD = dist; } }
  if (best === null) img.dets.push({ x, y, w: 0, h: 0, conf: 1, source: "user", removed: false });
  else if (best.source === "user") img.dets.splice(img.dets.indexOf(best), 1);
  else best.removed = true;
  draw(); renderList(); renderTotal();
});

// ---------- export ----------
function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click();
  URL.revokeObjectURL(a.href);
}
$("export-json").addEventListener("click", () => {
  const images = state.images.filter((i) => i.dets).map((i) => ({
    filename: i.name, width: i.result.width, height: i.result.height, threshold: i.threshold,
    count: visible(i).length, detections: visible(i).map(({ removed, ...d }) => d) }));
  download("beecount.json", JSON.stringify({ images, total: total() }, null, 2), "application/json");
});
$("export-csv").addEventListener("click", () => {
  const rows = [["filename", "bees"]];
  for (const i of state.images.filter((i) => i.dets)) rows.push([i.name, visible(i).length]);
  rows.push(["TOTAL", total()]);
  download("beecount.csv", rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n"), "text/csv");
});
$("clear").addEventListener("click", () => { if (confirm("Remove all images?")) { state.images = []; state.active = null; renderList(); renderTotal(); renderViewer(); } });

window.addEventListener("resize", draw);
renderTotal(); renderViewer();
