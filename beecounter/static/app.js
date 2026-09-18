// BeeCounter frontend: upload -> detect -> dots -> corrections -> hive total.
const DOT_COLOR = "#ffdc00";

const state = { images: [], active: null };
const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

// ---------- upload ----------
$("file").addEventListener("change", (e) => addFiles(e.target.files));
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); addFiles(e.dataTransfer.files); });

async function addFiles(files) {
  for (const file of files) {
    if (!file.type.startsWith("image/")) continue;
    const img = { id: crypto.randomUUID(), name: file.name, file, url: URL.createObjectURL(file),
                  bitmap: null, result: null, dets: null, status: "detecting…", error: null };
    state.images.push(img);
    renderList();
    detect(img);
  }
}

async function detect(img) {
  try {
    const fd = new FormData(); fd.append("file", img.file);
    const [bitmap, resp] = await Promise.all([
      createImageBitmap(img.file, { imageOrientation: "from-image" }),
      fetch("/api/detect", { method: "POST", body: fd }),
    ]);
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    img.bitmap = bitmap;
    img.result = await resp.json();
    img.dets = img.result.detections.map((d) => ({ ...d, source: "model" }));
    img.status = `${img.result.inference_ms} ms`;
  } catch (e) {
    img.error = e.message; img.status = "failed";
  }
  renderList(); renderTotal();
  if (!state.active) select(img.id); else if (state.active === img.id) renderViewer();
}

// ---------- counts ----------
const total = () => state.images.reduce((n, i) => n + (i.dets ? i.dets.length : 0), 0);
function renderTotal() { $("total").textContent = total(); }

// ---------- image list ----------
function renderList() {
  $("images").innerHTML = state.images.map((img) => `
    <div class="thumb ${img.id === state.active ? "active" : ""}" data-id="${img.id}">
      <img src="${img.url}" alt="">
      <div><div class="name" title="${img.name}">${img.name}</div>
           <div class="n">${img.dets ? img.dets.length + " bees" : ""}</div>
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

function renderViewer() {
  const img = activeImg();
  const ready = img && img.bitmap;
  $("viewer").hidden = !ready;
  $("empty").hidden = !!ready;
  if (!ready) return;
  $("viewer-name").textContent = img.name;
  canvas.width = img.bitmap.width; canvas.height = img.bitmap.height;
  draw();
}

function draw() {
  const img = activeImg();
  if (!img || !img.bitmap) return;
  ctx.drawImage(img.bitmap, 0, 0);
  $("viewer-count").textContent = img.dets.length;
  if (!$("show-dots").checked) return;
  const r = dotRadius(img);
  for (const d of img.dets) {
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
  if (!img || !img.bitmap) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  const hitR = dotRadius(img) * 2.5;
  let best = null, bestD = hitR;
  img.dets.forEach((d, i) => { const dist = Math.hypot(d.x - x, d.y - y); if (dist < bestD) { best = i; bestD = dist; } });
  if (best === null) img.dets.push({ x, y, w: 0, h: 0, conf: 1, source: "user" });
  else img.dets.splice(best, 1);
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
    filename: i.name, width: i.result.width, height: i.result.height, count: i.dets.length, detections: i.dets }));
  download("beecount.json", JSON.stringify({ images, total: total() }, null, 2), "application/json");
});
$("export-csv").addEventListener("click", () => {
  const rows = [["filename", "bees"]];
  for (const i of state.images.filter((i) => i.dets)) rows.push([i.name, i.dets.length]);
  rows.push(["TOTAL", total()]);
  download("beecount.csv", rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n"), "text/csv");
});
$("clear").addEventListener("click", () => { if (confirm("Remove all images?")) { state.images = []; state.active = null; renderList(); renderTotal(); renderViewer(); } });

window.addEventListener("resize", draw);
renderTotal(); renderViewer();
