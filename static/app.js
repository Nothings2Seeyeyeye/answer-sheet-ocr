const camera = document.getElementById("camera");
const preview = document.getElementById("preview");
const snapshot = document.getElementById("snapshot");
const emptyPreview = document.getElementById("emptyPreview");
const statusEl = document.getElementById("status");
const folderInput = document.getElementById("folderInput");
const fileInput = document.getElementById("fileInput");
const fileQueue = document.getElementById("fileQueue");
const scoreBody = document.getElementById("scoreBody");
const recordHead = document.getElementById("recordHead");
const recordBody = document.getElementById("recordBody");
const studentIdInput = document.getElementById("studentId");
const studentNameInput = document.getElementById("studentName");
const totalScoreEl = document.getElementById("totalScore");

let stream = null;
let currentBlob = null;
let currentName = "capture.jpg";
let selectedRecordId = null;
let queuedFiles = [];
let records = loadRecords();

function loadRecords() {
  try {
    const value = JSON.parse(localStorage.getItem("ocr-student-records") || "[]");
    return Array.isArray(value) ? value : [];
  } catch (_error) {
    return [];
  }
}

function saveRecords() {
  localStorage.setItem("ocr-student-records", JSON.stringify(records));
}

function setStatus(text, kind = "warning") {
  statusEl.className = `status ${kind}`;
  statusEl.textContent = text;
}

function showPreviewUrl(url) {
  camera.style.display = "none";
  preview.src = url;
  preview.style.display = "block";
  emptyPreview.style.display = "none";
}

function showCamera() {
  preview.style.display = "none";
  camera.style.display = "block";
  emptyPreview.style.display = "none";
}

function renderScoreRows(scores = null) {
  const values = scores || window.SCORE_FIELDS.map(([field, label]) => ({ field, label, value: "", confidence: 0 }));
  scoreBody.innerHTML = "";
  values.forEach((item) => {
    const row = document.createElement("tr");
    row.dataset.field = item.field;
    row.dataset.label = item.label;
    row.innerHTML = `<td>${item.label}</td><td><input class="score-input" type="text" value="${item.value || ""}" /></td><td>${item.confidence ? Number(item.confidence).toFixed(3) : "--"}</td>`;
    scoreBody.appendChild(row);
  });
  updateTotal();
}

function collectScores() {
  return Array.from(scoreBody.querySelectorAll("tr")).map((row) => ({
    field: row.dataset.field,
    label: row.dataset.label,
    value: row.querySelector("input").value.trim(),
  }));
}

function updateTotal() {
  const scores = collectScores();
  const totalRow = scores.find((item) => item.field === "total");
  if (totalRow && totalRow.value !== "") {
    totalScoreEl.textContent = totalRow.value;
    return;
  }
  let sum = 0;
  let ok = false;
  scores.filter((item) => item.field !== "total").forEach((item) => {
    const number = Number(item.value);
    if (!Number.isNaN(number) && item.value !== "") {
      sum += number;
      ok = true;
    }
  });
  totalScoreEl.textContent = ok ? String(Number(sum.toFixed(2))) : "--";
}

function fillResult(result) {
  studentIdInput.value = result.student_id || "";
  studentNameInput.value = result.name || "";
  renderScoreRows(result.scores);
  setStatus(result.message || "识别完成，请核对后确认记录。", result.source === "error" ? "warning" : "ok");
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
  camera.srcObject = stream;
  showCamera();
  setStatus("摄像头已打开", "ok");
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  camera.style.display = "none";
  if (!preview.src) emptyPreview.style.display = "grid";
  setStatus("摄像头已关闭", "warning");
}

async function recognizeBlob(blob, fileName) {
  currentBlob = blob;
  currentName = fileName;
  showPreviewUrl(URL.createObjectURL(blob));
  const form = new FormData();
  form.append("image", blob, fileName);
  setStatus("正在识别...", "busy");
  // 1. 提交识别任务，拿到 task_id
  const submitResp = await fetch("/api/recognize", { method: "POST", body: form });
  const submitData = await submitResp.json();
  if (!submitData.ok) throw new Error(submitData.error || "识别失败");
  const taskId = submitData.task_id;
  // 2. 轮询结果（间隔 1.5s，最多约 5 分钟）
  for (let attempt = 0; attempt < 200; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    const resResp = await fetch(`/api/result/${taskId}`);
    const resData = await resResp.json();
    if (resData.ok && resData.status === "done") {
      fillResult(resData.result);
      return;
    }
    if (!resData.ok) throw new Error(resData.error || "识别失败");
  }
  throw new Error("识别超时，请重试");
}

async function captureAndRecognize() {
  if (!stream) {
    setStatus("请先打开摄像头", "warning");
    return;
  }
  snapshot.width = camera.videoWidth || 1280;
  snapshot.height = camera.videoHeight || 720;
  const ctx = snapshot.getContext("2d");
  ctx.drawImage(camera, 0, 0, snapshot.width, snapshot.height);
  snapshot.toBlob(async (blob) => {
    await recognizeBlob(blob, `capture_${Date.now()}.jpg`);
  }, "image/jpeg", 0.95);
}

function enqueueFiles(files) {
  queuedFiles = Array.from(files).filter((file) => file.type.startsWith("image/"));
  fileQueue.innerHTML = "";
  queuedFiles.forEach((file, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = file.webkitRelativePath || file.name;
    fileQueue.appendChild(option);
  });
  if (queuedFiles.length > 0) {
    loadSelectedFile(false);
    setStatus(`已导入 ${queuedFiles.length} 张图片`, "ok");
  }
}

async function loadSelectedFile(autoRecognize = true) {
  const file = queuedFiles[Number(fileQueue.value || 0)];
  if (!file) return;
  currentBlob = file;
  currentName = file.name;
  showPreviewUrl(URL.createObjectURL(file));
  if (autoRecognize) await recognizeBlob(file, file.name);
}

async function recognizeCurrent() {
  if (!currentBlob) {
    setStatus("请先拍照或选择图片", "warning");
    return;
  }
  await recognizeBlob(currentBlob, currentName);
}

function confirmRecord() {
  const record = {
    id: `R${Date.now()}`,
    student_id: studentIdInput.value.trim(),
    name: studentNameInput.value.trim(),
    scores: collectScores(),
    created_at: new Date().toLocaleString(),
  };
  records.push(record);
  saveRecords();
  renderRecords(records);
  setStatus("已确认并记录（保存在当前浏览器）", "ok");
}

function recordColumns() {
  return ["学号", "姓名", ...window.SCORE_FIELDS.map(([_field, label]) => label)];
}

function renderRecords(records) {
  const columns = recordColumns();
  recordHead.innerHTML = `<th></th>${columns.map((col) => `<th>${col}</th>`).join("")}`;
  recordBody.innerHTML = "";
  records.forEach((record, index) => {
    const scoreMap = {};
    (record.scores || []).forEach((item) => { scoreMap[item.label] = item.value; });
    const row = document.createElement("tr");
    row.dataset.id = record.id;
    if (record.id === selectedRecordId) row.classList.add("selected");
    const cells = [record.student_id || "", record.name || "", ...window.SCORE_FIELDS.map(([_field, label]) => scoreMap[label] || "")];
    row.innerHTML = `<td>${index + 1}</td>${cells.map((value) => `<td>${value}</td>`).join("")}`;
    row.addEventListener("click", () => { selectedRecordId = record.id; renderRecords(records); });
    recordBody.appendChild(row);
  });
}

function deleteSelected() {
  if (!selectedRecordId) return;
  records = records.filter((record) => record.id !== selectedRecordId);
  saveRecords();
  selectedRecordId = null;
  renderRecords(records);
}

function clearAll() {
  records = [];
  saveRecords();
  selectedRecordId = null;
  renderRecords(records);
}

async function exportExcel() {
  const response = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ records }),
  });
  if (!response.ok) throw new Error("导出失败");
  const blob = await response.blob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `answer_sheet_records_${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "")}.xlsx`;
  link.click();
  URL.revokeObjectURL(link.href);
}

document.getElementById("startCamera").addEventListener("click", startCamera);
document.getElementById("stopCamera").addEventListener("click", stopCamera);
document.getElementById("capturePhoto").addEventListener("click", captureAndRecognize);
document.getElementById("folderButton").addEventListener("click", () => folderInput.click());
document.getElementById("fileButton").addEventListener("click", () => fileInput.click());
folderInput.addEventListener("change", (event) => enqueueFiles(event.target.files));
fileInput.addEventListener("change", (event) => enqueueFiles(event.target.files));
fileQueue.addEventListener("change", () => loadSelectedFile(false));
document.getElementById("recognizeSelected").addEventListener("click", () => loadSelectedFile(true));
document.getElementById("recognizeAgain").addEventListener("click", recognizeCurrent);
document.getElementById("confirmRecord").addEventListener("click", confirmRecord);
document.getElementById("deleteSelected").addEventListener("click", deleteSelected);
document.getElementById("clearAll").addEventListener("click", clearAll);
document.getElementById("exportExcel").addEventListener("click", exportExcel);
scoreBody.addEventListener("input", updateTotal);

renderScoreRows();
renderRecords(records);
