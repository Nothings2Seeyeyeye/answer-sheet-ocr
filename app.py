from __future__ import annotations

import io
import logging
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file
from openpyxl import Workbook

from scripts.infer_single import DEFAULT_SCORE_FIELDS, infer_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

app = Flask(__name__)

# 识别线程池：避免在线 OCR / 模型推理阻塞 Flask 请求
_executor = ThreadPoolExecutor(max_workers=4)
_tasks: dict[str, dict[str, Any]] = {}
_MAX_TASKS = 200  # 内存中最多保留的任务数，超出时清理最旧的


def empty_scores() -> list[dict[str, Any]]:
    return [{"field": field, "label": label, "value": "", "confidence": 0.0} for field, label in DEFAULT_SCORE_FIELDS]


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    scores = []
    score_map = {item.get("field"): item for item in result.get("scores", [])}
    for field, label in DEFAULT_SCORE_FIELDS:
        item = score_map.get(field, {})
        scores.append(
            {
                "field": field,
                "label": label,
                "value": str(item.get("value", "") or ""),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
            }
        )
    return {
        "student_id": str(result.get("student_id", "") or ""),
        "name": str(result.get("name", "") or ""),
        "scores": scores,
        "detections": result.get("detections", []),
        "source": result.get("source", "model"),
        "message": result.get("message", "识别完成，请人工核对后确认记录。"),
    }


def record_to_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {"学号": record.get("student_id", ""), "姓名": record.get("name", "")}
    for item in record.get("scores", []):
        row[item.get("label", item.get("field", ""))] = item.get("value", "")
    return row


def _run_inference(task_id: str, upload_path: Path) -> None:
    """在后台线程执行识别，结果写入 _tasks。"""
    try:
        result = normalize_result(infer_image(upload_path, ROOT))
        _tasks[task_id] = {"status": "done", "result": result}
    except Exception as exc:
        logger.exception("识别任务 %s 失败", task_id)
        _tasks[task_id] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        upload_path.unlink(missing_ok=True)


def _gc_tasks() -> None:
    while len(_tasks) > _MAX_TASKS:
        oldest = next(iter(_tasks))
        _tasks.pop(oldest, None)


@app.get("/")
def index():
    return render_template("index.html", score_fields=DEFAULT_SCORE_FIELDS)


@app.post("/api/recognize")
def recognize():
    """提交识别任务，返回 task_id；前端轮询 /api/result/<task_id> 获取结果。"""
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "未上传图片"}), 400
    file = request.files["image"]
    original_name = file.filename or f"capture_{int(time.time())}.jpg"
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    upload_path = Path(temp_file.name)
    temp_file.close()
    file.save(upload_path)

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "running", "original_name": original_name}
    _gc_tasks()
    _executor.submit(_run_inference, task_id, upload_path)
    logger.info("提交识别任务 %s (%s)", task_id, original_name)
    return jsonify({"ok": True, "task_id": task_id})


@app.get("/api/result/<task_id>")
def result(task_id: str):
    task = _tasks.get(task_id)
    if task is None:
        return jsonify({"ok": False, "error": "任务不存在或已过期"}), 404
    if task["status"] == "running":
        return jsonify({"ok": True, "status": "running"})
    if task["status"] == "error":
        return jsonify({"ok": False, "status": "error", "error": task["error"]})
    result_payload = dict(task["result"])
    result_payload["original_name"] = task.get("original_name", "")
    return jsonify({"ok": True, "status": "done", "result": result_payload})


@app.post("/api/export")
def export_excel():
    payload = request.get_json(silent=True) or {}
    records = payload.get("records", [])
    if not isinstance(records, list):
        return jsonify({"ok": False, "error": "记录格式不正确"}), 400
    rows = [record_to_row(record) for record in records]
    columns = ["学号", "姓名", *[label for _field, label in DEFAULT_SCORE_FIELDS]]
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "成绩汇总"
    sheet.append(columns)
    for row in rows:
        sheet.append([row.get(column, "") for column in columns])
    buffer = io.BytesIO()
    workbook.save(buffer)
    buffer.seek(0)
    filename = f"answer_sheet_records_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    # threaded=True：允许多个后台线程同时处理，配合 ThreadPoolExecutor
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)
