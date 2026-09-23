from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file
from openpyxl import Workbook

from scripts.infer_single import DEFAULT_SCORE_FIELDS, infer_image


ROOT = Path(__file__).resolve().parent

app = Flask(__name__)


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


@app.get("/")
def index():
    return render_template("index.html", score_fields=DEFAULT_SCORE_FIELDS)


@app.post("/api/recognize")
def recognize():
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
    try:
        result = normalize_result(infer_image(upload_path, ROOT))
    except Exception as exc:
        result = {
            "student_id": "",
            "name": "",
            "scores": empty_scores(),
            "source": "error",
            "message": f"模型识别失败：{type(exc).__name__}: {exc}",
        }
    finally:
        upload_path.unlink(missing_ok=True)
    result["original_name"] = original_name
    return jsonify({"ok": True, "result": result})


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
    app.run(host="127.0.0.1", port=7860, debug=False)
