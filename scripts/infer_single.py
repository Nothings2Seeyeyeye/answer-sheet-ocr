from __future__ import annotations

import argparse
import functools
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image, ImageOps

try:
    from scripts.onnx_digit import DigitRecognizer
except ModuleNotFoundError:
    from onnx_digit import DigitRecognizer

try:
    from scripts.text_utils import apply_total_check, clean_chinese_name, clean_digits, clean_number
except ModuleNotFoundError:
    from text_utils import apply_total_check, clean_chinese_name, clean_digits, clean_number

try:
    from scripts.paddleocr_api import recognize_text as _paddleocr_api_recognize
except ModuleNotFoundError:
    try:
        from paddleocr_api import recognize_text as _paddleocr_api_recognize
    except ModuleNotFoundError:
        _paddleocr_api_recognize = None


FIELD_NAMES = [
    "name",
    "student_id",
    "score_1",
    "score_2_1",
    "score_2_2",
    "score_2_3",
    "score_2_4",
    "score_2_5",
    "score_2_6",
    "total",
]

DEFAULT_SCORE_FIELDS = [
    ("score_1", "第1题"),
    ("score_2_1", "第2题(1)"),
    ("score_2_2", "第2题(2)"),
    ("score_2_3", "第2题(3)"),
    ("score_2_4", "第2题(4)"),
    ("score_2_5", "第2题(5)"),
    ("score_2_6", "第2题(6)"),
    ("total", "总分"),
]


def crop_box(image: Image.Image, box: list[float], pad_ratio: float) -> Image.Image:
    x0, y0, x1, y1 = box
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    left = max(0, int(round(x0 - w * pad_ratio)))
    top = max(0, int(round(y0 - h * pad_ratio)))
    right = min(image.width, int(round(x1 + w * pad_ratio)))
    bottom = min(image.height, int(round(y1 + h * pad_ratio)))
    return image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))


@functools.lru_cache(maxsize=4)
def load_onnx_session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


@functools.lru_cache(maxsize=4)
def load_digit_model(model_path: str) -> DigitRecognizer:
    return DigitRecognizer(model_path)


def box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    left_top = np.maximum(one[:2], many[:, :2])
    right_bottom = np.minimum(one[2:], many[:, 2:])
    intersection = np.maximum(right_bottom - left_top, 0).prod(axis=1)
    one_area = max(0.0, float(one[2] - one[0])) * max(0.0, float(one[3] - one[1]))
    many_area = np.maximum(many[:, 2] - many[:, 0], 0) * np.maximum(many[:, 3] - many[:, 1], 0)
    return intersection / np.maximum(one_area + many_area - intersection, 1e-7)


def yolo_detections(image: Image.Image, model_path: Path, image_size: int = 1280, confidence: float = 0.05) -> dict[str, dict[str, Any]]:
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    scale = min(image_size / height, image_size / width)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (image_size - resized_width) / 2, (image_size - resized_height) / 2
    left, top = round(pad_x - 0.1), round(pad_y - 0.1)
    right, bottom = round(pad_x + 0.1), round(pad_y + 0.1)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.transpose(padded.astype(np.float32) / 255.0, (2, 0, 1))[None]
    session = load_onnx_session(str(model_path))
    predictions = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    rows = predictions[0].T if predictions.shape[1] < predictions.shape[2] else predictions[0]
    class_scores = rows[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(rows)), class_ids]
    keep = scores >= confidence
    rows, class_ids, scores = rows[keep], class_ids[keep], scores[keep]
    if len(rows) == 0:
        return {}
    xywh = rows[:, :4]
    boxes = np.column_stack((xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2))
    selected: list[int] = []
    for class_id in np.unique(class_ids):
        indexes = np.where(class_ids == class_id)[0]
        indexes = indexes[np.argsort(scores[indexes])[::-1]]
        while len(indexes):
            current = int(indexes[0])
            selected.append(current)
            if len(indexes) == 1:
                break
            indexes = indexes[1:][box_iou(boxes[current], boxes[indexes[1:]]) <= 0.7]
    output: dict[str, dict[str, Any]] = {}
    for index in selected:
        class_id = int(class_ids[index])
        box = boxes[index].copy()
        box[[0, 2]] = np.clip((box[[0, 2]] - left) / scale, 0, width)
        box[[1, 3]] = np.clip((box[[1, 3]] - top) / scale, 0, height)
        conf = float(scores[index])
        if class_id < 0 or class_id >= len(FIELD_NAMES):
            continue
        field = FIELD_NAMES[class_id]
        current = output.get(field)
        if current is None or float(conf) > float(current["confidence"]):
            output[field] = {
                "field": field,
                "confidence": float(conf),
                "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            }
    return output


def recognize_student(model: DigitRecognizer, crop_path: Path) -> tuple[str, float]:
    """学号识别：复用投影分割 + CNN 分类（与分数识别一致的智能切分）。"""
    text, conf, _details = model.recognize_numeric_roi(crop_path, mode="student")
    return clean_digits(text), float(conf)


def recognize_score(model: DigitRecognizer, crop_path: Path) -> tuple[str, float]:
    text, conf, _details = model.recognize_numeric_roi(crop_path, mode="score")
    return clean_number(text), float(conf)


def run_chinese_ocr(root: Path, image: Image.Image) -> tuple[str, float]:
    """识别姓名字段：依次尝试 在线 API → 本地 PaddleOCR → chineseocr_lite。"""
    text, conf = run_paddleocr_api(image)
    if text:
        return text, conf
    text, conf = run_paddleocr(image)
    if text:
        return text, conf
    return run_chineseocr_lite(root, image)


def run_paddleocr_api(image: Image.Image) -> tuple[str, float]:
    """用 PaddleOCR-VL 在线服务识别姓名（需设置环境变量 PADDLEOCR_API_TOKEN）。"""
    if _paddleocr_api_recognize is None:
        return "", 0.0
    try:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
        text = _paddleocr_api_recognize(buffer.getvalue())
        name = clean_chinese_name(text)
        if 1 <= len(name) <= 6:
            return name, 1.0
    except Exception:
        pass
    return "", 0.0


@functools.lru_cache(maxsize=1)
def load_paddleocr():
    """延迟加载 PaddleOCR 实例（初始化耗时较长，进程内缓存）。"""
    from paddleocr import PaddleOCR  # type: ignore

    return PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)


def run_paddleocr(image: Image.Image) -> tuple[str, float]:
    """用 PaddleOCR 预训练中文模型识别姓名，兼容 PaddleOCR 2.x/3.x 的 ocr() 返回格式。"""
    try:
        engine = load_paddleocr()
        results = engine.ocr(np.asarray(image.convert("RGB")), cls=True)
    except Exception:
        return "", 0.0
    candidates = []
    for line in results or []:
        for item in line or []:
            try:
                _box, (text, score) = item
            except (ValueError, TypeError):
                continue
            name = clean_chinese_name(text)
            if 1 <= len(name) <= 6:
                candidates.append((name, float(score)))
    if not candidates:
        return "", 0.0
    return max(candidates, key=lambda item: (item[1], -abs(len(item[0]) - 3)))


def run_chineseocr_lite(root: Path, image: Image.Image) -> tuple[str, float]:
    """降级后端：内置的 chineseocr_lite 中文 OCR 运行组件。"""
    engine_dir = root / "third_party" / "chineseocr_lite"
    if not engine_dir.exists():
        return "", 0.0
    sys.path.insert(0, str(engine_dir.resolve()))
    try:
        from model import OcrHandle  # type: ignore
    except Exception:
        return "", 0.0
    try:
        handle = load_chinese_handle(str(engine_dir.resolve()))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = Path(tmp.name)
        image.save(temp_path, quality=95)
        results = handle.text_predict(Image.open(temp_path).convert("RGB"), 480)
        temp_path.unlink(missing_ok=True)
    except Exception:
        return "", 0.0
    candidates = []
    for _box, text, score in results:
        name = clean_chinese_name(text)
        if 1 <= len(name) <= 6:
            candidates.append((name, float(score)))
    if not candidates:
        return "", 0.0
    return max(candidates, key=lambda item: (item[1], -abs(len(item[0]) - 3)))


@functools.lru_cache(maxsize=1)
def load_chinese_handle(engine_dir: str):
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    from model import OcrHandle  # type: ignore

    return OcrHandle()


def infer_image(image_path: str | Path, root: str | Path | None = None, cpu: bool = True) -> dict[str, Any]:
    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    image_path = Path(image_path)
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    detections = yolo_detections(image, root_path / "weights" / "field_yolo" / "best.onnx")
    score_model = load_digit_model(str(root_path / "models" / "roi_score_digit_cnn" / "mnist_digit_cnn.onnx"))
    student_model = load_digit_model(str(root_path / "models" / "roi_student_digit_cnn" / "mnist_digit_cnn.onnx"))

    output_scores = []
    student_id = ""
    name = ""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        for field in FIELD_NAMES:
            det = detections.get(field)
            if det is None:
                if field not in {"name", "student_id"}:
                    label = dict(DEFAULT_SCORE_FIELDS).get(field, field)
                    output_scores.append({"field": field, "label": label, "value": "", "confidence": 0.0})
                continue
            pad = 0.10 if field in {"name", "student_id"} else 0.06
            crop = crop_box(image, det["box"], pad)
            crop_path = temp_root / f"{field}.jpg"
            crop.save(crop_path, quality=95)
            if field == "name":
                name, _conf = run_chinese_ocr(root_path, crop)
            elif field == "student_id":
                student_id, _conf = recognize_student(student_model, crop_path)
            else:
                value, conf = recognize_score(score_model, crop_path)
                label = dict(DEFAULT_SCORE_FIELDS).get(field, field)
                output_scores.append({"field": field, "label": label, "value": value, "confidence": conf})

    score_map = {item["field"]: item for item in output_scores}
    ordered_scores = [score_map.get(field, {"field": field, "label": label, "value": "", "confidence": 0.0}) for field, label in DEFAULT_SCORE_FIELDS]
    apply_total_check(ordered_scores)
    return {
        "student_id": student_id,
        "name": name,
        "scores": ordered_scores,
        "detections": list(detections.values()),
        "source": "model",
        "message": "识别完成，请人工核对后确认记录。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROI + OCR/MNIST inference on one answer-sheet image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    result = infer_image(args.image, args.root, cpu=not args.gpu)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
