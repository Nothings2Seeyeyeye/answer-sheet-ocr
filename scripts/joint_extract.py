from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from ultralytics import YOLO

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

SCORE_FIELDS = FIELD_NAMES[2:]
SCORE_MAX = {
    "score_1": 30.0,
    "score_2_1": 12.0,
    "score_2_2": 12.0,
    "score_2_3": 12.0,
    "score_2_4": 12.0,
    "score_2_5": 13.0,
    "score_2_6": 15.0,
    "total": 100.0,
}


def clean_digits(text: Any) -> str:
    return "".join(ch for ch in str(text) if ch.isdigit())


def clean_chinese_name(text: Any) -> str:
    return re.sub(r"[^\u4e00-\u9fff]", "", str(text))


def clean_number(text: Any) -> str:
    value = "".join(ch for ch in str(text) if ch.isdigit() or ch == ".")
    if value.count(".") > 1:
        first = value.find(".")
        value = value[: first + 1] + value[first + 1 :].replace(".", "")
    if value.startswith("."):
        value = "0" + value
    if value.endswith("."):
        value = value[:-1]
    return value


def is_number(text: Any) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", clean_number(text)))


def number_equal(a: Any, b: Any) -> bool:
    if not is_number(a) or not is_number(b):
        return False
    return abs(float(clean_number(a)) - float(clean_number(b))) < 1e-6


def iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(1e-9, area_a + area_b - inter)


def crop_box(image: Image.Image, box: list[float], pad_ratio: float) -> Image.Image:
    x0, y0, x1, y1 = box
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio
    left = max(0, int(round(x0 - pad_x)))
    top = max(0, int(round(y0 - pad_y)))
    right = min(image.width, int(round(x1 + pad_x)))
    bottom = min(image.height, int(round(y1 + pad_y)))
    return image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))


def prepare_ocr_image(crop: Image.Image, field: str) -> Image.Image:
    image = crop.convert("RGB")
    scale = 3 if field in {"name", "student_id"} else 4
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    border = max(12, min(image.width, image.height) // 8)
    canvas = Image.new("RGB", (image.width + border * 2, image.height + border * 2), "white")
    canvas.paste(image, (border, border))
    return canvas


def fixed_digit_slices(image: Image.Image, count: int) -> list[Image.Image]:
    gray = image.convert("L")
    arr = np.asarray(gray)
    threshold = min(220, int(np.percentile(arr, 35)) + 35)
    mask = arr < threshold
    ys, xs = np.where(mask)
    if len(xs):
        x0 = max(0, int(xs.min()) - 2)
        x1 = min(gray.width, int(xs.max()) + 3)
        y0 = max(0, int(ys.min()) - 3)
        y1 = min(gray.height, int(ys.max()) + 4)
        gray = gray.crop((x0, y0, x1, y1))
    cells = []
    for idx in range(count):
        left = round(idx * gray.width / count)
        right = round((idx + 1) * gray.width / count)
        cells.append(gray.crop((left, 0, max(left + 1, right), gray.height)))
    return cells


def recognize_student_fixed_slices(mnist: Any, model: Any, device: Any, crop_path: Path, count: int = 10) -> tuple[str, float]:
    image = Image.open(crop_path).convert("RGB")
    chars = []
    confs = []
    for cell in fixed_digit_slices(image, count):
        char, conf = mnist.classify_glyph(model, device, cell)
        chars.append(char)
        confs.append(float(conf))
    return "".join(chars), float(np.mean(confs)) if confs else 0.0


class DirectChineseOCR:
    def __init__(self, engine_dir: Path) -> None:
        self.engine_dir = engine_dir.resolve()
        sys.path.insert(0, str(self.engine_dir))
        from model import OcrHandle  # type: ignore

        self.handle = OcrHandle()

    @staticmethod
    def normalize_box(box: Any) -> list[list[int]]:
        return np.asarray(box).round().astype(int).reshape(4, 2).tolist()

    def run(self, image_path: Path, compress: int) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        short_size = 32 * (max(64, compress) // 32)
        start = time.time()
        results = self.handle.text_predict(image, short_size)
        blocks = []
        texts = []
        for box, text, score in results:
            clean_text = re.sub(r"^\s*\d+、\s*", "", str(text)).strip()
            if clean_text:
                texts.append(clean_text)
            blocks.append({"text": clean_text, "score": round(float(score), 4), "box": self.normalize_box(box)})
        return {"text": "\n".join(texts), "blocks": blocks, "elapsed": round(time.time() - start, 3)}


def run_chineseocr(image_path: Path, engine_dir: Path, output_json: Path, compress: int) -> dict[str, Any]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not hasattr(run_chineseocr, "_direct") or getattr(run_chineseocr, "_engine_dir", None) != str(engine_dir.resolve()):
            setattr(run_chineseocr, "_direct", DirectChineseOCR(engine_dir))
            setattr(run_chineseocr, "_engine_dir", str(engine_dir.resolve()))
        data = getattr(run_chineseocr, "_direct").run(image_path, compress)
        output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data
    except Exception as direct_exc:
        direct_error = f"{type(direct_exc).__name__}: {direct_exc}"

    cmd = [
        sys.executable,
        "-m",
        "chineseocr_lite",
        str(image_path.resolve()),
        "--compress",
        str(compress),
        "--output",
        str(output_json.resolve()),
    ]
    proc = subprocess.run(cmd, cwd=str(engine_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return {"text": "", "blocks": [], "error": f"direct={direct_error}; subprocess={proc.stderr.strip()[-500:]}"}
    try:
        data = json.loads(output_json.read_text(encoding="utf-8"))
        if direct_error:
            data["direct_error"] = direct_error
        return data
    except Exception as exc:
        return {"text": "", "blocks": [], "error": f"direct={direct_error}; read={type(exc).__name__}: {exc}"}


def ocr_candidates(data: dict[str, Any]) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []
    for block in data.get("blocks", []):
        text = str(block.get("text", ""))
        score = float(block.get("score", 0.0))
        if text:
            candidates.append((text, score))
    text = str(data.get("text", ""))
    if text:
        candidates.append((text, 0.0))
    return candidates


def recognize_name(data: dict[str, Any]) -> tuple[str, float]:
    candidates = []
    for text, score in ocr_candidates(data):
        name = clean_chinese_name(text)
        if 1 <= len(name) <= 6:
            candidates.append((name, score))
    if not candidates:
        return "", 0.0
    return max(candidates, key=lambda item: (item[1], -abs(len(item[0]) - 3)))


def recognize_ocr_digits(data: dict[str, Any]) -> tuple[str, float]:
    candidates = []
    for text, score in ocr_candidates(data):
        digits = clean_digits(text)
        if digits:
            candidates.append((digits, score))
    if not candidates:
        return "", 0.0
    return max(candidates, key=lambda item: (len(item[0]), item[1]))


def recognize_ocr_number(data: dict[str, Any]) -> tuple[str, float]:
    candidates = []
    for text, score in ocr_candidates(data):
        number = clean_number(text)
        if number:
            candidates.append((number, score))
    if not candidates:
        return "", 0.0
    return max(candidates, key=lambda item: (item[1], len(item[0])))


def valid_score(value: str, field: str) -> bool:
    if not is_number(value):
        return False
    number = float(clean_number(value))
    max_value = SCORE_MAX.get(field, 100.0)
    return 0.0 <= number <= max_value


def choose_student_id(mnist_text: str, mnist_conf: float, ocr_text: str, ocr_conf: float) -> tuple[str, str, float]:
    candidates = [
        ("mnist", clean_digits(mnist_text), float(mnist_conf)),
        ("ocr", clean_digits(ocr_text), float(ocr_conf)),
    ]
    candidates = [item for item in candidates if item[1]]
    if not candidates:
        return "", "empty", 0.0

    def score(item: tuple[str, str, float]) -> tuple[float, float]:
        _source, text, conf = item
        length_bonus = 3.0 - min(3.0, abs(len(text) - 10))
        plausible_bonus = 2.0 if 8 <= len(text) <= 12 else 0.0
        exact_bonus = 2.0 if len(text) == 10 else 0.0
        return (length_bonus + plausible_bonus + exact_bonus + conf, conf)

    source, text, conf = max(candidates, key=score)
    return text, source, conf


def choose_score(field: str, mnist_text: str, mnist_conf: float, ocr_text: str, ocr_conf: float) -> tuple[str, str, float, str]:
    candidates = [
        ("mnist", clean_number(mnist_text), float(mnist_conf)),
        ("ocr", clean_number(ocr_text), float(ocr_conf)),
    ]
    valid = [item for item in candidates if valid_score(item[1], field)]
    if valid:
        source, value, conf = max(valid, key=lambda item: (item[2], -len(item[1])))
        return value, source, conf, ""
    nonempty = [item for item in candidates if item[1]]
    if nonempty:
        source, value, conf = max(nonempty, key=lambda item: item[2])
        return value, source, conf, "invalid_range_or_format"
    return "", "empty", 0.0, "empty"


def load_mnist_model(script_path: Path, model_dir: Path, cpu: bool):
    sys.path.insert(0, str(script_path.resolve().parent))
    import mnist_roi_digits as mnist  # type: ignore

    model, device = mnist.load_model(Path(model_dir), cpu=cpu)
    return mnist, model, device


def load_mnist(args: argparse.Namespace):
    sys.path.insert(0, str(Path(args.mnist_script).resolve().parent))
    import mnist_roi_digits as mnist  # type: ignore

    model, device = mnist.load_model(Path(args.mnist_model_dir), cpu=args.cpu)
    return mnist, model, device


def best_detections(result: Any) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if result.boxes is None:
        return output
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    for box, class_id, conf in zip(boxes, classes, confs):
        if class_id < 0 or class_id >= len(FIELD_NAMES):
            continue
        field = FIELD_NAMES[int(class_id)]
        current = output.get(field)
        if current is None or float(conf) > float(current["det_conf"]):
            output[field] = {
                "field": field,
                "det_conf": float(conf),
                "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            }
    return output


def evaluate_match(field: str, pred: str, gt: str) -> bool:
    if field == "name":
        return clean_chinese_name(pred) == clean_chinese_name(gt)
    if field == "student_id":
        return clean_digits(pred) == clean_digits(gt)
    return number_equal(pred, gt)


def summarize(rows: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(rows)),
        "overall": {
            "text_accuracy": round(float(rows["text_match"].mean()), 6) if len(rows) else 0.0,
            "mean_roi_iou": round(float(rows["roi_iou"].mean()), 6) if len(rows) else 0.0,
            "mean_det_conf": round(float(rows["det_conf"].mean()), 6) if len(rows) else 0.0,
        },
        "fields": {},
        "splits": {},
        "image_level": {},
    }
    for field, group in rows.groupby("field"):
        summary["fields"][field] = {
            "rows": int(len(group)),
            "text_accuracy": round(float(group["text_match"].mean()), 6),
            "mean_roi_iou": round(float(group["roi_iou"].mean()), 6),
            "mean_det_conf": round(float(group["det_conf"].mean()), 6),
            "mean_final_conf": round(float(group["final_conf"].mean()), 6),
        }
    for split, group in rows.groupby("split"):
        summary["splits"][split] = {
            "rows": int(len(group)),
            "text_accuracy": round(float(group["text_match"].mean()), 6),
            "mean_roi_iou": round(float(group["roi_iou"].mean()), 6),
        }
    image_ok = rows.groupby(["split", "file_name"])["text_match"].all().reset_index()
    summary["image_level"]["all_fields_accuracy"] = round(float(image_ok["text_match"].mean()), 6) if len(image_ok) else 0.0
    for split, group in image_ok.groupby("split"):
        summary["image_level"][split] = round(float(group["text_match"].mean()), 6)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint answer-sheet extraction: YOLO ROI -> OCR/MNIST -> fused text evaluation.")
    parser.add_argument("--dataset-csv", type=Path, default=Path("data/field_yolo/field_ground_truth.csv"))
    parser.add_argument("--weights", type=Path, default=Path("runs/detect/runs/field_yolo/weights/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("results_joint"))
    parser.add_argument("--ocr-engine-dir", type=Path, default=Path("/root/autodl-tmp/chineseocr_lite"))
    parser.add_argument("--mnist-script", type=Path, default=Path("/root/autodl-tmp/system_migration/root/OCR-student/mnist_roi_digits.py"))
    parser.add_argument("--mnist-model-dir", type=Path, default=Path("/root/autodl-tmp/ocr_work/Test score verification20260602/models/mnist_digit_cnn"))
    parser.add_argument("--score-mnist-model-dir", type=Path)
    parser.add_argument("--student-mnist-model-dir", type=Path)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--det-conf", type=float, default=0.05)
    parser.add_argument("--ocr-compress", type=int, default=480)
    parser.add_argument("--ocr-digits", action="store_true", help="Also run ChineseOCR on student_id and score fields for slow comparison/fusion.")
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    crop_root = args.output / "crops"
    ocr_root = args.output / "ocr_json"
    gt = pd.read_csv(args.dataset_csv)
    gt = gt[(gt["assigned"]) & (gt["split"].isin(["train", "val"]))].copy()
    model = YOLO(str(args.weights))
    mnist, default_mnist_model, default_mnist_device = load_mnist(args)
    score_mnist_model, score_mnist_device = default_mnist_model, default_mnist_device
    student_mnist_model, student_mnist_device = default_mnist_model, default_mnist_device
    if args.score_mnist_model_dir:
        _mnist, score_mnist_model, score_mnist_device = load_mnist_model(args.mnist_script, args.score_mnist_model_dir, args.cpu)
    if args.student_mnist_model_dir:
        _mnist, student_mnist_model, student_mnist_device = load_mnist_model(args.mnist_script, args.student_mnist_model_dir, args.cpu)
    rows: list[dict[str, Any]] = []

    image_groups = list(gt.groupby("image_path"))
    if args.limit_images > 0:
        image_groups = image_groups[: args.limit_images]

    for image_path, image_rows in image_groups:
        image_path_obj = Path(str(image_path))
        image = ImageOps.exif_transpose(Image.open(image_path_obj)).convert("RGB")
        result = model.predict(str(image_path_obj), imgsz=args.imgsz, conf=args.det_conf, verbose=False)[0]
        detections = best_detections(result)
        image_values: dict[str, str] = {}

        for _, gt_row in image_rows.sort_values("class_id").iterrows():
            field = str(gt_row["field"])
            det = detections.get(field)
            gt_box = [float(gt_row["xmin"]), float(gt_row["ymin"]), float(gt_row["xmax"]), float(gt_row["ymax"])]
            crop_path = ""
            ocr_text = ""
            ocr_conf = 0.0
            mnist_text = ""
            mnist_conf = 0.0
            final_text = ""
            final_source = "missing_roi"
            final_conf = 0.0
            note = ""
            roi_iou = 0.0
            det_conf = 0.0
            pred_box = [np.nan, np.nan, np.nan, np.nan]

            if det is not None:
                det_conf = float(det["det_conf"])
                pred_box = det["box"]
                roi_iou = iou(pred_box, gt_box)
                pad = 0.10 if field in {"name", "student_id"} else 0.06
                crop = crop_box(image, pred_box, pad)
                split = str(gt_row["split"])
                crop_dir = crop_root / split / field
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path_obj = crop_dir / f"{Path(str(gt_row['file_name'])).stem}_{field}.jpg"
                crop.save(crop_path_obj, quality=95)
                crop_path = str(crop_path_obj)

                ocr_data: dict[str, Any] = {"text": "", "blocks": []}
                if field == "name" or args.ocr_digits:
                    ocr_image = prepare_ocr_image(crop, field)
                    ocr_input = crop_path_obj.with_name(crop_path_obj.stem + "_ocr.jpg")
                    ocr_image.save(ocr_input, quality=95)
                    ocr_data = run_chineseocr(
                        ocr_input,
                        args.ocr_engine_dir,
                        ocr_root / split / f"{Path(str(gt_row['file_name'])).stem}_{field}.json",
                        args.ocr_compress,
                    )

                if field == "name":
                    final_text, final_conf = recognize_name(ocr_data)
                    ocr_text, ocr_conf = final_text, final_conf
                    final_source = "ocr"
                    if not final_text:
                        note = "ocr_empty"
                elif field == "student_id":
                    fixed_text, fixed_conf = recognize_student_fixed_slices(mnist, student_mnist_model, student_mnist_device, crop_path_obj, count=10)
                    old_text, old_conf, _details = mnist.recognize_student_roi(student_mnist_model, student_mnist_device, crop_path_obj)
                    mnist_text, mnist_conf = (fixed_text, fixed_conf) if fixed_conf >= old_conf or len(clean_digits(old_text)) != 10 else (old_text, old_conf)
                    ocr_text, ocr_conf = recognize_ocr_digits(ocr_data)
                    final_text, final_source, final_conf = choose_student_id(mnist_text, mnist_conf, ocr_text, ocr_conf)
                    if len(clean_digits(final_text)) != 10:
                        note = "student_id_length_not_10"
                else:
                    mnist_text, mnist_conf, _details = mnist.recognize_numeric_roi(score_mnist_model, score_mnist_device, crop_path_obj, mode="score")
                    ocr_text, ocr_conf = recognize_ocr_number(ocr_data)
                    final_text, final_source, final_conf, note = choose_score(field, mnist_text, mnist_conf, ocr_text, ocr_conf)

            gt_text = str(gt_row["gt_text"])
            match = evaluate_match(field, final_text, gt_text)
            image_values[field] = final_text
            rows.append(
                {
                    "split": gt_row["split"],
                    "file_name": gt_row["file_name"],
                    "field": field,
                    "gt_text": gt_text,
                    "final_text": final_text,
                    "final_source": final_source,
                    "final_conf": round(float(final_conf), 6),
                    "text_match": bool(match),
                    "ocr_text": ocr_text,
                    "ocr_conf": round(float(ocr_conf), 6),
                    "mnist_text": mnist_text,
                    "mnist_conf": round(float(mnist_conf), 6),
                    "det_conf": round(float(det_conf), 6),
                    "roi_iou": round(float(roi_iou), 6),
                    "gt_xmin": gt_box[0],
                    "gt_ymin": gt_box[1],
                    "gt_xmax": gt_box[2],
                    "gt_ymax": gt_box[3],
                    "pred_xmin": pred_box[0],
                    "pred_ymin": pred_box[1],
                    "pred_xmax": pred_box[2],
                    "pred_ymax": pred_box[3],
                    "crop_path": crop_path,
                    "note": note,
                }
            )

        component_values = [image_values.get(field, "") for field in SCORE_FIELDS[:-1]]
        if all(is_number(value) for value in component_values):
            computed_total = sum(float(clean_number(value)) for value in component_values)
            total_pred = image_values.get("total", "")
            if is_number(total_pred) and abs(computed_total - float(clean_number(total_pred))) > 1e-6:
                for row in rows[-len(image_rows) :]:
                    if row["field"] == "total":
                        row["final_text"] = f"{computed_total:g}"
                        row["final_source"] = "computed_sum"
                        row["final_conf"] = min(1.0, max(float(row.get("final_conf", 0.0)), 0.99))
                        row["text_match"] = evaluate_match("total", row["final_text"], row["gt_text"])
                        row["note"] = (str(row["note"]) + ";total_inconsistent").strip(";")
                        row["computed_total"] = f"{computed_total:g}"

    pred = pd.DataFrame(rows)
    pred.to_csv(args.output / "joint_predictions.csv", index=False, encoding="utf-8-sig")
    pred.to_excel(args.output / "joint_predictions.xlsx", index=False)
    errors = pred[~pred["text_match"]].copy()
    errors.to_csv(args.output / "joint_errors.csv", index=False, encoding="utf-8-sig")
    errors.to_excel(args.output / "joint_errors.xlsx", index=False)
    summary = summarize(pred)
    (args.output / "joint_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
