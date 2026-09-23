from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from field_dataset import FIELD_NAMES


def iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(1e-9, area_a + area_b - inter)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO field detector by role-level best match.")
    parser.add_argument("--dataset-csv", type=Path, default=Path("data/field_yolo/field_ground_truth.csv"))
    parser.add_argument("--weights", type=Path, default=Path("runs/field_yolo/weights/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(args.dataset_csv)
    gt = gt[(gt["assigned"]) & (gt["split"].isin(["train", "val"]))].copy()
    model = YOLO(str(args.weights))
    rows: list[dict[str, object]] = []

    for image_path, group in gt.groupby("image_path"):
        result = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        pred_by_class: dict[int, list[dict[str, float]]] = {i: [] for i in range(len(FIELD_NAMES))}
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for box, cls_id, conf in zip(boxes, classes, confs):
                pred_by_class.setdefault(int(cls_id), []).append(
                    {"xmin": float(box[0]), "ymin": float(box[1]), "xmax": float(box[2]), "ymax": float(box[3]), "conf": float(conf)}
                )

        for _, row in group.iterrows():
            cls_id = int(row["class_id"])
            true_box = [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
            candidates = pred_by_class.get(cls_id, [])
            best = None
            best_iou = 0.0
            for cand in candidates:
                score = iou(true_box, [cand["xmin"], cand["ymin"], cand["xmax"], cand["ymax"]])
                if score > best_iou:
                    best_iou = score
                    best = cand
            rows.append(
                {
                    "split": row["split"],
                    "file_name": row["file_name"],
                    "field": row["field"],
                    "field_label": row["field_label"],
                    "gt_text": row["gt_text"],
                    "gt_xmin": row["xmin"],
                    "gt_ymin": row["ymin"],
                    "gt_xmax": row["xmax"],
                    "gt_ymax": row["ymax"],
                    "pred_conf": round(best["conf"], 6) if best else 0.0,
                    "pred_xmin": round(best["xmin"], 2) if best else None,
                    "pred_ymin": round(best["ymin"], 2) if best else None,
                    "pred_xmax": round(best["xmax"], 2) if best else None,
                    "pred_ymax": round(best["ymax"], 2) if best else None,
                    "iou": round(best_iou, 6),
                    "matched": best_iou >= args.iou_threshold,
                }
            )

    pred = pd.DataFrame(rows)
    pred.to_csv(args.output / "field_predictions.csv", index=False, encoding="utf-8-sig")
    pred.to_excel(args.output / "field_predictions.xlsx", index=False)

    summary: dict[str, object] = {
        "rows": int(len(pred)),
        "iou_threshold": args.iou_threshold,
        "overall": {
            "recall_at_iou": round(float(pred["matched"].mean()), 6),
            "mean_iou": round(float(pred["iou"].mean()), 6),
            "mean_conf": round(float(pred["pred_conf"].mean()), 6),
        },
        "splits": {},
        "fields": {},
    }
    for split, group in pred.groupby("split"):
        summary["splits"][split] = {
            "rows": int(len(group)),
            "recall_at_iou": round(float(group["matched"].mean()), 6),
            "mean_iou": round(float(group["iou"].mean()), 6),
        }
    for field, group in pred.groupby("field"):
        summary["fields"][field] = {
            "rows": int(len(group)),
            "recall_at_iou": round(float(group["matched"].mean()), 6),
            "mean_iou": round(float(group["iou"].mean()), 6),
            "mean_conf": round(float(group["pred_conf"].mean()), 6),
        }
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
