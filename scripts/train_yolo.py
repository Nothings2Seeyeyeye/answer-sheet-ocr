from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO field detector.")
    parser.add_argument("--data", type=Path, default=Path("data/field_yolo/data.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        project=str(args.output.resolve()),
        name="field_yolo",
        exist_ok=True,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        amp=args.amp,
        workers=4,
        cache=False,
        pretrained=True,
        seed=2026,
        plots=True,
    )


if __name__ == "__main__":
    main()
