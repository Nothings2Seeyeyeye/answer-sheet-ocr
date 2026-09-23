from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


DEFAULT_PATTERNS = [
    "README.md",
    "scripts/*.py",
    "data/field_yolo/data.yaml",
    "data/field_yolo/summary.json",
    "data/field_yolo/field_ground_truth.csv",
    "runs/field_yolo/args.yaml",
    "runs/field_yolo/results.csv",
    "runs/field_yolo/results.png",
    "runs/field_yolo/confusion_matrix.png",
    "runs/field_yolo/weights/best.pt",
    "runs/field_yolo/weights/last.pt",
    "runs/detect/runs/field_yolo/args.yaml",
    "runs/detect/runs/field_yolo/results.csv",
    "runs/detect/runs/field_yolo/results.png",
    "runs/detect/runs/field_yolo/confusion_matrix.png",
    "runs/detect/runs/field_yolo/labels.jpg",
    "runs/detect/runs/field_yolo/weights/best.pt",
    "runs/detect/runs/field_yolo/weights/last.pt",
    "models/roi_digit_cnn/mnist_digit_cnn.pt",
    "models/roi_digit_cnn/manifest.json",
    "models/roi_digit_cnn/glyph_manifest.csv",
    "models/roi_digit_cnn/glyph_rejects.csv",
    "models/roi_score_digit_cnn/mnist_digit_cnn.pt",
    "models/roi_score_digit_cnn/manifest.json",
    "models/roi_score_digit_cnn/glyph_manifest.csv",
    "models/roi_score_digit_cnn/glyph_rejects.csv",
    "models/roi_student_digit_cnn/mnist_digit_cnn.pt",
    "models/roi_student_digit_cnn/manifest.json",
    "models/roi_student_digit_cnn/glyph_manifest.csv",
    "models/roi_student_digit_cnn/glyph_rejects.csv",
    "results/*.json",
    "results/*.csv",
    "results/*.xlsx",
    "results_joint/*.json",
    "results_joint/*.csv",
    "results_joint/*.xlsx",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Package key code and experiment outputs.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("student_field_yolo_results.zip"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pattern in DEFAULT_PATTERNS:
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    zf.write(path, path.relative_to(root))
    print(output)


if __name__ == "__main__":
    main()
