from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


class MnistDigitCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def export_digit(model_dir: Path) -> None:
    model = MnistDigitCnn()
    state = torch.load(model_dir / "mnist_digit_cnn.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 1, 28, 28),
        model_dir / "mnist_digit_cnn.onnx",
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def main() -> None:
    for name in ("roi_score_digit_cnn", "roi_student_digit_cnn"):
        export_digit(ROOT / "models" / name)

    model = YOLO(str(ROOT / "weights" / "field_yolo" / "best.pt"))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=1280,
            opset=17,
            simplify=False,
            dynamic=False,
            device="cpu",
        )
    )
    target = ROOT / "weights" / "field_yolo" / "best.onnx"
    if exported.resolve() != target.resolve():
        exported.replace(target)
    print(target)


if __name__ == "__main__":
    main()
