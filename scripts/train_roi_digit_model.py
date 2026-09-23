from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCORE_FIELDS = ["score_1", "score_2_1", "score_2_2", "score_2_3", "score_2_4", "score_2_5", "score_2_6", "total"]


def clean_digits(text: Any) -> str:
    return "".join(ch for ch in str(text) if ch.isdigit())


def clean_number_digits(text: Any) -> str:
    return "".join(ch for ch in str(text) if ch.isdigit())


def crop_box(image: Image.Image, box: list[float], pad_ratio: float) -> Image.Image:
    x0, y0, x1, y1 = box
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    left = max(0, int(round(x0 - w * pad_ratio)))
    top = max(0, int(round(y0 - h * pad_ratio)))
    right = min(image.width, int(round(x1 + w * pad_ratio)))
    bottom = min(image.height, int(round(y1 + h * pad_ratio)))
    return image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))


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


def load_mnist_module(script_path: Path):
    sys.path.insert(0, str(script_path.resolve().parent))
    import mnist_roi_digits as mnist  # type: ignore

    return mnist


class GlyphDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], mnist: Any, augment: bool) -> None:
        self.samples = samples
        self.mnist = mnist
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = sample["image"]
        if self.augment:
            angle = random.uniform(-8.0, 8.0)
            image = image.rotate(angle, expand=True, fillcolor=255)
        x = self.mnist.glyph_to_tensor(image)
        y = torch.tensor(int(sample["label"]), dtype=torch.long)
        return x, y


def collect_samples(csv_path: Path, mnist: Any, sample_dir: Path, task: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table = pd.read_csv(csv_path)
    table = table[(table["assigned"]) & (table["split"].isin(["train", "val"]))].copy()
    samples: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    sample_dir.mkdir(parents=True, exist_ok=True)

    for _, row in table.iterrows():
        field = str(row["field"])
        if task == "score" and field not in SCORE_FIELDS:
            continue
        if task == "student" and field != "student_id":
            continue
        if task == "both" and field != "student_id" and field not in SCORE_FIELDS:
            continue
        image = ImageOps.exif_transpose(Image.open(str(row["image_path"]))).convert("RGB")
        box = [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
        pad = 0.10 if field == "student_id" else 0.06
        crop = crop_box(image, box, pad)
        target = clean_digits(row["gt_text"]) if field == "student_id" else clean_number_digits(row["gt_text"])
        if not target:
            continue
        if field == "student_id":
            glyph_images = fixed_digit_slices(crop, len(target))
        else:
            glyphs = [g for g in mnist.segment_glyphs(crop, mode="score") if not getattr(g, "is_dot", False)]
            glyph_images = [g.image.convert("L") for g in glyphs]
        if len(glyph_images) != len(target):
            rejects.append(
                {
                    "split": row["split"],
                    "file_name": row["file_name"],
                    "field": field,
                    "target": target,
                    "glyph_count": len(glyph_images),
                    "reason": "glyph_count_mismatch",
                }
            )
            continue
        for pos, (glyph_image, label) in enumerate(zip(glyph_images, target)):
            if len(samples) < 300:
                out_dir = sample_dir / str(label)
                out_dir.mkdir(parents=True, exist_ok=True)
                glyph_image.save(out_dir / f"{Path(str(row['file_name'])).stem}_{field}_{pos}.png")
            samples.append(
                {
                    "split": row["split"],
                    "file_name": row["file_name"],
                    "field": field,
                    "position": pos,
                    "label": int(label),
                    "image": glyph_image,
                }
            )
    return samples, rejects


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * x.size(0)
            total += x.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
    return correct / max(1, total), loss_sum / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the MNIST digit CNN on real answer-sheet ROI glyphs.")
    parser.add_argument("--dataset-csv", type=Path, default=Path("data/field_yolo/field_ground_truth.csv"))
    parser.add_argument("--mnist-script", type=Path, default=Path("/root/autodl-tmp/system_migration/root/OCR-student/mnist_roi_digits.py"))
    parser.add_argument("--base-model-dir", type=Path, default=Path("/root/autodl-tmp/ocr_work/Test score verification20260602/models/mnist_digit_cnn"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/roi_digit_cnn"))
    parser.add_argument("--task", choices=["score", "student", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    mnist = load_mnist_module(args.mnist_script)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, rejects = collect_samples(args.dataset_csv, mnist, args.output_dir / "glyph_samples", args.task)
    train_samples = [s for s in samples if s["split"] == "train"]
    val_samples = [s for s in samples if s["split"] == "val"]
    if not train_samples or not val_samples:
        raise RuntimeError(f"not enough aligned glyphs: train={len(train_samples)}, val={len(val_samples)}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, _ = mnist.load_model(args.base_model_dir, cpu=args.cpu)
    model = model.to(device)
    train_loader = DataLoader(GlyphDataset(train_samples, mnist, augment=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(GlyphDataset(val_samples, mnist, augment=False), batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = correct = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * x.size(0)
            total += x.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_acc": correct / max(1, total),
            "train_loss": loss_sum / max(1, total),
            "val_acc": val_acc,
            "val_loss": val_loss,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} train_acc={record['train_acc']:.4f} "
            f"val_acc={val_acc:.4f} train_loss={record['train_loss']:.4f} val_loss={val_loss:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.output_dir / "mnist_digit_cnn.pt")
    pd.DataFrame([{k: v for k, v in s.items() if k != "image"} for s in samples]).to_csv(
        args.output_dir / "glyph_manifest.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(rejects).to_csv(args.output_dir / "glyph_rejects.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "train_glyphs": len(train_samples),
        "val_glyphs": len(val_samples),
        "rejects": len(rejects),
        "task": args.task,
        "best_val_acc": best_val_acc,
        "epochs": args.epochs,
        "history": history,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
