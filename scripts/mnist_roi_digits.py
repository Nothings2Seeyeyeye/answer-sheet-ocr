from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

FIELDS = ["一", "二(1)", "二(2)", "二(3)", "二(4)", "二(5)", "二(6)", "总分"]
IMAGE_SIZE = 28
FIELD_MAX = dict(zip(FIELDS, [30.0, 12.0, 12.0, 12.0, 12.0, 13.0, 15.0, 100.0]))
FIELD_MIN_CONF = 0.60


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Glyph:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    area: int
    is_dot: bool = False


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.RandomRotation(args.rotate),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.85, 1.12)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    data_dir = Path(args.data_dir)
    train_full = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    eval_full = datasets.MNIST(str(data_dir), train=True, download=True, transform=eval_transform)
    train_size = int(len(train_full) * (1.0 - args.val_fraction))
    val_size = len(train_full) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, _ = random_split(train_full, [train_size, val_size], generator=generator)
    _, val_set = random_split(eval_full, [train_size, val_size], generator=generator)
    test_set = datasets.MNIST(str(data_dir), train=False, download=True, transform=eval_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MnistDigitCnn().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_val_acc = -1.0
    history: list[dict[str, float]] = []
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
            "epoch": float(epoch),
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
    test_acc, test_loss = evaluate(model, test_loader, criterion, device)
    torch.save(model.state_dict(), out_dir / "mnist_digit_cnn.pt")
    manifest = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "epochs": args.epochs,
        "history": history,
        "note": "MNIST is single-digit pretraining; ROI segmentation quality controls answer-sheet accuracy.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_dir / 'mnist_digit_cnn.pt'}")
    print(f"best_val_acc={best_val_acc:.4f} test_acc={test_acc:.4f} test_loss={test_loss:.4f}")


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total = correct = 0
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


def load_model(model_dir: Path, cpu: bool = False) -> tuple[MnistDigitCnn, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model = MnistDigitCnn().to(device)
    state = torch.load(model_dir / "mnist_digit_cnn.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def preprocess_binary(image: Image.Image, mode: str = "score") -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    if mode == "score":
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        red_mask = (r > 80) & (r > g + 18) & (r > b + 18)
        if int(red_mask.sum()) >= 20:
            return remove_small_components(red_mask, min_area=8)

    gray = np.asarray(image.convert("L"))
    dark_cutoff = int(np.percentile(gray, 18)) + 18
    threshold = max(65, min(185, dark_cutoff))
    mask = gray < threshold
    return remove_small_components(mask, min_area=12 if mode == "student" else 8)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    visited = np.zeros(mask.shape, dtype=bool)
    output = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    ys, xs = np.where(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = True
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(points) >= min_area:
            for y, x in points:
                output[y, x] = True
    return output


def segment_glyphs(image: Image.Image, min_width: int = 3, mode: str = "score") -> list[Glyph]:
    mask = preprocess_binary(image, mode=mode)
    if not mask.any():
        return []
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    col = mask.sum(axis=0)
    active = col > max(1, int(mask.shape[0] * 0.015))
    active = close_1d(active, max_gap=max(2, int(mask.shape[1] * 0.012)))
    runs = bool_runs(active)
    glyphs: list[Glyph] = []
    for start, end in runs:
        if end - start + 1 < min_width:
            continue
        sub = mask[:, start : end + 1]
        sy, sx = np.where(sub)
        if len(sx) == 0:
            continue
        gx0 = x0 + start + int(sx.min())
        gx1 = x0 + start + int(sx.max()) + 1
        gy0 = y0 + int(sy.min())
        gy1 = y0 + int(sy.max()) + 1
        area = int(sub.sum())
        height = gy1 - gy0
        width = gx1 - gx0
        full_height = image.height
        is_dot = height <= full_height * 0.22 and width <= image.width * 0.12 and gy0 > full_height * 0.48
        glyphs.append(Glyph(image.crop((gx0, gy0, gx1, gy1)), (gx0, gy0, gx1, gy1), area, is_dot))
    return merge_oversegmented(glyphs, image)


def connected_glyphs(image: Image.Image, mode: str = "student") -> list[Glyph]:
    mask = preprocess_binary(image, mode=mode)
    comps = connected_components(mask, min_area=18)
    glyphs: list[Glyph] = []
    height, width = mask.shape
    for x0, y0, x1, y1, area in comps:
        box_w = x1 - x0
        box_h = y1 - y0
        if box_w < 3 or box_h < 8:
            continue
        if box_w > width * 0.18 or box_h > height * 0.35:
            continue
        if area / max(1, box_w * box_h) < 0.08:
            continue
        glyphs.append(Glyph(image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1), area, False))
    return sorted(glyphs, key=lambda g: (g.bbox[1], g.bbox[0]))


def connected_components(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int, int]]:
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    components: list[tuple[int, int, int, int, int]] = []
    ys, xs = np.where(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = True
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(points) >= min_area:
            py = [p[0] for p in points]
            px = [p[1] for p in points]
            components.append((min(px), min(py), max(px) + 1, max(py) + 1, len(points)))
    return components


def bool_runs(active: np.ndarray) -> list[tuple[int, int]]:
    idx = np.where(active)[0]
    if len(idx) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for raw in idx[1:]:
        cur = int(raw)
        if cur - prev > 1:
            runs.append((start, prev))
            start = cur
        prev = cur
    runs.append((start, prev))
    return runs


def close_1d(active: np.ndarray, max_gap: int) -> np.ndarray:
    runs = bool_runs(active)
    closed = np.zeros_like(active, dtype=bool)
    if not runs:
        return closed
    start, end = runs[0]
    for cur_start, cur_end in runs[1:]:
        if cur_start - end <= max_gap:
            end = cur_end
        else:
            closed[start : end + 1] = True
            start, end = cur_start, cur_end
    closed[start : end + 1] = True
    return closed


def merge_oversegmented(glyphs: list[Glyph], image: Image.Image) -> list[Glyph]:
    if len(glyphs) <= 1:
        return glyphs
    merged: list[Glyph] = []
    i = 0
    median_width = np.median([g.bbox[2] - g.bbox[0] for g in glyphs])
    while i < len(glyphs):
        cur = glyphs[i]
        if i + 1 < len(glyphs) and not cur.is_dot:
            nxt = glyphs[i + 1]
            gap = nxt.bbox[0] - cur.bbox[2]
            cur_width = cur.bbox[2] - cur.bbox[0]
            if gap <= max(2, median_width * 0.18) and cur_width < median_width * 0.55 and not nxt.is_dot:
                box = (
                    min(cur.bbox[0], nxt.bbox[0]),
                    min(cur.bbox[1], nxt.bbox[1]),
                    max(cur.bbox[2], nxt.bbox[2]),
                    max(cur.bbox[3], nxt.bbox[3]),
                )
                merged.append(Glyph(image.crop(box), box, cur.area + nxt.area, False))
                i += 2
                continue
        merged.append(cur)
        i += 1
    return merged


def glyph_to_tensor(glyph: Image.Image) -> torch.Tensor:
    gray = glyph.convert("L")
    arr = np.asarray(gray)
    mask = arr < 220
    if mask.any():
        ys, xs = np.where(mask)
        pad = 3
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(gray.width, int(xs.max()) + pad + 1)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(gray.height, int(ys.max()) + pad + 1)
        gray = gray.crop((x0, y0, x1, y1))
    scale = min(20 / max(1, gray.width), 20 / max(1, gray.height))
    new_size = (max(1, int(gray.width * scale)), max(1, int(gray.height * scale)))
    gray = gray.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 255)
    canvas.paste(gray, ((28 - gray.width) // 2, (28 - gray.height) // 2))
    ink = (255.0 - np.asarray(canvas, dtype=np.float32)) / 255.0
    tensor = torch.from_numpy(ink[None, :, :])
    return (tensor - 0.1307) / 0.3081


def classify_glyph(model: nn.Module, device: torch.device, glyph: Image.Image) -> tuple[str, float]:
    x = glyph_to_tensor(glyph).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    return str(int(idx.item())), float(conf.item())


def recognize_numeric_roi(
    model: nn.Module,
    device: torch.device,
    image_path: Path,
    mode: str = "score",
) -> tuple[str, float, list[dict[str, Any]]]:
    image = Image.open(image_path).convert("RGB")
    glyphs = segment_glyphs(image, mode=mode)
    chars: list[str] = []
    confidences: list[float] = []
    details: list[dict[str, Any]] = []
    for glyph in glyphs:
        if glyph.is_dot:
            chars.append(".")
            confidences.append(0.95)
            details.append({"char": ".", "confidence": 0.95, "bbox": glyph.bbox})
            continue
        char, conf = classify_glyph(model, device, glyph.image)
        chars.append(char)
        confidences.append(conf)
        details.append({"char": char, "confidence": conf, "bbox": glyph.bbox})
    text = "".join(chars)
    text = cleanup_numeric_text(text)
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return text, confidence, details


def recognize_student_roi(model: nn.Module, device: torch.device, image_path: Path) -> tuple[str, float, list[dict[str, Any]]]:
    image = Image.open(image_path).convert("RGB")
    glyphs = connected_glyphs(image, mode="student")
    records: list[dict[str, Any]] = []
    for glyph in glyphs:
        char, conf = classify_glyph(model, device, glyph.image)
        if conf < 0.35:
            continue
        records.append({"char": char, "confidence": conf, "bbox": glyph.bbox})
    if not records:
        return "", 0.0, []

    rows: list[list[dict[str, Any]]] = []
    for rec in sorted(records, key=lambda r: ((r["bbox"][1] + r["bbox"][3]) / 2, r["bbox"][0])):
        cy = (rec["bbox"][1] + rec["bbox"][3]) / 2
        placed = False
        for row in rows:
            row_cy = np.mean([(r["bbox"][1] + r["bbox"][3]) / 2 for r in row])
            row_h = np.mean([r["bbox"][3] - r["bbox"][1] for r in row])
            if abs(cy - row_cy) <= max(10.0, row_h * 0.75):
                row.append(rec)
                placed = True
                break
        if not placed:
            rows.append([rec])

    candidates: list[list[dict[str, Any]]] = []
    for row in rows:
        ordered = sorted(row, key=lambda r: r["bbox"][0])
        widths = [r["bbox"][2] - r["bbox"][0] for r in ordered]
        median_w = float(np.median(widths)) if widths else 8.0
        seq: list[dict[str, Any]] = []
        prev_x1: int | None = None
        for rec in ordered:
            gap_ok = prev_x1 is None or rec["bbox"][0] - prev_x1 <= max(18.0, median_w * 1.35)
            if not gap_ok:
                if seq:
                    candidates.append(seq)
                seq = []
            seq.append(rec)
            prev_x1 = rec["bbox"][2]
        if seq:
            candidates.append(seq)

    plausible = [seq for seq in candidates if 4 <= len(seq) <= 14]
    if not plausible:
        plausible = candidates

    def candidate_score(seq: list[dict[str, Any]]) -> float:
        x0 = min(r["bbox"][0] for r in seq)
        x1 = max(r["bbox"][2] for r in seq)
        y0 = min(r["bbox"][1] for r in seq)
        y1 = max(r["bbox"][3] for r in seq)
        avg_conf = float(np.mean([r["confidence"] for r in seq]))
        score = len(seq) * 2.0 + avg_conf
        if x0 < image.width * 0.08:
            score -= 2.0
        if x1 > image.width * 0.82:
            score -= 3.0
        if y0 < image.height * 0.20 or y1 > image.height * 0.82:
            score -= 2.0
        if 6 <= len(seq) <= 12:
            score += 2.0
        return score

    best = max(plausible, key=candidate_score)
    text = "".join(r["char"] for r in best)
    conf = float(np.mean([r["confidence"] for r in best])) if best else 0.0
    return text, conf, best


def apply_score_constraints(value: str, confidence: float, field: str) -> tuple[str, str]:
    if not value:
        return "", f"{field}_empty"
    try:
        number = float(value)
    except ValueError:
        return "", f"{field}_not_numeric"
    notes: list[str] = []
    max_value = FIELD_MAX.get(field)
    if max_value is not None and not (0.0 <= number <= max_value):
        notes.append(f"{field}_out_of_range_{value}")
    if confidence < FIELD_MIN_CONF:
        notes.append(f"{field}_low_conf_{confidence:.2f}")
    if notes:
        return value, ";".join(notes)
    return value, ""


def cleanup_numeric_text(text: str) -> str:
    text = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if text.count(".") > 1:
        first = text.find(".")
        text = text[: first + 1] + text[first + 1 :].replace(".", "")
    if text.startswith("."):
        text = "0" + text
    if text.endswith("."):
        text = text[:-1]
    return text


def run_rois(args: argparse.Namespace) -> None:
    model, device = load_model(Path(args.model_dir), cpu=args.cpu)
    manifest = pd.read_csv(args.roi_manifest).fillna("")
    rows: dict[str, dict[str, Any]] = {}
    digit_rows: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        file_name = str(item.get("file_name", ""))
        roi_type = str(item.get("roi_type", ""))
        field = str(item.get("field", ""))
        crop_path = resolve_path(str(item.get("crop_path", "")), Path(args.roi_manifest))
        row = rows.setdefault(file_name, {"file_name": file_name, "review_notes": ""})
        if roi_type == "score_cell" and field in FIELDS:
            value, conf, details = recognize_numeric_roi(model, device, crop_path, mode="score")
            value, note = apply_score_constraints(value, conf, field)
            row[field] = value
            row[f"{field}_confidence"] = round(conf, 4)
            row[f"{field}_crop"] = str(crop_path)
            row[f"{field}_glyphs"] = json.dumps(details, ensure_ascii=False)
            if note:
                row["review_notes"] = append_note(str(row.get("review_notes", "")), note)
        elif roi_type == "student":
            value, conf, details = recognize_student_roi(model, device, crop_path)
            row["student_roi"] = str(crop_path)
            row["student_id_mnist_digits"] = value
            row["student_id_mnist_confidence"] = round(conf, 4)
            row["student_digit_glyphs"] = json.dumps(details, ensure_ascii=False)
            if len(value) < 4:
                row["review_notes"] = append_note(str(row.get("review_notes", "")), "student_id_short_or_missing")
            digit_rows.append(
                {
                    "file_name": file_name,
                    "student_roi": str(crop_path),
                    "student_id_mnist_digits": value,
                    "confidence": round(conf, 4),
                    "glyphs": json.dumps(details, ensure_ascii=False),
                }
            )
        elif roi_type == "score_table":
            row["score_table_roi"] = str(crop_path)
    for row in rows.values():
        total = safe_sum([row.get(field, "") for field in FIELDS[:-1]])
        if total is not None:
            row["computed_total"] = f"{total:g}"
        for field in FIELDS:
            row.setdefault(field, "")
            row.setdefault(f"{field}_confidence", 0.0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_rows = list(rows.values())
    pd.DataFrame(score_rows).to_csv(output_dir / "mnist_score_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(score_rows).to_excel(output_dir / "mnist_score_results.xlsx", index=False)
    pd.DataFrame(digit_rows).to_csv(output_dir / "mnist_student_digits.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(digit_rows).to_excel(output_dir / "mnist_student_digits.xlsx", index=False)
    print(f"wrote {output_dir / 'mnist_score_results.xlsx'}")
    print(f"wrote {output_dir / 'mnist_student_digits.xlsx'}")


def safe_sum(values: list[Any]) -> float | None:
    total = 0.0
    for value in values:
        text = cleanup_numeric_text(str(value))
        if not text:
            return None
        try:
            total += float(text)
        except ValueError:
            return None
    return total


def append_note(existing: str, note: str) -> str:
    if not existing:
        return note
    return existing + ";" + note


def resolve_path(path_text: str, manifest_path: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [manifest_path.parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MNIST digit model and run it on thresholded answer-sheet ROIs.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data-dir", default="data/mnist")
    p.add_argument("--model-dir", default="models/mnist_digit_cnn")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--rotate", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=train)
    p = sub.add_parser("run-rois")
    p.add_argument("--model-dir", default="models/mnist_digit_cnn")
    p.add_argument("--roi-manifest", default="outputs/roi_first_test/roi_manifest.csv")
    p.add_argument("--output-dir", default="outputs/mnist_roi_digits")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=run_rois)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
