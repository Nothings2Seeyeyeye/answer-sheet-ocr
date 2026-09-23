from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image


@dataclass
class Glyph:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    area: int
    is_dot: bool = False


class DigitRecognizer:
    def __init__(self, model_path: str | Path) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def classify(self, glyph: Image.Image) -> tuple[str, float]:
        logits = self.session.run(None, {self.input_name: glyph_to_array(glyph)})[0][0]
        logits = logits - np.max(logits)
        probabilities = np.exp(logits) / np.exp(logits).sum()
        index = int(np.argmax(probabilities))
        return str(index), float(probabilities[index])

    def recognize_numeric_roi(self, image_path: Path, mode: str = "score") -> tuple[str, float, list[dict[str, Any]]]:
        image = Image.open(image_path).convert("RGB")
        glyphs = segment_glyphs(image, mode=mode)
        chars: list[str] = []
        confidences: list[float] = []
        details: list[dict[str, Any]] = []
        for glyph in glyphs:
            if glyph.is_dot:
                char, confidence = ".", 0.95
            else:
                char, confidence = self.classify(glyph.image)
            chars.append(char)
            confidences.append(confidence)
            details.append({"char": char, "confidence": confidence, "bbox": glyph.bbox})
        return cleanup_numeric_text("".join(chars)), float(np.mean(confidences)) if confidences else 0.0, details


def glyph_to_array(glyph: Image.Image) -> np.ndarray:
    gray = glyph.convert("L")
    array = np.asarray(gray)
    mask = array < 220
    if mask.any():
        ys, xs = np.where(mask)
        pad = 3
        gray = gray.crop(
            (
                max(0, int(xs.min()) - pad),
                max(0, int(ys.min()) - pad),
                min(gray.width, int(xs.max()) + pad + 1),
                min(gray.height, int(ys.max()) + pad + 1),
            )
        )
    scale = min(20 / max(1, gray.width), 20 / max(1, gray.height))
    size = (max(1, int(gray.width * scale)), max(1, int(gray.height * scale)))
    gray = gray.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 255)
    canvas.paste(gray, ((28 - gray.width) // 2, (28 - gray.height) // 2))
    ink = (255.0 - np.asarray(canvas, dtype=np.float32)) / 255.0
    normalized = (ink - 0.1307) / 0.3081
    return normalized[None, None, :, :].astype(np.float32)


def preprocess_binary(image: Image.Image, mode: str = "score") -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    if mode == "score":
        red = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        blue = rgb[:, :, 2].astype(np.int16)
        red_mask = (red > 80) & (red > green + 18) & (red > blue + 18)
        if int(red_mask.sum()) >= 20:
            return remove_small_components(red_mask, min_area=8)
    gray = np.asarray(image.convert("L"))
    threshold = max(65, min(185, int(np.percentile(gray, 18)) + 18))
    return remove_small_components(gray < threshold, min_area=12 if mode == "student" else 8)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    visited = np.zeros(mask.shape, dtype=bool)
    output = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    next_y, next_x = y + dy, x + dx
                    if 0 <= next_y < height and 0 <= next_x < width and mask[next_y, next_x] and not visited[next_y, next_x]:
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
        if len(points) >= min_area:
            for y, x in points:
                output[y, x] = True
    return output


def segment_glyphs(image: Image.Image, min_width: int = 3, mode: str = "score") -> list[Glyph]:
    mask = preprocess_binary(image, mode=mode)
    if not mask.any():
        return []
    ys, xs = np.where(mask)
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    cropped = mask[top : bottom + 1, left : right + 1]
    active = cropped.sum(axis=0) > max(1, int(cropped.shape[0] * 0.015))
    active = close_1d(active, max_gap=max(2, int(cropped.shape[1] * 0.012)))
    glyphs: list[Glyph] = []
    for start, end in bool_runs(active):
        if end - start + 1 < min_width:
            continue
        sub = cropped[:, start : end + 1]
        sub_y, sub_x = np.where(sub)
        if len(sub_x) == 0:
            continue
        x0 = left + start + int(sub_x.min())
        x1 = left + start + int(sub_x.max()) + 1
        y0 = top + int(sub_y.min())
        y1 = top + int(sub_y.max()) + 1
        height, width = y1 - y0, x1 - x0
        is_dot = height <= image.height * 0.22 and width <= image.width * 0.12 and y0 > image.height * 0.48
        glyphs.append(Glyph(image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1), int(sub.sum()), is_dot))
    return merge_oversegmented(glyphs, image)


def bool_runs(active: np.ndarray) -> list[tuple[int, int]]:
    indexes = np.where(active)[0]
    if len(indexes) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indexes[0])
    for raw in indexes[1:]:
        current = int(raw)
        if current - previous > 1:
            runs.append((start, previous))
            start = current
        previous = current
    runs.append((start, previous))
    return runs


def close_1d(active: np.ndarray, max_gap: int) -> np.ndarray:
    runs = bool_runs(active)
    closed = np.zeros_like(active, dtype=bool)
    if not runs:
        return closed
    start, end = runs[0]
    for current_start, current_end in runs[1:]:
        if current_start - end <= max_gap:
            end = current_end
        else:
            closed[start : end + 1] = True
            start, end = current_start, current_end
    closed[start : end + 1] = True
    return closed


def merge_oversegmented(glyphs: list[Glyph], image: Image.Image) -> list[Glyph]:
    if len(glyphs) <= 1:
        return glyphs
    merged: list[Glyph] = []
    index = 0
    median_width = np.median([glyph.bbox[2] - glyph.bbox[0] for glyph in glyphs])
    while index < len(glyphs):
        current = glyphs[index]
        if index + 1 < len(glyphs) and not current.is_dot:
            following = glyphs[index + 1]
            gap = following.bbox[0] - current.bbox[2]
            current_width = current.bbox[2] - current.bbox[0]
            if gap <= max(2, median_width * 0.18) and current_width < median_width * 0.55 and not following.is_dot:
                box = (
                    min(current.bbox[0], following.bbox[0]),
                    min(current.bbox[1], following.bbox[1]),
                    max(current.bbox[2], following.bbox[2]),
                    max(current.bbox[3], following.bbox[3]),
                )
                merged.append(Glyph(image.crop(box), box, current.area + following.area, False))
                index += 2
                continue
        merged.append(current)
        index += 1
    return merged


def cleanup_numeric_text(text: str) -> str:
    value = "".join(character for character in text if character.isdigit() or character == ".")
    if value.count(".") > 1:
        first = value.find(".")
        value = value[: first + 1] + value[first + 1 :].replace(".", "")
    if value.startswith("."):
        value = "0" + value
    if value.endswith("."):
        value = value[:-1]
    return value
