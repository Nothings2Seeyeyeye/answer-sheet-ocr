from __future__ import annotations

import math
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

FIELD_LABELS = {
    "name": "姓名",
    "student_id": "学号",
    "score_1": "一",
    "score_2_1": "二(1)",
    "score_2_2": "二(2)",
    "score_2_3": "二(3)",
    "score_2_4": "二(4)",
    "score_2_5": "二(5)",
    "score_2_6": "二(6)",
    "total": "总分",
}

FIELD_NAMES = list(FIELD_LABELS)
SCORE_FIELDS = FIELD_NAMES[2:]
NAME_KEYWORD = "姓名"
ID_KEYWORD = "学号"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")


@dataclass(frozen=True)
class ObjectBox:
    text: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def cx(self) -> float:
        return (self.xmin + self.xmax) / 2

    @property
    def cy(self) -> float:
        return (self.ymin + self.ymax) / 2


def clean_digits(text: str) -> str:
    return "".join(ch for ch in str(text) if ch.isdigit())


def is_number_text(text: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", str(text).strip()))


def parse_xml(xml_path: Path) -> tuple[int, int, list[ObjectBox]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0
    objects: list[ObjectBox] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        coords = [int(round(float(box.findtext(k, "0")))) for k in ("xmin", "ymin", "xmax", "ymax")]
        objects.append(ObjectBox(name, *coords))
    return width, height, objects


def assign_field_roles(objects: Iterable[ObjectBox], image_width: int) -> dict[str, ObjectBox]:
    objs = list(objects)
    assigned: dict[str, ObjectBox] = {}
    name_label = next((o for o in objs if o.text == NAME_KEYWORD), None)
    id_label = next((o for o in objs if o.text == ID_KEYWORD), None)

    if name_label is not None:
        candidates = [
            o
            for o in objs
            if o.text not in {NAME_KEYWORD, ID_KEYWORD}
            and not is_number_text(o.text)
            and o.cx > name_label.cx
            and abs(o.cy - name_label.cy) < max(90, image_width * 0.04)
        ]
        if candidates:
            assigned["name"] = min(candidates, key=lambda o: abs(o.cy - name_label.cy) + max(0.0, o.cx - name_label.cx) * 0.01)

    if id_label is not None:
        candidates = [
            o
            for o in objs
            if len(clean_digits(o.text)) >= 8
            and o.cx > id_label.cx
            and abs(o.cy - id_label.cy) < max(90, image_width * 0.04)
        ]
        if candidates:
            assigned["student_id"] = min(candidates, key=lambda o: abs(o.cy - id_label.cy) + max(0.0, o.cx - id_label.cx) * 0.01)

    score_objs = [o for o in objs if is_number_text(o.text) and o.cx > image_width * 0.48]
    score_objs = sorted(score_objs, key=lambda o: (o.cx, o.cy))
    if len(score_objs) >= len(SCORE_FIELDS):
        for field, obj in zip(SCORE_FIELDS, score_objs[: len(SCORE_FIELDS)]):
            assigned[field] = obj
    return assigned


def image_for_stem(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def split_for_stem(stem: str, split_root: Path) -> str:
    train = split_root / "training_set" / f"{stem}.jpg"
    test = split_root / "test_set" / f"{stem}.jpg"
    if train.exists():
        return "train"
    if test.exists():
        return "val"
    return "unknown"


def build_field_table(dataset_dir: Path, split_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    for xml_path in sorted(label_dir.glob("*.xml")):
        width, height, objects = parse_xml(xml_path)
        image_path = image_for_stem(image_dir, xml_path.stem)
        if image_path is None:
            continue
        assigned = assign_field_roles(objects, width)
        split = split_for_stem(xml_path.stem, split_root)
        for field in FIELD_NAMES:
            obj = assigned.get(field)
            rows.append(
                {
                    "split": split,
                    "stem": xml_path.stem,
                    "file_name": image_path.name,
                    "image_path": str(image_path.resolve()),
                    "image_width": width,
                    "image_height": height,
                    "field": field,
                    "class_id": FIELD_NAMES.index(field),
                    "field_label": FIELD_LABELS[field],
                    "gt_text": obj.text if obj else "",
                    "xmin": obj.xmin if obj else math.nan,
                    "ymin": obj.ymin if obj else math.nan,
                    "xmax": obj.xmax if obj else math.nan,
                    "ymax": obj.ymax if obj else math.nan,
                    "assigned": obj is not None,
                }
            )
    return pd.DataFrame(rows)


def yolo_line(row: pd.Series) -> str:
    width = float(row["image_width"])
    height = float(row["image_height"])
    xmin = float(row["xmin"])
    ymin = float(row["ymin"])
    xmax = float(row["xmax"])
    ymax = float(row["ymax"])
    x = ((xmin + xmax) / 2) / width
    y = ((ymin + ymax) / 2) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return f"{int(row['class_id'])} {x:.8f} {y:.8f} {w:.8f} {h:.8f}"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)
