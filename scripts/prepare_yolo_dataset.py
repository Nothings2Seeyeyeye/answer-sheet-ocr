from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from field_dataset import FIELD_NAMES, build_field_table, link_or_copy, yolo_line


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LabelImg XML annotations into a YOLO field-detection dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("../labelimg_dataset"))
    parser.add_argument("--split-root", type=Path, default=Path("/root/ocr/data_set"))
    parser.add_argument("--output", type=Path, default=Path("data/field_yolo"))
    args = parser.parse_args()

    out = args.output.resolve()
    table = build_field_table(args.dataset.resolve(), args.split_root.resolve())
    table = table[table["split"].isin(["train", "val"])].copy()
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "field_ground_truth.csv", index=False, encoding="utf-8-sig")

    summary = {
        "images": int(table["stem"].nunique()),
        "rows": int(len(table)),
        "assigned_boxes": int(table["assigned"].sum()),
        "splits": {k: int(v) for k, v in table.groupby("split")["stem"].nunique().to_dict().items()},
        "classes": FIELD_NAMES,
    }

    for split, split_df in table.groupby("split"):
        img_dir = out / "images" / split
        label_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for stem, group in split_df.groupby("stem"):
            image_path = Path(str(group.iloc[0]["image_path"]))
            image_dst = img_dir / image_path.name
            link_or_copy(image_path, image_dst)
            labels = [yolo_line(row) for _, row in group[group["assigned"]].iterrows()]
            (label_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    data_yaml = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(FIELD_NAMES)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
