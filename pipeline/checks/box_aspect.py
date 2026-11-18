"""Kiểm bổ sung: cảnh báo box quá dài hoặc quá cao theo tỉ lệ pixel."""

import argparse
from pathlib import Path

from ..common import WARN, Dataset, Issue, load_config, parse_yolo_line, write_issues_csv


def run(ds, max_ratio):
    issues = []
    for path in ds.image_files():
        size = ds.image_size(path.stem)
        if size is None:
            continue
        for line, text in ds.read_label_lines(path.stem) or []:
            try:
                cls, _, _, width, height = parse_yolo_line(text)
            except (ValueError, OverflowError):
                continue
            if not 0 <= cls < len(ds.classes) or width <= 0 or height <= 0:
                continue
            w_px, h_px = width * size[0], height * size[1]
            ratio = max(w_px, h_px) / min(w_px, h_px)
            if ratio > max_ratio:
                issues.append(Issue("box_aspect", WARN, "BOX_ASPECT_EXTREME", path.stem,
                                    "Box có tỉ lệ hai cạnh cao, cần xem lại", line,
                                    {"ratio": round(ratio, 3), "max_ratio": max_ratio}))
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-ratio", type=float, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.max_ratio >= 1:
        parser.error("--max-ratio phải từ 1 trở lên")
    ds = Dataset(args.dataset, load_config()["classes"])
    issues = run(ds, args.max_ratio)
    write_issues_csv(issues, args.out)
    print(f"{len(issues)} cảnh báo trên {len(ds.image_files())} ảnh → {args.out}")


if __name__ == "__main__":
    main()
