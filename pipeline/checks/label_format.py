"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from collections import defaultdict

from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line


def run(ds, cfg):
    rules = cfg["label_format"]
    issues = []
    for label in ds.label_files():
        stem = label.stem
        size = ds.image_size(stem)
        previous_boxes = defaultdict(list)
        for line, text in ds.read_label_lines(stem) or []:
            try:
                cls, cx, cy, width, height = parse_yolo_line(text)
            except (ValueError, OverflowError) as exc:
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", stem, str(exc), line))
                continue

            if not 0 <= cls < len(ds.classes):
                issues.append(Issue("label_format", ERROR, "BAD_CLASS_ID", stem, f"ID lớp {cls} không hợp lệ", line))
                continue
            if width <= 0 or height <= 0:
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", stem, "Box có cạnh không dương", line))
                continue

            box = (cx, cy, width, height)
            overflow = max(
                width / 2 - cx, height / 2 - cy,
                cx + width / 2 - 1, cy + height / 2 - 1,
            )
            if overflow > rules["overflow_tol"]:
                issues.append(Issue("label_format", ERROR, "BOX_OUT_OF_IMAGE", stem, "Box tràn ảnh", line,
                                    {"overflow": overflow}))
            if any(iou_xywh(box, previous) > rules["dup_box_iou"] for previous in previous_boxes[cls]):
                issues.append(Issue("label_format", ERROR, "DUP_BOX", stem, "Box trùng một dòng trước cùng lớp", line))
            previous_boxes[cls].append(box)
            if size and min(width * size[0], height * size[1]) < rules["min_box_px"]:
                issues.append(Issue("label_format", WARN, "BOX_TINY", stem, "Cạnh ngắn dưới ngưỡng pixel", line))
            if width * height > rules["max_box_frac"]:
                issues.append(Issue("label_format", WARN, "BOX_HUGE", stem, "Box chiếm quá nhiều diện tích ảnh", line))

    return issues
