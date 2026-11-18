"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict

import yaml

from ..common import ERROR, iou_xywh, parse_yolo_line


DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def clipped_line(text):
    cls, cx, cy, width, height = parse_yolo_line(text)
    left, right = max(0.0, cx - width / 2), min(1.0, cx + width / 2)
    top, bottom = max(0.0, cy - height / 2), min(1.0, cy + height / 2)
    if right <= left or bottom <= top:
        return None
    x, y = (left + right) / 2, (top + bottom) / 2
    return f"{cls} {x:.16g} {y:.16g} {right - left:.16g} {bottom - top:.16g}"


def duplicate_after_clipping(text, changed, kept, threshold):
    try:
        cls, *box = parse_yolo_line(text)
    except (ValueError, OverflowError):
        return False
    if box[2] <= 0 or box[3] <= 0:
        return False
    if any((changed or was_clipped) and iou_xywh(box, prior) > threshold
           for prior, was_clipped in kept[cls]):
        return True
    kept[cls].append((box, changed))
    return False


def fix(ds, issues, cfg):
    by_image = defaultdict(list)
    for issue in issues:
        if issue.severity == ERROR:
            by_image[issue.image].append(issue)

    fixes = []

    def record(image, action, before, after, line=None):
        fixes.append(dict(image=image, line=line, action=action, before=before, after=after))

    for image, errors in sorted(by_image.items()):
        codes = {issue.code for issue in errors}
        if "BAD_CLASS_NAME" in codes:
            before = ds.yaml_path.read_text(encoding="utf-8") if ds.yaml_path.exists() else ""
            content = yaml.safe_load(before) or {}
            if content.get("names") != cfg["classes"]:
                content["names"] = list(cfg["classes"])
                after = yaml.safe_dump(content, allow_unicode=True, sort_keys=False)
                ds.yaml_path.write_text(after, encoding="utf-8")
                record(image, "rename_classes", before, after)

        if codes & REMOVE_CODES or any(code.startswith("DUP_IMAGE_") for code in codes):
            path = ds.image_path(image)
            if path:
                path.unlink()
                ds.label_path(image).unlink(missing_ok=True)
                record(image, "remove_image", path.name, "")
            continue

        label = ds.label_path(image)
        if not label.exists():
            if "MISSING_LABEL_FILE" in codes:
                record(image, "needs_label", "(không có file)", "chuyển người gán")
            continue

        dropped = {issue.line for issue in errors if issue.code in DROP_CODES}
        clipped = {issue.line for issue in errors if issue.code == "BOX_OUT_OF_IMAGE"}
        if not (dropped or clipped):
            continue

        updated = []
        kept = defaultdict(list)
        original = label.read_text(encoding="utf-8", errors="replace").splitlines()
        for line, before in enumerate(original, 1):
            if line in dropped:
                record(image, "drop_line", before, "", line)
                continue
            after = clipped_line(before) if line in clipped else before
            if after is None or (clipped and duplicate_after_clipping(
                after, after != before, kept, cfg["label_format"]["dup_box_iou"]
            )):
                record(image, "drop_line", before, "", line)
                continue
            if after != before:
                record(image, "clip_box", before, after, line)
            updated.append(after)

        if updated != original:
            label.write_text("".join(f"{text}\n" for text in updated), encoding="utf-8")

    return fixes
