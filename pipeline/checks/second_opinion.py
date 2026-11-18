"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def run(ds, cfg, preds=None):
    rules = cfg["second_opinion"]
    if preds is None:
        preds = get_preds(ds, cfg, "checker")
    issues = []
    for path in ds.image_files():
        stem = path.stem
        if stem not in preds:
            continue
        lines = ds.read_label_lines(stem)
        size = ds.image_size(stem)
        if lines is None or size is None:
            continue

        labels = []
        for line, text in lines:
            try:
                cls, *box = parse_yolo_line(text)
            except (ValueError, OverflowError):
                continue
            if 0 <= cls < len(ds.classes) and box[2] > 0 and box[3] > 0:
                labels.append((line, cls, box))

        area = size[0] * size[1]
        for cls, confidence, *box in preds[stem]:
            if confidence < rules["conf"] or box[2] * box[3] * area < rules["min_area_px"]:
                continue
            candidates = [(label_cls, iou_xywh(box, label_box)) for _, label_cls, label_box in labels]
            candidates = [(other, iou) for other, iou in candidates if iou >= rules["iou_missing"]]
            same_class = [iou for other, iou in candidates if other == cls]
            extra = {"conf": confidence}
            if not candidates:
                code, message = "POSSIBLE_MISSING", f"Nghi thiếu box lớp {ds.classes[cls]}"
            elif same_class:
                extra["iou"] = max(same_class)
                if extra["iou"] >= rules["iou_loose"]:
                    continue
                code, message = "POSSIBLE_LOOSE_BOX", "Box cùng lớp có IoU thấp"
            else:
                other, best_iou = max(candidates, key=lambda candidate: candidate[1])
                if best_iou < rules["iou_match"]:
                    continue
                extra["iou"] = best_iou
                code = "POSSIBLE_WRONG_CLASS"
                message = f"Model thấy {ds.classes[cls]}, nhãn ghi {ds.classes[other]}"
            issues.append(Issue("second_opinion", WARN, code, stem, message, extra=extra))

        fp_boxes = [pred[2:] for pred in preds[stem] if pred[1] >= rules["fp_conf"]]
        for line, _, box in labels:
            best_iou = max((iou_xywh(box, pred) for pred in fp_boxes), default=0.0)
            if best_iou < rules["iou_missing"]:
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_FALSE_POS", stem,
                                    "Box nhãn không khớp dự đoán nào", line, {"iou": best_iou}))

    return issues
