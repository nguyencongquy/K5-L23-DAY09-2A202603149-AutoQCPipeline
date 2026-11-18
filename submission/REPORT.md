# Báo cáo Day 09 — Nguyễn Công Quý, 2A202603149

## 1. Kết luận trước

**Hold.** Pipeline nhận 205/224 ảnh sau khi loại 19 ảnh ở bước intake và xuất bản nháp gồm 205 ảnh với 753 box. Sau 80 lượt duyệt mô phỏng, còn 0 lỗi bắt buộc và 38/205 ảnh chưa duyệt còn cảnh báo (18,54%), vượt ngưỡng 10%. Bản xuất được dùng cho thí nghiệm raw/clean.

## 2. Lỗi và phân việc

Các số lấy từ [danh sách v1](reports/02_error_list_v1.csv). Intake có 19 lượt báo trên 224 ảnh. QC vòng đầu có 185 lượt báo trên 205 ảnh được nhận. Một ảnh có thể có nhiều dòng lỗi hoặc cảnh báo.

| Mã lỗi và số lượt | Ai xử lý | Cách xử lý |
|---|---|---|
| `DUP_IMAGE_EXACT` 8, `DUP_IMAGE_NEAR` 2, `LEAK_TEST` 4 | Máy | Loại ảnh trùng hoặc trùng kho val/test |
| `BAD_SIZE` 3, `CORRUPT_IMAGE` 2 | Máy | Loại ảnh quá nhỏ hoặc không giải mã được |
| `DUP_BOX` 7, `BAD_CLASS_ID` 6, `BOX_ZERO` 4, `BAD_FORMAT` 3 | Máy | Xóa dòng lỗi, giữ số dòng gốc trong log |
| `BOX_OUT_OF_IMAGE` 8, `BAD_CLASS_NAME` 2 | Máy | Cắt box theo mép ảnh, chuẩn hóa tên lớp trong YAML |
| `MISSING_LABEL_FILE` 6, `EMPTY_LABEL` 12 | Người | Gán ảnh còn thiếu và kiểm ảnh có file rỗng |
| `POSSIBLE_MISSING` 53, `POSSIBLE_LOOSE_BOX` 28, `POSSIBLE_FALSE_POS` 25, `POSSIBLE_WRONG_CLASS` 18 | Người | Đối chiếu từng cảnh báo YOLO với vật trong ảnh |
| `BOX_TINY` 6, `BOX_HUGE` 3, `CLASS_RARE` 4 | Người | Kiểm kích thước box và độ phủ từng lớp |

Sau autofix, hàng đợi có 125 ảnh. `img_0020` đứng đầu (11,524 điểm) vì có dòng vừa bị xóa, nghi thiếu box và nghi sai lớp. Mô phỏng duyệt 80/125 ảnh. Các ảnh được chọn chứa 94/123 cảnh báo ngữ nghĩa ở v2, gồm 51 thiếu, 18 sai lớp, 13 thừa và 12 lỏng. Đây là số cảnh báo được đưa vào lượt duyệt, chưa phải số lỗi ngữ nghĩa xác nhận. Cả sáu file nhãn thiếu được bổ sung trong lượt mô phỏng.

## 3. Calibration

| Ngưỡng | Trước → sau | Precision / recall trên 39 ảnh calib | Lý do |
|---|---|---|---|
| `duplicates.phash_dist` | 2 → 4 | Gần trùng 1,00/0,50 → 1,00/0,67 | Bắt 4/6 thay vì 3/6, không thêm FP. Chọn mốc nhỏ nhất đạt 4/6 vì ảnh bị loại tự động. |
| `second_opinion.conf` | 0,25 → 0,60 | Thiếu 0,50/1,00 → 0,57/1,00, sai lớp 0,56/0,83 → 0,62/0,83, box lỏng 0,29/0,40 → 0,20/0,20 | Giữ 8/8 ca thiếu và 5/6 ca sai lớp, giảm FP của hai nhóm. Chấp nhận mất một ca box lỏng trên calib. |

Hai cặp raw `0042–0076` và `0037–0143` có pHash cách 0 dù kích thước khác. Đối chiếu ảnh thu về 128×128 cho sai khác RGB trung bình 1,32 và 1,81/255. Các ca gần trùng này dễ hơn hai ca còn lọt trên calib. Vì thế precision 1,00 trên calib chỉ phản ánh mẫu nhỏ đã thử, đặc biệt khi tăng ngưỡng để bắt thêm ảnh biến đổi mạnh.

## 4. Cửa chặn và điểm mù

Chạy `python -m pipeline.run --budget 0 --work work/run_nb --reports work/rep_nb --export-dir work/exp_nb`. Còn sáu lỗi thiếu file nhãn, trả **Rework**, mã **20**, không sinh manifest. Hai lần chạy với ngân sách chuẩn cho cùng quyết định, số liệu và nội dung export.

Đối chiếu tọa độ raw xác nhận box trùng ở `img_0001` dòng 3 và `img_0006` dòng 2, cùng box tràn biên 2,30% ở `img_0030` dòng 9. Hai điểm mù có số đo từ calib: pHash bỏ sót **2/6** ảnh gần trùng ở ngưỡng 4, và checker bỏ sót **4/5** ca box lỏng ở confidence 0,60. Dùng đối chiếu trực quan cho ảnh biến đổi mạnh hoặc box còn hợp lệ về cú pháp.

## 5. Train raw vs clean

Bốn lượt chạy trên Kaggle Tesla T4 dùng cùng YOLO26n, 20 epoch, ảnh 640 px, batch 16 và seed 0, 1. Mỗi lượt chọn checkpoint bằng 40 ảnh val và đo lần cuối trên cùng 100 ảnh test. Ultralytics đọc 224 ảnh đầu vào của `raw` rồi loại 11 ảnh hoặc nhãn lỗi, nên số ảnh thực sự dùng là 213. Với `clean`, cả 205 ảnh đầu vào được dùng.

| Biến thể | mAP50 | mAP50–95 | Precision | Recall |
|---|---:|---:|---:|---:|
| `raw` | 0,726 ± 0,000 | 0,512 ± 0,004 | 0,764 ± 0,061 | 0,649 ± 0,028 |
| `clean` | 0,700 ± 0,030 | 0,486 ± 0,013 | 0,715 ± 0,073 | 0,615 ± 0,017 |

Các giá trị là trung bình ± độ lệch chuẩn mẫu giữa hai seed. Chênh lệch mAP50 `clean − raw` trung bình là **−0,0254**. Chênh lệch theo seed là −0,0468 và −0,0040, còn độ lệch chuẩn của `clean` là 0,0305. Kết quả quan sát được nghiêng về `raw`, nhưng với hai seed và mức dao động này, chưa đủ bằng chứng để kết luận QC làm giảm hiệu năng ổn định. [Bảng và kết quả từng lượt](results/).

Hai giả thuyết cần kiểm thêm:

- Intake phát hiện bốn ảnh `raw` trùng hoặc gần trùng ảnh test. Rò rỉ có thể làm điểm `raw` cao hơn. Có thể đánh giá lại các checkpoint đã khóa trên cùng 96 ảnh test còn lại để đo độ nhạy với bốn ca này.
- AP50 của lớp `truck` giảm từ trung bình 0,648 xuống 0,559. Thay đổi số mẫu hoặc chất lượng box lớp này có thể góp phần làm điểm giảm. Cần đối chiếu số ảnh và box `truck` thực sự vào train, rà cảnh báo còn lại và chạy thêm seed trên một tập test độc lập.

## 6. Quyết định và ghi chú triển khai

Giữ **Hold**. Với 205 ảnh và 0 lỗi bắt buộc, Release cần tối đa 20 ảnh chưa duyệt còn cảnh báo. Cần xử lý ít nhất 18 trong 38 ảnh đang bị cờ và chạy lại QC. Ưu tiên ca thiếu box và sai lớp, sau đó kiểm các ca box lỏng.

- Prompt chính yêu cầu đọc dòng YOLO và cấu hình, trả `Issue` đúng mã và số dòng, chỉ ghi bản làm việc và log trước/sau.
- Kiểm bằng 79 test gốc và hai test bổ sung, lỗi chèn trên 39 ảnh calib, phép thử gate ngân sách 0 và chạy lặp trên lô raw.
- Bản nháp autofix tạo box trùng sau khi cắt biên trong ca thử bổ sung. Bản sửa so IoU của box vừa cắt với các box cùng lớp và ghi một lần xóa cho dòng trùng.

**Kiểm bổ sung.** Chạy `python -m pipeline.checks.box_aspect work/run/ds --max-ratio 6 --out submission/evidence/box_aspect.csv` sau `make run`. Check dùng tỉ lệ cạnh dài/cạnh ngắn theo pixel, có hai test riêng và gắn cờ 3 box trên 205 ảnh ở ngưỡng 6. Tỉ lệ lớn có thể là vật bị che hoặc bị cắt ở mép ảnh. Check chỉ dùng để chọn ca xem lại, ngoài cổng QC chính thức.

**CVAT.** Import YOLO 1.1 cho 20 ảnh đầu hàng đợi vào job 21 và áp dụng `Width < 20`. Kết quả quan sát trong giao diện: `img_0020` 3, `img_0030` 3, `img_0031` 0, `img_0053` 1 và `img_0135` 2. Danh sách v3 không có `BOX_TINY` trên năm ảnh này vì dùng cạnh ngắn dưới 12 px, còn CVAT dùng chiều rộng dưới 20 px. Người gán bật `can_review` khi cần người khác xác minh box. Pipeline chuyển ca khó sang hàng đợi qua mã cảnh báo và điểm ưu tiên. [Ảnh bộ lọc trên img_0135](evidence/cvat_width_filter.png).
