# DocAudit — đánh giá kiểu giảng viên (13/11/2026)

Mọi con số dưới đây là **kết quả nhóm tự đo** trên máy Mac (PaddleOCR 2.10, CPU), chạy lại được bằng lệnh ghi bên cạnh. Không có số nào lấy từ bài báo.

## 1. Đo được gì

| Thước đo | Trước 13/11 | Sau 13/11 | Lệnh |
|---|---|---|---|
| Đọc đúng TOTAL trên ảnh SROIE (so với nhãn `total` của dataset) | 45/107 = **42.1%** (28 sai số, 34 không tìm thấy) | 94/105 = **89.5%** (4 sai, 7 không tìm thấy) | `python -m eval.real_world sroie` |
| Đọc đúng TOTAL trên 20 receipt nhiều kiểu trình bày (tự dựng, biết đáp án) | chưa đo bản cũ | 20/20 | `python -m eval.real_world varied` |
| Receipt **sạch** bị gắn ≥ 1 cảnh báo (báo động giả) — SROIE | chưa đo theo cách này | 22/105 = **21%** | như trên (`receipts_with_warning`) |
| Receipt sạch bị cảnh báo — bộ nhiều kiểu (19 receipt có tổng) | — | 0/19 | như trên |
| TOTAL **kiểm tra được** (đủ dòng hàng để so) — SROIE | — | 40/105 = 38%; 57 "chưa đủ dữ liệu", 7 không có TOTAL, 1 "suy ra" | như trên (`total_check`) |
| Phát hiện lỗi chèn có kiểm soát (JSON, `eval/results/metrics.json`) | sai tổng P 0.35 / R 0.30 / F1 0.32 / FPR 0.089 · trùng 1.00/1.00 · giá bất thường P 1.00 / R 0.42 | **không đổi** (đã chạy lại trong bản sao, kết quả giống hệt) | `python -m codes.run_full_eval` |
| Test tự động | 120 | **170 passed, 0 skip** | `python -m pytest -q tests/` |

Ảnh thật nhóm tự chụp/tải (5 ảnh, chạy trước khi bộ nhớ tạm bị xóa nên ảnh gốc không còn để chạy lại):
Green Supermarket (.AVIF) — trước: "Extraction failed"; sau: 14 dòng, TOTAL 27.35, khớp. El Pollo Loco (Mỹ, không in chữ TOTAL) — sau: TOTAL 13.12 = hàng 12.06 + thuế 1.06, đánh dấu "suy ra, cần duyệt". Soon Huat — 327.00 đúng. Luckin (tiếng Trung, không in giá) và Meituan (tiếng Trung, ảnh cắt mất phần tổng) — sau: đọc được chữ Trung bằng model `ch`, báo đúng là không có TOTAL.

## 2. Nguyên nhân lỗi đã tìm ra và sửa

1. **Không mở được ảnh AVIF / ảnh iPhone 24 MP** — `cv2.imread` không đọc AVIF. Nay đọc bằng Pillow, xoay theo EXIF, thu về ≤ 2400 px; file hỏng báo lỗi dễ hiểu.
2. **Chọn nhầm TOTAL** (nguyên nhân chính của 42%): lấy dòng TOTAL thấp nhất nên trúng bảng *GST SUMMARY* ("Total: 106.10 6.37" → lấy tiền thuế); bỏ qua "Total Sales (Inclusive of GST)" vì có chữ GST; không hiểu làm tròn ("60.31 → −0.01 → RM 60.30"), "Tota1", "lnclusive", "TOTAL ROUNDED", "Amt Due", "Balance Due", "Grand Total", nhãn và số nằm hai dòng; tiền "$", "¥17.4", "1,234.50", "RH" (OCR đọc nhầm RM). Nay: bỏ vùng dưới "GST Summary", ưu tiên dòng trước CASH/CHANGE/VISA, xử lý làm tròn.
3. **Receipt tiếng Trung**: model OCR chỉ có tiếng Anh. Nay khi không tìm thấy TOTAL thì đọc lại bằng model `ch` (đọc cả Anh và số), giữ lần đọc tốt hơn; nhãn 合计/总计/应付/实付, 找零/现金.
4. **Không có nhãn TOTAL**: tìm dòng có số tiền = hàng + thuế in trên receipt, **không coi là đã xác minh** (đã chọn bằng chính phép cộng) → TOTAL "suy ra", người dùng phải Accept.
5. **Dòng hàng**: "2 APPLE 1.00", "1PEAR" (OCR dính số), "2 Taco @ 1.89", "鸡肉云吞 x1", tên ở dòng trên "2 x 2.50 5.00", mã hàng bị đọc thành số lượng (801 × 6.00), cột 0.00 cuối dòng, đơn giá in đã làm tròn (17 × 0.90 = 15.25), dòng tổng phụ không nhãn ("Food 12.06").
6. **Loại tiền**: nhận ¥/$/RM; receipt không phải MYR vẫn được kiểm tra số học nhưng **không so với bảng giá MYR** (tránh cảnh báo giá sai); thông báo dùng đúng ký hiệu tiền.
7. **Thuế không rõ cách tính** (có "Total GST" nhưng không có SUBTOTAL): chỉ báo sai khi TOTAL không khớp *cả hai* cách hiểu.

## 3. Chấm điểm

| Tiêu chí | Trọng số | Điểm | Lý do |
|---|---|---|---|
| Xác định bài toán, phạm vi (theo góp ý GV) | 10 | 9 | Đúng phạm vi: một loại chứng từ, OCR có sẵn, ít loại bất thường, cảnh báo ≠ gian lận. |
| Pipeline hoàn chỉnh + demo chạy thật | 25 | 22 | Ảnh → OCR → dòng hàng → kiểm tra → giải thích + vùng ảnh → duyệt → xuất CSV/JSON, có tài khoản, chạy được qua web. Trừ: phụ thuộc máy cá nhân + tunnel, OCR CPU 5–20 s/ảnh. |
| Độ tin cậy trích xuất | 20 | 13 | TOTAL 89.5% SROIE là tốt; nhưng chỉ 38% receipt SROIE đủ dòng hàng để kiểm tra tổng — trích dòng hàng vẫn là nút thắt. |
| Phát hiện + giải thích | 15 | 10 | Giải thích có phép tính và vùng ảnh, có trạng thái "chưa đủ dữ liệu"/"suy ra" thay vì đoán. Trừ: recall sai tổng 0.30; "duplicate" 1.00 là gần như hiển nhiên do cách chèn lỗi; báo động giả 21% trên receipt sạch. |
| Phương pháp đánh giá, trung thực | 15 | 11 | Tách extraction / detection, có test sạch, có bộ nhiều kiểu, ghi rõ nguồn số. Trừ: bộ nhiều kiểu là ảnh tự dựng, chưa có bộ ảnh thật có nhãn; bảng metrics.json lệch 1 đơn vị (27 dương tính vs 26 nhãn) chưa giải thích. |
| Chất lượng kỹ thuật | 10 | 9 | 170 test, chạy lại đo được bằng một lệnh, mật khẩu băm, không lộ file dự án. |
| Giao diện, trình bày | 5 | 5 | Trang giới thiệu, đăng ký, đăng nhập, app cùng một bộ màu và kiểu thẻ. |
| **Tổng** | 100 | **79** | **B+** |

## 4. Trước Stage 3 nên làm

1. **Bộ ảnh thật có nhãn** (30–50 receipt nhóm tự chụp, có total đúng) — đây là câu GV sẽ hỏi đầu tiên; công cụ đo đã sẵn (`eval/real_world.py`, thêm thư mục + `expected.json`).
2. **Trích dòng hàng**: số lượng dạng "2.0"/"10.00NOS", cột "giá + GST" (55.00 → 58.30) còn gây 22 receipt báo giả; đây là việc có lợi nhất cho cả độ tin cậy lẫn số receipt kiểm tra được tổng.
3. Giải thích dòng lệch 27/26 trong `metrics.json` trước khi đưa bảng lên slide.
4. Khi trình bày: nói rõ "duplicate P = R = 1.0" là trên lỗi chèn chép y nguyên, và báo động giả 21% trên receipt sạch.
