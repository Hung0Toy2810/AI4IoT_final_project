# Nhận dạng hoạt động từ PPG và ACC trên ESP32

Đồ án phân loại tám hoạt động thường nhật từ tín hiệu PPG và gia tốc kế cổ tay của bộ dữ liệu PPG-DaLiA. Pipeline duy nhất nằm trong `notebooks/har_complete_pipeline.ipynb` và đi từ tải dữ liệu, tiền xử lý, huấn luyện E1–E6, lượng tử hóa Q1 đến đánh giá trên ESP32.

## Thiết kế thí nghiệm

- Train: S1–S10; Validation: S11–S12; Test: S13–S15.
- Cửa sổ 8 giây, bước nhảy 2 giây.
- PPG: 64 Hz, hạ mẫu về 32 Hz; ACC: 32 Hz; nhãn hoạt động: 4 Hz.
- E1: Random Forest, PPG+ACC, 38 đặc trưng.
- E2: CNN 1-D, PPG+ACC.
- E3: CNN 2-D STFT, PPG+ACC.
- E4: CNN 2-D STFT, chỉ ACC.
- E5: CNN 2-D STFT, chỉ PPG.
- E6: CNN 1-D MultiKernel, Mixup và causal EMA.
- Q1: E3 seed 42 được lượng tử hóa INT8.

Checkpoint và tham số EMA của E6 được chọn trên Validation. Test chỉ dùng để đánh giá sau khi đã chốt cấu hình.

## Kết quả chính trên ESP32

| Model | Accuracy (%) | Macro F1 (%) |
|---|---:|---:|
| E1 | 73,09 | 75,87 |
| E2 | 75,42 | 78,48 |
| E3 | 73,85 | 75,90 |
| E4 | 70,98 | 73,06 |
| E5 | 44,56 | 44,88 |
| E6 | 81,46 | 85,31 |
| Q1 | 71,03 | 73,74 |

E2–E5 là trung bình của ba checkpoint. E1, E6 và Q1 dùng một checkpoint. E6 đạt kết quả cao nhất nhưng mới được đánh giá trên ba người Test. Nhãn Làm việc còn yếu với Recall 50,03%; 43,10% cửa sổ của nhãn này bị nhận thành Nghỉ trưa/ăn trưa.

Q1 giảm 72,31% dung lượng trọng số và 16,31% thời gian inference so với E3/42, nhưng Macro F1 giảm 1,73 điểm. Thời gian toàn pipeline chỉ giảm 4,07% vì STFT vẫn dùng số thực.

## Chạy notebook

Yêu cầu Python 3.11. Tạo môi trường và cài dependency:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install -r requirements.esp32.txt
python -m pip install -r requirements.report.txt
```

Mở `notebooks/har_complete_pipeline.ipynb`, chọn kernel của môi trường trên và chạy theo thứ tự từ đầu đến cuối. Notebook tự tải PPG-DaLiA khi dữ liệu chưa có. Dữ liệu tải xuống, checkpoint và artifact sinh ra được loại khỏi Git vì dung lượng lớn.

## Firmware

- `firmware/esp32_baselines`: E1–E5 và Q1 dùng cho đánh giá chung.
- `firmware/esp32_har`: firmware riêng của E6.

Cả hai dùng PlatformIO và board `esp32dev`. Thí nghiệm đã chạy trên ESP32-D0WDQ6, 240 MHz, flash 4 MiB, không có PSRAM. Dữ liệu Test đã lưu được gửi qua USB; chưa đo trực tiếp bằng cảm biến và chưa đo điện năng.

## Tác giả

Lương Nguyễn Thành Hưng — MSSV 23110111
Học phần: Trí tuệ nhân tạo cho IoT
Giảng viên hướng dẫn: ThS. Hồ Nhựt Minh
