# ESP32 HAR runtime

Các file firmware được sinh từ phần 15 của `notebooks/har_complete_pipeline.ipynb`. Sửa source trong notebook rồi sinh lại file trước khi build.

Checkpoint mở rộng giữ nguyên SHA256 `d92e77c4c053087c6ee0b1786361d8eb4caf3f2b16a4dabe7acbe29928d5c543`. Runtime C++ FP32 fuse BatchNorm vào Conv; không dùng TorchScript trên MCU và không train lại.

Thiết bị đã nhận diện: ESP32-D0WDQ6 rev 1.1, MAC `f8:b3:b7:a2:56:78`, flash 4 MB, CPU 240 MHz, không có PSRAM. Cổng macOS: `/dev/cu.usbserial-0001`.

Từ thư mục gốc project:

```sh
.venv/bin/python -m pip install -r requirements.esp32.txt
.venv/bin/pio run --project-dir firmware/esp32_har
.venv/bin/pio run --project-dir firmware/esp32_har --target upload
```

Cell notebook sao lưu flash trước khi nạp. Bản sao gốc nằm ở `artifacts/esp32/original_flash.bin`; checksum và MAC nằm ở `backup.json`.

## Giao tiếp USB

UART 921600 baud. Header little-endian `<III`: magic `0x50524148`, command, argument. Payload theo sau và kết thúc bằng CRC32 IEEE uint32. Phản hồi là JSON một dòng có tiền tố `@INFO`, `@RESULT` hoặc `@BENCH`; lỗi giao thức có tiền tố `@ERROR`.

| Command | Payload | Argument |
|---|---|---|
| 0 | Không có | 0; đọc thông tin board |
| 1 | 4×256 float32, channel-major | 1 để reset EMA, 0 để tiếp tục |
| 2 | 512 float64 PPG + 3×256 float32 ACC | Reset EMA như command 1 |
| 3 | Payload như command 1 | Số lần benchmark, 1–1000 |
| 4 | Payload như command 2 | Số lần benchmark, 1–1000 |

PPG 64 Hz, ACC 32 Hz trong g, cửa sổ 8 giây. Hạ mẫu PPG q=2 bằng SOS Chebyshev I bậc 8 và odd padding 27 mẫu; forward/backward chỉ dùng dữ liệu trong cửa sổ hiện có. Classifier nhận 4×256 mẫu ở 32 Hz và chuẩn hóa bằng mean/std Train. Runtime SOS dùng float64, cache dataset gốc dùng float32; sai lệch được ghi riêng. Temperature = 0.8096222555193905, EMA alpha = 0.35. Reset EMA khi đổi subject hoặc thiếu cửa sổ; hop = 2 giây.

Benchmark dùng 50 warm-up; không tính USB, I/O cảm biến hoặc 8 giây thu cửa sổ. Dữ liệu được phát lại từ tập test; chưa nối cảm biến trực tiếp. Các tệp CSV/JSON, confusion matrix và biểu đồ Plotly nằm ở `artifacts/esp32/`.

RAM tĩnh do linker báo, workspace model và heap runtime là các phép đo khác nhau. Không cộng các số này để suy ra tổng bộ nhớ. Stack high-water ghi giá trị của ESP-IDF; đây là phần stack nhỏ nhất còn trống, không phải peak heap.
