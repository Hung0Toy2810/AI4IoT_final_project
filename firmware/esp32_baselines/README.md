# Frozen E1–E5 checkpoints on ESP32

The implementation, exporters and replay program are in section 16 of `notebooks/har_complete_pipeline.ipynb`. Files in this directory are generated build inputs. Keep the original checkpoints unchanged.

## Models and runtime

- E1: all 100 original Random Forest trees. Each tree uses lossless DEFLATE; unpack one tree into the shared arena, traverse it and accumulate float64 leaf probabilities. Features and split inputs remain float32. Directed-down float32 thresholds preserve sklearn decisions for float32 feature inputs.
- E2–E5: all seeds 42, 2026 and 3407. Native FP32 CNNs with BatchNorm folded into Conv at export. Architectures and layer order match the checkpoint.
- Q1: the existing E3 INT8 TFLite file. Integer kernels follow TFLite 2.15.1 reference rounding; the default optimized CPU runtime has a separate comparison table.

Build uses `espressif32@6.12.0`, Arduino 2.0.17, C++17-compatible kernels, `-O3 -ffp-contract=off`, CPU 240 MHz, flash DIO 20 MHz. The pinned SDK includes a 40-MHz ELF bootloader; the notebook generates a DIO 20-MHz boot image from that ELF with esptool. Board: ESP32-D0WDQ6, flash 4 MiB, no PSRAM. A single-app partition accommodates the complete forest and all CNN constants. There is no OTA partition. A 64-KiB coredump partition captures future firmware exceptions. Compute kernels are placed in IRAM. Each RF compressed tree is read with `esp_flash_read` into a 32-KiB DRAM buffer before ROM decompression; flash reading is included in RF inference timing. Loop task stack is 16 KiB; arena/heap checks and RTC trace run outside compute timers.

## Input and timing

Window 8 seconds, hop 2 seconds. Raw PPG has 512 samples at 64 Hz; wrist ACC has 3 × 256 samples at 32 Hz and is already in g. Native preprocessing uses float64 zero-phase SOS decimation, then float32 arrays and Train-only normalization. STFT: periodic Hann 64, hop 4, FFT 64, no padding, spectrum scaling, 0–12 Hz crop, log1p magnitude, physical-axis interpolation to 32 × 32.

The classifier parity test uses all 9,420 cached Test S13–S15 windows. End-to-end latency uses the same 1,000 fixed IDs as the CPU outline benchmark, after 50 warm-up windows. Each checkpoint repeats its own full preprocessing; ACC-only skips the PPG filter. CPU and MCU compile the same pipeline source. MCU runs 14 checkpoints in fixed order per window; CPU measures each checkpoint in its own loop. The combined fixture may affect cache behavior. Timers exclude Python, USB, disk, model initialization and 8-second acquisition. Total latency is timed directly. Empty stages may include timer-read overhead.

RF batches amortize decompression only for prediction parity. The single-window timing includes decompression of all 100 trees. Do not report RF batch time as single-window inference latency.

## Binary protocol

UART: 460800 baud; upload: 460800 baud. USB replay needs no sensors. Input is a little-endian 12-byte header: magic `0x42524148`, command, argument. Commands with payload append payload CRC32.

| Command | Payload | Response |
|---|---|---|
| 0 | None | `@INFO` JSON with board identity, checkpoint hashes and memory counters |
| 1 | 4 × 256 float32 cached raw samples | 13 CNN/INT8 records; normalization/STFT are shared; only inference/postprocessing timing is meaningful |
| 2 | 512 float64 raw PPG + 3 × 256 float32 ACC | 14 independent full-pipeline records, including E1 |
| 3 | argument × 38 float32 RF features; argument ≤ 64 | argument × 8 float64 probabilities |

Binary response header: four uint32 values, magic `0x53424d48`, command, body length, status; followed by body and CRC32. A record is eight uint32 values (filter, normalization, representation, quantization, inference, postprocessing, total in µs; label) and eight float32 outputs. RF outputs in command 2 are probability summaries; its label is selected from float64 probabilities before logging them as float32.

Earlier firmware sessions encountered IllegalInstruction failures; their root cause remains unconfirmed. The flashed image verified against its build digest. Accuracy records span the configuration history with identical learned weights and numerical kernels; the final full latency benchmark uses DIO 20 MHz, IRAM kernels and RF direct driver reads. Firmware errors stop measurement and preserve diagnostics.

Invalid CRC frames are discarded and retransmitted; recovery logs are retained. Only validated successful responses enter metrics. Latency excludes failed computations and host reconnection/rewarm; it is conditional compute latency, not unconditional service latency. RF failure traces are retained; bounded host retries require model 13 and no reported memory fault. Panics and memory faults stop measurement. Flush/resume checkpoints preserve the accepted prefix during long runs.

## Memory and reproducibility

The activation/tree arena has a 49,152-byte payload and 16 guard bytes (49,168 bytes allocated once). CNN and RF inventories overlap. Linker static RAM, arena capacity, free/minimum heap, stack watermark and whole-process CPU RSS have different scopes; do not add these values or assign the combined firmware size to an individual CNN.

Artifacts are in `artifacts/esp32_baselines`: registry/hashes, CPU and MCU predictions, class/subject metrics, stage timing CSVs, build/device metadata, transport logs and Plotly HTML/PNG/SVG. The original CPU study remains in `artifacts/benchmark`.

After measuring, restore `firmware/esp32_har` with PlatformIO and verify the extension checkpoint hash. The original 4 MiB flash backup remains in `artifacts/esp32/original_flash.bin`.

Reference kernel sources: [TFLite integer arithmetic](https://github.com/tensorflow/tensorflow/blob/v2.15.1/tensorflow/lite/kernels/internal/common.h), [integer average pooling](https://github.com/tensorflow/tensorflow/blob/v2.15.1/tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h).
