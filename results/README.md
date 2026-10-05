# ESP32 latency comparison

`esp32_latency_comparison.csv` compares E1–E6 and Q1 on the same 1,000 fixed Test window IDs. Every row uses 50 warm-up windows, ESP32-D0WDQ6 at 240 MHz, raw PPG 64 Hz plus ACC 32 Hz, and on-device full-pipeline timing. USB transfer, file I/O, sensor acquisition and the 8-second collection interval are excluded.

`e6_fair_latency.csv` contains the 1,000 E6 measurements. `e6_fair_latency.json` records the board, checkpoint hash, timing scope and summary statistics.

E1–E5/Q1 run in a combined firmware and E6 runs in its standalone firmware. The inputs, board, clock, warm-up count and timer scope are the same; firmware layout remains a limitation of the comparison.
