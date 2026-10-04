#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_system.h>
#include "har_model.h"

// Gói USB: magic HARP, command, argument; payload rồi CRC32 little-endian.
// command 1/3: 4×256 float32; command 2/4: 512 float64 PPG + 3×256 float32 ACC.
// 1/2 trả logits; 3/4 benchmark. Argument = reset EMA hoặc số lần đo.
alignas(8) static uint8_t packet[7168];
static float raw[1024], logits[8];
static uint32_t elapsed[1000];
static int ema_label = 0;
static uint32_t crc32(const uint8_t *data, size_t size) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}
static void info() {
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    char address[18];
    snprintf(address, sizeof(address), "%02x:%02x:%02x:%02x:%02x:%02x", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    Serial.printf("@INFO {\"mac\":\"%s\",\"checkpoint_sha256\":\"%s\",\"cpu_mhz\":%u,\"flash_bytes\":%u,\"psram_bytes\":%u,\"free_heap_bytes\":%u,\"min_free_heap_bytes\":%u,\"largest_free_block_bytes\":%u,\"sketch_bytes\":%u,\"model_workspace_bytes\":%u,\"coefficient_bytes\":%u}\n",
        address, har_model_hash(),
        ESP.getCpuFreqMHz(), ESP.getFlashChipSize(), ESP.getPsramSize(), ESP.getFreeHeap(),
        ESP.getMinFreeHeap(), heap_caps_get_largest_free_block(MALLOC_CAP_8BIT),
        ESP.getSketchSize(), unsigned(har_workspace_bytes()), unsigned(har_weight_bytes()));
}
static bool compute(bool full, bool reset, uint32_t &pre, uint32_t &infer, uint32_t &post) {
    const uint32_t start = micros();
    if (full && !har_preprocess(reinterpret_cast<double *>(packet),
            reinterpret_cast<float *>(packet + 4096), raw)) return false;
    pre = micros() - start;
    uint32_t at = micros();
    har_infer(full ? raw : reinterpret_cast<float *>(packet), logits);
    infer = micros() - at;
    at = micros();
    ema_label = har_postprocess(logits, reset);
    post = micros() - at;
    return true;
}
void setup() {
    Serial.begin(921600);
    Serial.setTimeout(10000);
    delay(500);
    info();
}
void loop() {
    if (Serial.available() < 12) { delay(1); return; }
    uint32_t header[3];
    if (Serial.readBytes(reinterpret_cast<char *>(header), sizeof(header)) != sizeof(header)) return;
    if (header[0] != 0x50524148u) { Serial.println("@ERROR invalid_header"); return; }
    const uint32_t command = header[1], argument = header[2];
    if (command == 0) { info(); return; }
    if (command > 4) { Serial.println("@ERROR invalid_command"); return; }
    const bool full = command == 2 || command == 4;
    const size_t size = full ? 7168 : 4096;
    uint32_t checksum;
    if (Serial.readBytes(reinterpret_cast<char *>(packet), size) != size ||
        Serial.readBytes(reinterpret_cast<char *>(&checksum), 4) != 4 ||
        checksum != crc32(packet, size)) { Serial.println("@ERROR payload_crc"); return; }
    uint32_t pre = 0, infer = 0, post = 0;
    if (command <= 2) {
        const uint32_t start = micros();
        if (!compute(full, argument != 0, pre, infer, post)) {
            Serial.println("@ERROR nonfinite_input"); return;
        }
        const uint32_t total = micros() - start;
        Serial.printf("@RESULT {\"pre_us\":%u,\"infer_us\":%u,\"post_us\":%u,\"total_us\":%u,\"logits\":[", pre, infer, post, total);
        for (int i = 0; i < 8; ++i) { if (i) Serial.print(','); Serial.printf("%.9g", double(logits[i])); }
        Serial.printf("],\"ema_label\":%d}\n", ema_label);
        return;
    }
    if (argument == 0 || argument > 1000) { Serial.println("@ERROR invalid_repeats"); return; }
    // Warm-up và yield nằm ngoài khoảng đo; USB không tính vào latency.
    for (int i = 0; i < 50; ++i) { compute(full, true, pre, infer, post); delay(1); }
    const uint32_t heap_before = ESP.getFreeHeap();
    uint64_t pre_sum = 0, infer_sum = 0, post_sum = 0;
    for (uint32_t i = 0; i < argument; ++i) {
        const uint32_t start = micros();
        if (!compute(full, true, pre, infer, post)) { Serial.println("@ERROR nonfinite_input"); return; }
        elapsed[i] = full ? micros() - start : infer;
        pre_sum += pre; infer_sum += infer; post_sum += post;
        delay(1);
    }
    Serial.printf("@BENCH {\"full_pipeline\":%s,\"repeats\":%u,\"warmup\":50,\"heap_before_bytes\":%u,\"heap_after_bytes\":%u,\"min_free_heap_bytes\":%u,\"stack_high_water_bytes\":%u,\"pre_mean_us\":%.3f,\"infer_mean_us\":%.3f,\"post_mean_us\":%.3f,\"times_us\":[",
        full ? "true" : "false", argument, heap_before, ESP.getFreeHeap(), ESP.getMinFreeHeap(),
        unsigned(uxTaskGetStackHighWaterMark(nullptr)), double(pre_sum)/argument,
        double(infer_sum)/argument, double(post_sum)/argument);
    for (uint32_t i = 0; i < argument; ++i) { if (i) Serial.print(','); Serial.print(elapsed[i]); }
    Serial.println("]}");
}
