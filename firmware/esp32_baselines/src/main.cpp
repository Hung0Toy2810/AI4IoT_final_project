#include "baselines.h"
#include "pipeline.h"
#include "rf_runtime.h"
#include <Arduino.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <esp_attr.h>
#include <esp_heap_caps.h>
#include <esp_system.h>

SET_LOOP_TASK_STACK_SIZE(16384);
struct PersistentTrace {
  uint32_t magic, command, input_crc, phase;
  int model;
};
RTC_NOINIT_ATTR static volatile PersistentTrace trace;
static uint32_t fault_bits = 0;
static bool memory_healthy() {
  if (!baseline_memory_ok())
    fault_bits |= 1;
  if (!heap_caps_check_integrity_all(false))
    fault_bits |= 2;
  if (uxTaskGetStackHighWaterMark(nullptr) < 512)
    fault_bits |= 4;
  return fault_bits == 0;
}

// Protocol binary HARB: command 1 CNN test, 2 full pipeline, 3 RF batch.
// Payload CRC32; response cũng có CRC32. Timer chỉ bao compute trên MCU.
static uint8_t *packet = nullptr;
static double *rf_probabilities = nullptr;
struct Record {
  uint32_t filter, norm, repr, quant, infer, post, total, label;
  float logits[8];
};
static Record result[14];
static uint32_t crc32(const uint8_t *data, size_t size) {
  uint32_t crc = 0xffffffffu;
  for (size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}
static void response(uint32_t command, const void *data, size_t size,
                     uint32_t status = 0) {
  const uint32_t header[4] = {0x53424d48u, command, uint32_t(size), status};
  const uint32_t crc = crc32(static_cast<const uint8_t *>(data), size);
  Serial.write(reinterpret_cast<const uint8_t *>(header), sizeof(header));
  if (size)
    Serial.write(static_cast<const uint8_t *>(data), size);
  Serial.write(reinterpret_cast<const uint8_t *>(&crc), 4);
}
static int classify(float *logits) {
  const float max = *std::max_element(logits, logits + 8);
  float sum = 0, probability[8];
  for (int i = 0; i < 8; ++i) {
    probability[i] = std::exp(logits[i] - max);
    sum += probability[i];
  }
  for (int i = 0; i < 8; ++i)
    probability[i] /= sum;
  return int(std::max_element(probability, probability + 8) - probability);
}
static void info() {
  uint8_t mac[6];
  esp_read_mac(mac, ESP_MAC_WIFI_STA);
  Serial.printf("@INFO "
                "{\"mac\":\"%02x:%02x:%02x:%02x:%02x:%02x\",\"cpu_mhz\":%u,"
                "\"flash_bytes\":%u,\"psram_bytes\":%u,\"sketch_bytes\":%u,"
                "\"free_heap_bytes\":%u,\"min_free_heap_bytes\":%u,\"largest_"
                "free_block_bytes\":%u,\"cnn_workspace_bytes\":%u,\"rf_"
                "workspace_bytes\":%u,\"stack_free_bytes\":%u,\"models\":[",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5],
                ESP.getCpuFreqMHz(), ESP.getFlashChipSize(), ESP.getPsramSize(),
                ESP.getSketchSize(), ESP.getFreeHeap(), ESP.getMinFreeHeap(),
                heap_caps_get_largest_free_block(MALLOC_CAP_8BIT),
                unsigned(baseline_workspace()), unsigned(rf_workspace()),
                unsigned(uxTaskGetStackHighWaterMark(nullptr)));
  for (int i = 0; i < 13; ++i) {
    if (i)
      Serial.print(',');
    Serial.printf(
        "{\"name\":\"%s\",\"sha256\":\"%s\",\"coefficients_bytes\":%u}",
        baseline_name(i), baseline_sha(i), unsigned(baseline_coefficients(i)));
  }
  Serial.printf(",{\"name\":\"E1_42\",\"sha256\":\"%s\"}],\"flash_mhz\":%u,"
                "\"arena_allocation_"
                "bytes\":%u,\"loop_stack_bytes\":%u,\"fault_bits\":%u,\"reset_"
                "reason\":%u,\"last_command\":%u,\"last_model\":%d,\"last_"
                "phase\":%u,\"last_input_crc\":%u}\n",
                rf_hash(), unsigned(ESP.getFlashChipSpeed() / 1000000),
                unsigned(baseline_arena_allocation()),
                unsigned(getArduinoLoopTaskStackSize()), fault_bits,
                unsigned(esp_reset_reason()), trace.command, trace.model,
                trace.phase, trace.input_crc);
}
static bool full_run(int model, Record &r) {
  PipelineResult measured;
  if (!baseline_pipeline(model, reinterpret_cast<double *>(packet),
                         reinterpret_cast<float *>(packet + 4096), &measured))
    return false;
  uint32_t *times = &r.filter;
  for (int i = 0; i < 7; ++i)
    times[i] = uint32_t(measured.duration_ns[i] / 1000);
  r.label = measured.label;
  std::memcpy(r.logits, measured.logits, sizeof(r.logits));
  return true;
}

void setup() {
  if (trace.magic != 0x54524143u) {
    trace.magic = 0x54524143u;
    trace.command = 0;
    trace.input_crc = 0;
    trace.phase = 0;
    trace.model = -1;
  }
  Serial.setRxBufferSize(4096);
  Serial.begin(460800);
  packet =
      static_cast<uint8_t *>(heap_caps_malloc(64 * 38 * 4, MALLOC_CAP_8BIT));
  rf_probabilities = static_cast<double *>(
      heap_caps_malloc(64 * 8 * sizeof(double), MALLOC_CAP_8BIT));
  if (!packet || !rf_probabilities || !baseline_initialize()) {
    Serial.println("@ERROR init_memory");
    while (true)
      delay(1000);
  }
  Serial.setTimeout(10000);
  delay(500);
  info();
}
void loop() {
  if (Serial.available() < 12) {
    delay(1);
    return;
  }
  uint32_t header[3];
  if (Serial.readBytes(reinterpret_cast<char *>(header), 12) != 12)
    return;
  if (header[0] != 0x42524148u) {
    response(0, nullptr, 0, 1);
    return;
  }
  const uint32_t command = header[1], argument = header[2];
  if (command == 0) {
    info();
    return;
  }
  if (command > 3 || (command == 3 && (argument == 0 || argument > 64))) {
    response(command, nullptr, 0, 2);
    return;
  }
  const size_t size = command == 1   ? 4096
                      : command == 2 ? 7168
                                     : argument * 38 * 4;
  uint32_t crc;
  if (Serial.readBytes(reinterpret_cast<char *>(packet), size) != size ||
      Serial.readBytes(reinterpret_cast<char *>(&crc), 4) != 4 ||
      crc != crc32(packet, size)) {
    response(command, nullptr, 0, 3);
    return;
  }
  trace.command = command;
  trace.input_crc = crc;
  trace.phase = 1;
  trace.model = -1;
  if (!memory_healthy()) {
    response(command, nullptr, 0, 6);
    return;
  }
  if (command == 1) {
    baseline_prepare(reinterpret_cast<float *>(packet));
    for (int i = 0; i < 13; ++i) {
      trace.model = i;
      trace.phase = 2;
      uint32_t at = micros();
      baseline_infer(i, result[i].logits);
      result[i].infer = micros() - at;
      at = micros();
      result[i].label = uint32_t(classify(result[i].logits));
      result[i].post = micros() - at;
      if (!memory_healthy()) {
        response(command, nullptr, 0, 6);
        return;
      }
      delay(1);
    }
    trace.phase = 3;
    response(command, result, 13 * sizeof(Record));
  } else if (command == 2) {
    for (int i = 0; i < 14; ++i) {
      trace.model = i;
      trace.phase = 2;
      if (!full_run(i, result[i])) {
        response(command, nullptr, 0, 4);
        return;
      }
      if (!memory_healthy()) {
        response(command, nullptr, 0, 6);
        return;
      }
      delay(1);
    }
    trace.phase = 3;
    response(command, result, sizeof(result));
  } else {
    trace.model = 13;
    trace.phase = 2;
    if (!rf_batch(reinterpret_cast<float *>(packet), int(argument),
                  rf_probabilities)) {
      response(command, nullptr, 0, 5);
      return;
    }
    if (!memory_healthy()) {
      response(command, nullptr, 0, 6);
      return;
    }
    trace.phase = 3;
    response(command, rf_probabilities, argument * 8 * sizeof(double));
  }
}
