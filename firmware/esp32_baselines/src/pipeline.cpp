#include "pipeline.h"
#include "baselines.h"
#include "rf_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#ifdef ARDUINO
#include <esp_timer.h>
static uint64_t now_ns() { return uint64_t(esp_timer_get_time()) * 1000; }
#else
#include <chrono>
static uint64_t now_ns() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}
#endif
static float pipeline_raw[1024], pipeline_features[38], probability[8];
static double rf_probability[8];
bool baseline_pipeline(int model, const double *ppg, const float *acc,
                       PipelineResult *r) {
  const bool rf = model == 13;
  const int first = rf ? 0 : baseline_first(model),
            channels = rf ? 4 : baseline_channels(model);
  baseline_initialize(); // Buffer đã cấp phát trong setup; không cấp phát theo
                         // cửa sổ.
  const uint64_t start = now_ns();
  uint64_t at = start;
  if (first == 1)
    std::memcpy(pipeline_raw + 256, acc, 768 * sizeof(float));
  else if (!baseline_decimate(ppg, acc, pipeline_raw))
    return false;
  r->duration_ns[0] = now_ns() - at;
  at = now_ns();
  if (!rf)
    baseline_normalize(pipeline_raw, first, channels);
  r->duration_ns[1] = now_ns() - at;
  at = now_ns();
  if (rf)
    rf_features(pipeline_raw, pipeline_features);
  else if (!baseline_temporal(model))
    baseline_spectral(first, channels);
  r->duration_ns[2] = now_ns() - at;
  at = now_ns();
  if (model == 12)
    baseline_quantize();
  r->duration_ns[3] = now_ns() - at;
  at = now_ns();
  if (rf) {
    if (!rf_infer(pipeline_features, rf_probability))
      return false;
  } else
    baseline_infer(model, r->logits);
  r->duration_ns[4] = now_ns() - at;
  at = now_ns();
  if (rf) {
    r->label = uint32_t(std::max_element(rf_probability, rf_probability + 8) -
                        rf_probability);
    for (int i = 0; i < 8; ++i)
      r->logits[i] = float(rf_probability[i]);
  } else {
    const float maximum = *std::max_element(r->logits, r->logits + 8);
    float sum = 0;
    for (int i = 0; i < 8; ++i) {
      probability[i] = std::exp(r->logits[i] - maximum);
      sum += probability[i];
    }
    for (int i = 0; i < 8; ++i)
      probability[i] /= sum;
    r->label =
        uint32_t(std::max_element(probability, probability + 8) - probability);
  }
  r->duration_ns[5] = now_ns() - at;
  r->duration_ns[6] = now_ns() - start;
  return true;
}
extern "C" int host_pipeline(int model, const double *p, const float *a,
                             PipelineResult *r) {
  return baseline_pipeline(model, p, a, r) ? 1 : 0;
}
