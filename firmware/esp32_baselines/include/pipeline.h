#pragma once
#include <cstdint>
struct PipelineResult {
  uint64_t duration_ns[7];
  uint32_t label;
  float logits[8];
};
bool baseline_pipeline(int model, const double *ppg, const float *acc,
                       PipelineResult *result);
extern "C" int host_pipeline(int model, const double *ppg, const float *acc,
                             PipelineResult *result);
