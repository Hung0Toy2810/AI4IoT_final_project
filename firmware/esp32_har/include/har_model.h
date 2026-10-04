#pragma once
#include <cstddef>
#include <cstdint>

struct ConvSpec {
    int inputs, outputs, kernel;
    const float *weights, *bias;
};
void har_infer(const float *raw, float *logits);
bool har_preprocess(const double *ppg64, const float *acc32, float *raw32);
int har_postprocess(const float *logits, bool reset);
void har_reset();
const char *har_model_hash();
size_t har_workspace_bytes();
size_t har_weight_bytes();
extern "C" void har_host_infer(const float *raw, float *logits);
extern "C" int har_host_preprocess(const double *ppg, const float *acc, float *raw);
