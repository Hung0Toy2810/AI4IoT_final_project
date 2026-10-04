#pragma once
#include <cstddef>
bool rf_infer(const float *features, double *probability);
void rf_features(const float *raw, float *features);
bool rf_batch(const float *features, int count, double *probability);
size_t rf_workspace();
const char *rf_hash();
extern "C" int host_rf_infer(const float *features, double *probability);
extern "C" void host_rf_features(const float *raw, float *features);
