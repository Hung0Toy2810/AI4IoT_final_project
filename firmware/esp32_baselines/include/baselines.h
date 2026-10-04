#pragma once
#include <cstddef>
struct ConvLayer {
  int in, out, kernel;
  const float *w, *b;
};
struct Baseline {
  const char *name, *sha;
  int channels, first_channel;
  bool temporal;
  ConvLayer conv[3];
  const float *w1, *b1, *w2, *b2;
  int parameters;
  size_t coefficients;
};
struct QuantLayer {
  int in, out, iz, oz;
  const signed char *w;
  const int *b, *mult, *shift;
};
void baseline_prepare(const float *raw);
void baseline_infer(int model, float *out);
bool baseline_decimate(const double *ppg, const float *acc, float *raw);
const char *baseline_name(int model);
const char *baseline_sha(int model);
size_t baseline_workspace();
size_t baseline_coefficients(int model);
int baseline_parameters(int model);
extern "C" void host_prepare(const float *raw);
extern "C" void host_infer(int model, float *out);
extern "C" void host_spectrum(const float *raw, float *out);

void baseline_normalize(const float *raw, int first, int channels);
void baseline_spectral(int first, int channels);
void baseline_quantize();
int baseline_channels(int model);
int baseline_first(int model);
bool baseline_temporal(int model);
extern "C" void host_set_spectrum(const float *input);

bool baseline_initialize();
void *baseline_tensor_arena();

bool baseline_memory_ok();
size_t baseline_arena_allocation();
