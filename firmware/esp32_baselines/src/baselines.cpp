#include "baselines.h"
#include "baseline_weights.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#ifdef ARDUINO
#include <esp_heap_caps.h>
#include <esp_attr.h>
#define BASELINE_IRAM IRAM_ATTR
#else
#define BASELINE_IRAM
#endif

// Buffer dùng chung cho các model; không cấp phát động trong inference.
static float normalized[4 * 256], spectra[4 * 32 * 32];
#ifdef ARDUINO
static float *activation_storage = nullptr;
static uint32_t *guarded_allocation = nullptr;
#else
static float host_activation_storage[12288];
static float *activation_storage = host_activation_storage;
#endif
static float *activation_a = nullptr, *activation_b = nullptr;
bool baseline_initialize() {
#ifdef ARDUINO
  if (!activation_storage) {
    guarded_allocation = static_cast<uint32_t *>(
        heap_caps_malloc(12288 * sizeof(float) + 16, MALLOC_CAP_8BIT));
    if (guarded_allocation) {
      guarded_allocation[0] = guarded_allocation[12290] = 0xa11c0de1u;
      guarded_allocation[1] = guarded_allocation[12291] = 0x5ee5cafeu;
      activation_storage = reinterpret_cast<float *>(guarded_allocation + 2);
    }
  }
#endif
  activation_a = activation_storage;
  activation_b = activation_storage ? activation_storage + 8192 : nullptr;
  return activation_storage != nullptr;
}
bool baseline_memory_ok() {
#ifdef ARDUINO
  return guarded_allocation && guarded_allocation[0] == 0xa11c0de1u &&
         guarded_allocation[1] == 0x5ee5cafeu &&
         guarded_allocation[12290] == 0xa11c0de1u &&
         guarded_allocation[12291] == 0x5ee5cafeu;
#else
  return true;
#endif
}
size_t baseline_arena_allocation() { return 12288 * sizeof(float) + 16; }
void *baseline_tensor_arena() { return activation_storage; }
static signed char quant_a[4096], quant_b[8192];
static float spectral_grid[25 * 49], fft_re[64], fft_im[64];
static double filter_window[566];

// FFT radix-2: cửa sổ Hann 64, bước 4 mẫu; chuẩn hóa spectrum /sum(Hann).
static void BASELINE_IRAM fft64() {
  for (int i = 1, j = 0; i < 64; ++i) {
    int bit = 32;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j) {
      std::swap(fft_re[i], fft_re[j]);
      std::swap(fft_im[i], fft_im[j]);
    }
  }
  for (int length = 2; length <= 64; length *= 2) {
    const int half = length / 2, step = 64 / length;
    for (int base = 0; base < 64; base += length)
      for (int j = 0; j < half; ++j) {
        const int at = base + j, next = at + half;
        const float wr = twiddle_re[j * step], wi = twiddle_im[j * step];
        const float re = wr * fft_re[next] - wi * fft_im[next];
        const float im = wr * fft_im[next] + wi * fft_re[next];
        fft_re[next] = fft_re[at] - re;
        fft_im[next] = fft_im[at] - im;
        fft_re[at] += re;
        fft_im[at] += im;
      }
  }
}
void BASELINE_IRAM baseline_normalize(const float *raw, int first, int channels) {
  for (int c = first; c < first + channels; ++c)
    for (int t = 0; t < 256; ++t)
      normalized[c * 256 + t] = (raw[c * 256 + t] - norm_mean[c]) / norm_std[c];
}
void BASELINE_IRAM baseline_spectral(int first, int channels) {
  for (int c = first; c < first + channels; ++c) {
    for (int frame = 0; frame < 49; ++frame) {
      for (int t = 0; t < 64; ++t) {
        fft_re[t] = normalized[c * 256 + frame * 4 + t] * hann64[t];
        fft_im[t] = 0;
      }
      fft64();
      for (int f = 0; f < 25; ++f) {
        const float re = fft_re[f] / 32.0f, im = fft_im[f] / 32.0f;
        spectral_grid[f * 49 + frame] =
            std::log1p(std::sqrt(re * re + im * im));
      }
    }
    // Nội suy cùng tọa độ vật lý: f=0..12 Hz, tâm khung t=1..7 s.
    for (int f = 0; f < 32; ++f)
      for (int t = 0; t < 32; ++t) {
        const double ff = double(f) * 24 / 31, tt = double(t) * 48 / 31;
        const int f0 = std::min(int(ff), 23), t0 = std::min(int(tt), 47);
        const double df = ff - f0, dt = tt - t0;
        const double value = (1 - df) * (1 - dt) * spectral_grid[f0 * 49 + t0] +
                             (1 - df) * dt * spectral_grid[f0 * 49 + t0 + 1] +
                             df * (1 - dt) * spectral_grid[(f0 + 1) * 49 + t0] +
                             df * dt * spectral_grid[(f0 + 1) * 49 + t0 + 1];
        spectra[(c * 32 + f) * 32 + t] = float(value);
      }
  }
}
void baseline_prepare(const float *raw) {
  baseline_initialize();
  baseline_normalize(raw, 0, 4);
  baseline_spectral(0, 4);
  baseline_quantize();
}
// Kích thước cố định cho phép unroll kernel; thứ tự cộng FP32 giữ nguyên.
template <int Inputs, int Outputs, int Kernel, int Time>
static void BASELINE_IRAM conv1d(const ConvLayer &layer, const float *input, float *out) {
  constexpr int pad = Kernel / 2;
  for (int oc = 0; oc < Outputs; ++oc)
    for (int t = 0; t < Time; ++t) {
      float value = layer.b[oc];
      for (int ic = 0; ic < Inputs; ++ic) {
        const float *w = layer.w + (oc * Inputs + ic) * Kernel;
        if (t >= pad && t < Time - pad) {
          const float *x = input + ic * Time + t - pad;
          for (int k = 0; k < Kernel; ++k)
            value += x[k] * w[k];
        } else {
          const int low = std::max(0, pad - t),
                    high = std::min(Kernel, Time + pad - t);
          for (int k = low; k < high; ++k)
            value += input[ic * Time + t - pad + k] * w[k];
        }
      }
      out[oc * Time + t] = std::max(0.0f, value);
    }
}
template <int Inputs, int Outputs, int Side>
static void BASELINE_IRAM conv2d(const ConvLayer &layer, const float *input, float *out) {
  for (int oc = 0; oc < Outputs; ++oc)
    for (int y = 0; y < Side; ++y)
      for (int x = 0; x < Side; ++x) {
        float value = layer.b[oc];
        for (int ic = 0; ic < Inputs; ++ic) {
          const float *w = layer.w + (oc * Inputs + ic) * 9;
          if (y > 0 && y < Side - 1 && x > 0 && x < Side - 1) {
            const float *p = input + (ic * Side + y - 1) * Side + x - 1;
            value += p[0] * w[0];
            value += p[1] * w[1];
            value += p[2] * w[2];
            value += p[Side] * w[3];
            value += p[Side + 1] * w[4];
            value += p[Side + 2] * w[5];
            value += p[Side * 2] * w[6];
            value += p[Side * 2 + 1] * w[7];
            value += p[Side * 2 + 2] * w[8];
          } else {
            for (int ky = std::max(0, 1 - y); ky < std::min(3, Side + 1 - y);
                 ++ky)
              for (int kx = std::max(0, 1 - x); kx < std::min(3, Side + 1 - x);
                   ++kx)
                value += input[(ic * Side + y + ky - 1) * Side + x + kx - 1] *
                         w[ky * 3 + kx];
          }
        }
        out[(oc * Side + y) * Side + x] = std::max(0.0f, value);
      }
}
static void BASELINE_IRAM pool1d(const float *input, float *out, int channels, int time) {
  for (int c = 0; c < channels; ++c)
    for (int t = 0; t < time / 2; ++t)
      out[c * (time / 2) + t] =
          std::max(input[c * time + 2 * t], input[c * time + 2 * t + 1]);
}
static void BASELINE_IRAM pool2d(const float *input, float *out, int channels, int side) {
  for (int c = 0; c < channels; ++c)
    for (int y = 0; y < side / 2; ++y)
      for (int x = 0; x < side / 2; ++x) {
        float value = 0;
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx)
            value = std::max(
                value, input[(c * side + 2 * y + dy) * side + 2 * x + dx]);
        out[(c * (side / 2) + y) * (side / 2) + x] = value;
      }
}
static void BASELINE_IRAM linear(const float *input, float *output, const float *w,
                   const float *b, int inputs, int outputs) {
  for (int o = 0; o < outputs; ++o) {
    float value = b[o];
    for (int i = 0; i < inputs; ++i)
      value += input[i] * w[o * inputs + i];
    output[o] = value;
  }
}
// Requantization double-rounding theo TFLite/gemmlowp, bias và accumulator
// INT32.
static int BASELINE_IRAM requant(int value, int multiplier, int shift) {
  const int left = std::max(shift, 0), right = std::max(-shift, 0);
  const int64_t product = int64_t(value) * int64_t(1u << left) * multiplier;
  const int64_t nudge =
      product >= 0 ? (int64_t(1) << 30) : 1 - (int64_t(1) << 30);
  const int high = int((product + nudge) / (int64_t(1) << 31));
  const int mask = (1 << right) - 1, remainder = high & mask;
  const int threshold = (mask >> 1) + (high < 0);
  return (high >> right) + (remainder > threshold);
}
static void BASELINE_IRAM qconv(const QuantLayer &layer, const signed char *input,
                  signed char *out, int side) {
  for (int y = 0; y < side; ++y)
    for (int x = 0; x < side; ++x)
      for (int oc = 0; oc < layer.out; ++oc) {
        int value = layer.b[oc];
        for (int ky = 0; ky < 3; ++ky)
          for (int kx = 0; kx < 3; ++kx) {
            const int iy = y + ky - 1, ix = x + kx - 1;
            if (iy < 0 || iy >= side || ix < 0 || ix >= side)
              continue;
            for (int ic = 0; ic < layer.in; ++ic)
              value +=
                  (int(input[(iy * side + ix) * layer.in + ic]) - layer.iz) *
                  int(layer.w[((oc * 3 + ky) * 3 + kx) * layer.in + ic]);
          }
        value = requant(value, layer.mult[oc], layer.shift[oc]) + layer.oz;
        out[(y * side + x) * layer.out + oc] =
            static_cast<signed char>(std::max(layer.oz, std::min(127, value)));
      }
}
static void BASELINE_IRAM qpool(const signed char *input, signed char *out, int channels,
                  int side) {
  for (int y = 0; y < side / 2; ++y)
    for (int x = 0; x < side / 2; ++x)
      for (int c = 0; c < channels; ++c) {
        int value = -128;
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx)
            value = std::max(
                value,
                int(input[((2 * y + dy) * side + 2 * x + dx) * channels + c]));
        out[(y * (side / 2) + x) * channels + c] =
            static_cast<signed char>(value);
      }
}
void BASELINE_IRAM baseline_quantize() {
  for (int y = 0; y < 32; ++y)
    for (int x = 0; x < 32; ++x)
      for (int c = 0; c < 4; ++c) {
        const float value = spectra[(c * 32 + y) * 32 + x] / q_input_scale[0];
        // np.rint của pipeline CPU dùng ties-to-even.
        const int q = int(std::nearbyint(value)) + q_input_zero;
        quant_a[(y * 32 + x) * 4 + c] =
            static_cast<signed char>(std::max(-128, std::min(127, q)));
      }
}
static void BASELINE_IRAM quantized_infer(float *out) {
  qconv(quant_layers[0], quant_a, quant_b, 32);
  qpool(quant_b, quant_a, 8, 32);
  qconv(quant_layers[1], quant_a, quant_b, 16);
  qpool(quant_b, quant_a, 16, 16);
  qconv(quant_layers[2], quant_a, quant_b, 8);
  for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 2; ++x)
      for (int c = 0; c < 24; ++c) {
        int sum = 0;
        for (int dy = 0; dy < 4; ++dy)
          for (int dx = 0; dx < 4; ++dx)
            sum += quant_b[((y * 4 + dy) * 8 + x * 4 + dx) * 24 + c];
        quant_a[(y * 2 + x) * 24 + c] =
            static_cast<signed char>((sum + (sum >= 0 ? 8 : -8)) / 16);
      }
  const QuantLayer &head = quant_layers[3];
  for (int o = 0; o < 8; ++o) {
    int value = head.b[o];
    for (int i = 0; i < 96; ++i)
      value += (int(quant_a[i]) - head.iz) * int(head.w[o * 96 + i]);
    const int q = std::max(
        -128,
        std::min(127, requant(value, head.mult[o], head.shift[o]) + head.oz));
    out[o] = float(q - q_output_zero) * q_output_scale[0];
  }
}
void BASELINE_IRAM baseline_infer(int index, float *out) {
  if (index == 12) {
    quantized_infer(out);
    return;
  }
  const Baseline &m = models[index];
  if (m.temporal) {
    conv1d<4, 16, 5, 256>(m.conv[0], normalized, activation_a);
    pool1d(activation_a, activation_b, 16, 256);
    conv1d<16, 32, 5, 128>(m.conv[1], activation_b, activation_a);
    pool1d(activation_a, activation_b, 32, 128);
    conv1d<32, 64, 3, 64>(m.conv[2], activation_b, activation_a);
    for (int c = 0; c < 64; ++c) {
      float sum = 0;
      for (int t = 0; t < 64; ++t)
        sum += activation_a[c * 64 + t];
      activation_b[c] = sum / 64;
    }
    linear(activation_b, activation_a, m.w1, m.b1, 64, 32);
    linear(activation_a, out, m.w2, m.b2, 32, 8);
  } else {
    if (m.channels == 4)
      conv2d<4, 8, 32>(m.conv[0], spectra, activation_a);
    else if (m.channels == 3)
      conv2d<3, 8, 32>(m.conv[0], spectra + 1024, activation_a);
    else
      conv2d<1, 8, 32>(m.conv[0], spectra, activation_a);
    pool2d(activation_a, activation_b, 8, 32);
    conv2d<8, 16, 16>(m.conv[1], activation_b, activation_a);
    pool2d(activation_a, activation_b, 16, 16);
    conv2d<16, 24, 8>(m.conv[2], activation_b, activation_a);
    for (int c = 0; c < 24; ++c)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
          float sum = 0;
          for (int dy = 0; dy < 4; ++dy)
            for (int dx = 0; dx < 4; ++dx)
              sum += activation_a[(c * 8 + y * 4 + dy) * 8 + x * 4 + dx];
          activation_b[(c * 2 + y) * 2 + x] = sum / 16;
        }
    linear(activation_b, out, m.w1, m.b1, 96, 8);
  }
}
static void BASELINE_IRAM filter_pass(double *values, int length) {
  const double first = values[0];
  for (int section = 0; section < 4; ++section) {
    const double *s = filter_sos + section * 6;
    double z0 = filter_zi[section * 2] * first,
           z1 = filter_zi[section * 2 + 1] * first;
    for (int i = 0; i < length; ++i) {
      const double x = values[i], y = s[0] * x + z0;
      z0 = s[1] * x - s[4] * y + z1;
      z1 = s[2] * x - s[5] * y;
      values[i] = y;
    }
  }
}
bool BASELINE_IRAM baseline_decimate(const double *ppg, const float *acc, float *raw) {
  for (int i = 0; i < 512; ++i)
    if (!std::isfinite(ppg[i]))
      return false;
  for (int i = 0; i < 768; ++i)
    if (!std::isfinite(acc[i]))
      return false;
  for (int i = 0; i < 27; ++i) {
    filter_window[i] = 2 * ppg[0] - ppg[27 - i];
    filter_window[539 + i] = 2 * ppg[511] - ppg[510 - i];
  }
  for (int i = 0; i < 512; ++i)
    filter_window[27 + i] = ppg[i];
  filter_pass(filter_window, 566);
  std::reverse(filter_window, filter_window + 566);
  filter_pass(filter_window, 566);
  std::reverse(filter_window, filter_window + 566);
  for (int i = 0; i < 256; ++i)
    raw[i] = float(filter_window[27 + 2 * i]);
  std::memcpy(raw + 256, acc, 768 * sizeof(float));
  return true;
}
const char *baseline_name(int i) { return i == 12 ? "Q1_42" : models[i].name; }
const char *baseline_sha(int i) { return i == 12 ? q_hash : models[i].sha; }
int baseline_parameters(int i) {
  return i == 12 ? models[3].parameters : models[i].parameters;
}
size_t baseline_coefficients(int i) {
  return i == 12 ? q_coefficients : models[i].coefficients;
}
size_t baseline_workspace() {
  return sizeof(normalized) + sizeof(spectra) + 12288 * sizeof(float) +
         sizeof(quant_a) + sizeof(quant_b) + sizeof(spectral_grid) +
         sizeof(fft_re) + sizeof(fft_im) + sizeof(filter_window);
}
extern "C" void host_prepare(const float *raw) { baseline_prepare(raw); }
extern "C" void host_infer(int i, float *out) { baseline_infer(i, out); }
extern "C" void host_spectrum(const float *raw, float *out) {
  baseline_prepare(raw);
  std::memcpy(out, spectra, sizeof(spectra));
}

extern "C" void host_set_spectrum(const float *input) {
  std::memcpy(spectra, input, sizeof(spectra));
  baseline_quantize();
}
int baseline_channels(int i) { return i == 12 ? 4 : models[i].channels; }
int baseline_first(int i) { return i == 12 ? 0 : models[i].first_channel; }
bool baseline_temporal(int i) { return i < 12 && models[i].temporal; }
