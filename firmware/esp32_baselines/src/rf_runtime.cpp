#include "rf_runtime.h"
#include "baselines.h"
#include "rf_metadata.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#ifdef ARDUINO
#include <esp32/rom/miniz.h>
#include <esp_flash.h>
#include <esp_spi_flash.h>
static uint8_t compressed_tree[32768];
extern "C" const uint8_t rf_model_start[];
static tinfl_decompressor inflater;
#else
#include <fstream>
#include <iterator>
#include <vector>
#include <zlib.h>
static const std::vector<uint8_t> rf_blob = []() {
  std::ifstream input("firmware/esp32_baselines/include/rf_forest.bin",
                      std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
}();
#endif
static uint8_t *tree_buffer = nullptr;
static double wave[2][256], fft_r[128], fft_i[128], psd[65];
static double hann[128], wr[64], wi[64];
static bool ready = false;
static void prepare_fft() {
  if (ready)
    return;
  const double pi = std::acos(-1.0);
  for (int i = 0; i < 128; ++i)
    hann[i] = .5 - .5 * std::cos(2 * pi * i / 128);
  for (int i = 0; i < 64; ++i) {
    wr[i] = std::cos(2 * pi * i / 128);
    wi[i] = -std::sin(2 * pi * i / 128);
  }
  ready = true;
}
static void fft128() {
  for (int i = 1, j = 0; i < 128; ++i) {
    int bit = 64;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j) {
      std::swap(fft_r[i], fft_r[j]);
      std::swap(fft_i[i], fft_i[j]);
    }
  }
  for (int length = 2; length <= 128; length *= 2) {
    const int half = length / 2, step = 128 / length;
    for (int base = 0; base < 128; base += length)
      for (int j = 0; j < half; ++j) {
        const int a = base + j, b = a + half;
        const double re = wr[j * step] * fft_r[b] - wi[j * step] * fft_i[b];
        const double im = wr[j * step] * fft_i[b] + wi[j * step] * fft_r[b];
        fft_r[b] = fft_r[a] - re;
        fft_i[b] = fft_i[a] - im;
        fft_r[a] += re;
        fft_i[a] += im;
      }
  }
}
void rf_features(const float *raw, float *features) {
  prepare_fft();
  for (int c = 0; c < 4; ++c) {
    double mean = 0, squares = 0, lo = raw[c * 256], hi = lo;
    for (int i = 0; i < 256; ++i) {
      const double x = raw[c * 256 + i];
      mean += x;
      squares += x * x;
      lo = std::min(lo, x);
      hi = std::max(hi, x);
    }
    mean /= 256;
    double variance = 0, fourth = 0;
    int changes = 0;
    for (int i = 0; i < 256; ++i) {
      const double d = double(raw[c * 256 + i]) - mean;
      variance += d * d;
      fourth += d * d * d * d;
      if (i && (double(raw[c * 256 + i - 1]) - mean >= 0) != (d >= 0))
        ++changes;
    }
    variance /= 256;
    const double kurt = lo == hi ? 0 : fourth / 256 / (variance * variance) - 3;
    const double values[7] = {
        mean, std::sqrt(variance),  lo, hi, std::sqrt(squares / 256),
        kurt, double(changes) / 255};
    for (int k = 0; k < 7; ++k)
      features[c * 7 + k] = float(values[k]);
  }
  for (int i = 0; i < 256; ++i) {
    wave[0][i] = raw[i];
    double sum = 0;
    for (int c = 1; c < 4; ++c) {
      const double x = raw[c * 256 + i];
      sum += x * x;
    }
    wave[1][i] = std::sqrt(sum);
  }
  for (int c = 0; c < 2; ++c) {
    std::fill(psd, psd + 65, 0.0);
    for (int frame = 0; frame < 3; ++frame) {
      double mean = 0;
      for (int i = 0; i < 128; ++i)
        mean += wave[c][frame * 64 + i];
      mean /= 128;
      for (int i = 0; i < 128; ++i) {
        fft_r[i] = (wave[c][frame * 64 + i] - mean) * hann[i];
        fft_i[i] = 0;
      }
      fft128();
      for (int f = 0; f <= 64; ++f) {
        const double magnitude = fft_r[f] * fft_r[f] + fft_i[f] * fft_i[f];
        psd[f] += magnitude * (f == 0 || f == 64 ? 1 : 2) / (1536 * 3);
      }
    }
    const int lower[4] = {2, 8, 16, 32}, upper[4] = {8, 16, 32, 49};
    for (int b = 0; b < 4; ++b) {
      double sum = 0;
      for (int f = lower[b]; f < upper[b]; ++f)
        sum += psd[f];
      features[28 + c * 4 + b] = float(sum * .25);
    }
    const int peak = int(std::max_element(psd + 2, psd + 49) - psd);
    const auto range = std::minmax_element(wave[c], wave[c] + 256);
    features[36 + c] = *range.first == *range.second ? 0 : float(peak) * .25f;
  }
}
static bool inflate_tree(int index) {
  if (!baseline_initialize())
    return false;
  tree_buffer = static_cast<uint8_t *>(baseline_tensor_arena());
  const size_t size = rf_offsets[index + 1] - rf_offsets[index];
#ifdef ARDUINO
  const size_t physical = spi_flash_cache2phys(rf_model_start);
  if (physical == SPI_FLASH_CACHE2PHYS_FAIL || size > sizeof(compressed_tree) ||
      esp_flash_read(nullptr, compressed_tree, uint32_t(physical + rf_offsets[index]), uint32_t(size)) != ESP_OK)
    return false;
  tinfl_init(&inflater);
  size_t input_size = size, output_size = rf_max_bytes;
  const auto status = tinfl_decompress(
      &inflater, compressed_tree, &input_size, tree_buffer,
      tree_buffer, &output_size,
      TINFL_FLAG_PARSE_ZLIB_HEADER | TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF);
  return status == TINFL_STATUS_DONE && output_size == rf_sizes[index];
#else
  uLongf output = rf_max_bytes;
  return uncompress(tree_buffer, &output, rf_blob.data() + rf_offsets[index],
                    size) == Z_OK &&
         output == rf_sizes[index];
#endif
}
static void tree_prediction(const float *features, double *probability) {
  uint32_t nodes;
  std::memcpy(&nodes, tree_buffer, 4);
  const uint8_t *records = tree_buffer + 4;
  uint16_t i = 0;
  while (records[i * 7] != 255) {
    const uint8_t *node = records + i * 7;
    float threshold;
    std::memcpy(&threshold, node + 3, 4);
    if (features[node[0]] <= threshold)
      ++i;
    else
      std::memcpy(&i, node + 1, 2);
  }
  uint32_t meta;
  std::memcpy(&meta, records + i * 7 + 3, 4);
  const uint8_t mask = meta & 255;
  if ((mask & (mask - 1)) == 0) {
    for (int c = 0; c < 8; ++c)
      if (mask & (1 << c))
        probability[c] += 1;
    return;
  }
  const uint8_t *values = records + nodes * 7 + (meta >> 8) * 8;
  int offset = 0;
  for (int c = 0; c < 8; ++c)
    if (mask & (1 << c)) {
      double p;
      std::memcpy(&p, values + offset * 8, 8);
      probability[c] += p;
      ++offset;
    }
}
bool rf_batch(const float *features, int count, double *probability) {
  std::fill(probability, probability + count * 8, 0.0);
  for (int tree = 0; tree < 100; ++tree) {
    if (!inflate_tree(tree))
      return false;
    for (int i = 0; i < count; ++i)
      tree_prediction(features + i * 38, probability + i * 8);
  }
  for (int i = 0; i < count * 8; ++i)
    probability[i] /= 100;
  return true;
}
bool rf_infer(const float *features, double *probability) {
  return rf_batch(features, 1, probability);
}
size_t rf_workspace() {
  size_t size = rf_max_bytes + sizeof(wave) + sizeof(fft_r) + sizeof(fft_i) +
                sizeof(psd) + sizeof(hann) + sizeof(wr) + sizeof(wi);
#ifdef ARDUINO
  size += sizeof(inflater) + sizeof(compressed_tree);
#endif
  return size;
}
const char *rf_hash() { return rf_checkpoint_hash; }
extern "C" int host_rf_infer(const float *features, double *p) {
  return rf_infer(features, p) ? 1 : 0;
}
extern "C" void host_rf_features(const float *raw, float *f) {
  rf_features(raw, f);
}
