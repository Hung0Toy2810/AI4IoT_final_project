#include "har_model.h"
#include "model_weights.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Workspace tĩnh; không cấp phát heap trong inference. Layout C×T.
static float feature_a[24 * 256];
static float feature_b[24 * 256];
static float branch_temp[8 * 128];
static float concatenated[24 * 128];
static double filter_window[512 + 54];
static float ema[8];
static bool ema_ready = false;

static float elu(float x) { return x >= 0.0f ? x : std::expm1(x); }

static void convolution(const ConvSpec &s, const float *input,
                        float *output, int length, bool activation) {
    const int pad = s.kernel / 2;
    for (int oc = 0; oc < s.outputs; ++oc) {
        for (int t = 0; t < length; ++t) {
            float value = s.bias[oc];
            for (int ic = 0; ic < s.inputs; ++ic) {
                const float *w = s.weights + (oc * s.inputs + ic) * s.kernel;
                const float *x = input + ic * length;
                for (int k = 0; k < s.kernel; ++k) {
                    const int at = t + k - pad;
                    if (at >= 0 && at < length) value += w[k] * x[at];
                }
            }
            output[oc * length + t] = activation ? elu(value) : value;
        }
    }
}

static void max_pool(const float *input, float *output, int length) {
    for (int c = 0; c < 24; ++c)
        for (int t = 0; t < length / 2; ++t)
            output[c * (length / 2) + t] =
                std::max(input[c * length + 2 * t], input[c * length + 2 * t + 1]);
}

void har_infer(const float *raw, float *logits) {
    for (int c = 0; c < 4; ++c)
        for (int t = 0; t < 256; ++t)
            feature_b[c * 256 + t] = (raw[c * 256 + t] - input_mean[c]) / input_std[c];
    convolution(stem, feature_b, feature_a, 256, true);
    max_pool(feature_a, feature_b, 256);
    int length = 128;
    for (int block = 0; block < 3; ++block) {
        for (int branch = 0; branch < 3; ++branch) {
            convolution(blocks[block][2 * branch], feature_b, branch_temp, length, true);
            convolution(blocks[block][2 * branch + 1], branch_temp,
                        concatenated + branch * 8 * length, length, true);
        }
        convolution(blocks[block][6], concatenated, feature_a, length, false);
        for (int i = 0; i < 24 * length; ++i)
            feature_a[i] = elu(feature_a[i] + feature_b[i]);
        if (block < 2) {
            max_pool(feature_a, feature_b, length);
            length /= 2;
        }
    }
    float pooled[24];
    for (int c = 0; c < 24; ++c) {
        float sum = 0.0f;
        for (int t = 0; t < length; ++t) sum += feature_a[c * length + t];
        pooled[c] = sum / length;
    }
    for (int label = 0; label < 8; ++label) {
        float sum = head_b[label];
        for (int c = 0; c < 24; ++c) sum += head_w[label * 24 + c] * pooled[c];
        logits[label] = sum;
    }
}

// SOS + odd padding 27 mẫu: cùng decimate(q=2,n=8,zero_phase=True) trên host.
static void filter_pass(double *values, int length) {
    const double first = values[0];
    for (int section = 0; section < 4; ++section) {
        const double *s = filter_sos + section * 6;
        double z0 = filter_zi[section * 2] * first;
        double z1 = filter_zi[section * 2 + 1] * first;
        for (int i = 0; i < length; ++i) {
            const double x = values[i];
            const double y = s[0] * x + z0;
            z0 = s[1] * x - s[4] * y + z1;
            z1 = s[2] * x - s[5] * y;
            values[i] = y;
        }
    }
}

bool har_preprocess(const double *ppg64, const float *acc32, float *raw32) {
    for (int i = 0; i < 512; ++i) if (!std::isfinite(ppg64[i])) return false;
    for (int i = 0; i < 768; ++i) if (!std::isfinite(acc32[i])) return false;
    for (int i = 0; i < 27; ++i) {
        filter_window[i] = 2.0 * ppg64[0] - ppg64[27 - i];
        filter_window[539 + i] = 2.0 * ppg64[511] - ppg64[510 - i];
    }
    for (int i = 0; i < 512; ++i) filter_window[27 + i] = ppg64[i];
    filter_pass(filter_window, 566);
    std::reverse(filter_window, filter_window + 566);
    filter_pass(filter_window, 566);
    std::reverse(filter_window, filter_window + 566);
    for (int i = 0; i < 256; ++i) raw32[i] = static_cast<float>(filter_window[27 + 2 * i]);
    std::memcpy(raw32 + 256, acc32, 768 * sizeof(float));
    return true;
}

void har_reset() { ema_ready = false; }
int har_postprocess(const float *logits, bool reset) {
    float probability[8];
    const float maximum = *std::max_element(logits, logits + 8);
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) {
        probability[i] = std::exp((logits[i] - maximum) / temperature);
        sum += probability[i];
    }
    for (int i = 0; i < 8; ++i) {
        probability[i] /= sum;
        ema[i] = reset || !ema_ready ? probability[i]
            : ema_alpha * probability[i] + (1.0f - ema_alpha) * ema[i];
    }
    ema_ready = true;
    return static_cast<int>(std::max_element(ema, ema + 8) - ema);
}
const char *har_model_hash() { return model_hash; }
size_t har_workspace_bytes() {
    return sizeof(feature_a) + sizeof(feature_b) + sizeof(branch_temp) +
           sizeof(concatenated) + sizeof(filter_window) + sizeof(ema);
}
size_t har_weight_bytes() { return (7248 + 200 + 8 + 2) * sizeof(float) + (24 + 8) * sizeof(double); }
extern "C" void har_host_infer(const float *raw, float *logits) { har_infer(raw, logits); }
extern "C" int har_host_preprocess(const double *ppg, const float *acc, float *raw) {
    return har_preprocess(ppg, acc, raw) ? 1 : 0;
}
