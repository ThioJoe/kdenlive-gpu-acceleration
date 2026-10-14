/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

/*
 * Per channel color effects, done with lookup tables computed on the host
 * with the same formulas as the CPU filters:
 *  - "brightness" (MLT core): Kdenlive's Brightness and its Fade in / Fade out
 *  - "lift_gamma_gain" (MLT plus)
 * They are cheap on the CPU, so they are marked lightweight: they only run on
 * the GPU when the frame is already there for another accelerated effect.
 */

#include "effect.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace gpufx {

namespace {

class LutJob final : public EffectJob
{
public:
    uint8_t lut[4 * 256];

    void setIdentity(int channel)
    {
        for (int i = 0; i < 256; ++i) {
            lut[channel * 256 + i] = uint8_t(i);
        }
    }

    CUresult run(GpuScope &scope, int width, int height, CUdeviceptr in, CUdeviceptr out) override
    {
        DeviceBuffer table;
        CUresult res = table.allocate(sizeof(lut));
        if (res == CUDA_SUCCESS) {
            res = scope.upload(table.ptr(), lut, sizeof(lut));
        }
        if (res == CUDA_SUCCESS) {
            const int count = width * height;
            res = scope.launch1D(KernelId::gpufx_lut_rgba, count, in, out, count, table.ptr());
        }
        return res;
    }
};

EffectPlan planBrightness(const EffectContext &ctx)
{
    // Same parameter handling as filter_brightness.c
    double level = 1.0;
    if (mlt_properties_get(ctx.properties, "level")) {
        level = mlt_properties_anim_get_double(ctx.properties, "level", ctx.position, ctx.length);
    } else {
        level = std::fabs(mlt_properties_get_double(ctx.properties, "start"));
        if (mlt_properties_get(ctx.properties, "end")) {
            const double end = std::fabs(mlt_properties_get_double(ctx.properties, "end"));
            level += (end - level) * mlt_filter_get_progress(ctx.filter, ctx.frame);
        }
    }
    double alphaLevel =
        mlt_properties_exists(ctx.properties, "alpha") ? mlt_properties_anim_get_double(ctx.properties, "alpha", ctx.position, ctx.length) : 1.0;
    if (alphaLevel < 0.0) {
        alphaLevel = level;
    }
    if (level == 1.0 && alphaLevel == 1.0) {
        return EffectPlan::passThrough();
    }
    // The CPU filter works in the requested format; only rgba is identical here
    if (ctx.format != mlt_image_rgba || !std::isfinite(level) || !std::isfinite(alphaLevel)) {
        return EffectPlan::cpu();
    }
    auto job = std::make_unique<LutJob>();
    for (int i = 0; i < 256; ++i) {
        const uint8_t value = uint8_t(std::clamp(std::round(double(i) * level), 0.0, 255.0));
        job->lut[i] = job->lut[256 + i] = job->lut[512 + i] = level != 1.0 ? value : uint8_t(i);
        job->lut[768 + i] = alphaLevel != 1.0 ? uint8_t(std::clamp(std::round(double(i) * alphaLevel), 0.0, 255.0)) : uint8_t(i);
    }
    return EffectPlan::gpu(std::move(job));
}

// calc_lift_gamma_gain() from filter_lift_gamma_gain.c
void liftGammaGain(double normalized, double lift, double gamma, double gain, double *result)
{
    double v = std::pow(normalized, 1.0 / 2.2);
    v += lift * (1.0 - v);
    v = std::max(v, 0.0);
    v = std::pow(v, 2.2 / gamma);
    v *= std::pow(gain, 1.0 / gamma);
    *result = std::clamp(v, 0.0, 1.0);
}

EffectPlan planLiftGammaGain(const EffectContext &ctx)
{
    if (ctx.format != mlt_image_rgba) {
        return EffectPlan::cpu();
    }
    const char *names[3][3] = {{"lift_r", "gamma_r", "gain_r"}, {"lift_g", "gamma_g", "gain_g"}, {"lift_b", "gamma_b", "gain_b"}};
    auto job = std::make_unique<LutJob>();
    for (int c = 0; c < 3; ++c) {
        const double lift = mlt_properties_anim_get_double(ctx.properties, names[c][0], ctx.position, ctx.length);
        const double gamma = mlt_properties_anim_get_double(ctx.properties, names[c][1], ctx.position, ctx.length);
        const double gain = mlt_properties_anim_get_double(ctx.properties, names[c][2], ctx.position, ctx.length);
        for (int i = 0; i < 256; ++i) {
            double value;
            liftGammaGain(double(i) / 255.0, lift, gamma, gain, &value);
            job->lut[c * 256 + i] = uint8_t(std::lrint(value * 255.0));
        }
    }
    job->setIdentity(3);
    return EffectPlan::gpu(std::move(job));
}

} // namespace

const EffectDescriptor &brightnessEffect()
{
    static const EffectDescriptor descriptor{"brightness", true, planBrightness};
    return descriptor;
}

const EffectDescriptor &liftGammaGainEffect()
{
    static const EffectDescriptor descriptor{"lift_gamma_gain", true, planLiftGammaGain};
    return descriptor;
}

} // namespace gpufx
