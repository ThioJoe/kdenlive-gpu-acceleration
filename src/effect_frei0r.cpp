/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

/*
 * GPU versions of frei0r plugins, as run by MLT's frei0r module:
 *  - "frei0r.squareblur" (Square Blur), "frei0r.glow" (Glow) and
 *    "frei0r.softglow" (Soft Glow), built on frei0r's square blur
 *    (include/frei0r/blur.h)
 *  - "frei0r.sharpness" (Sharpen), MPlayer's unsharp mask
 *
 * The parameters are read the way MLT passes them to the plugin
 * (process_frei0r_item() in frei0r_helper.c). The plugin code is the same in
 * every frei0r release since 1.4. The plugins always produce 8-bit RGBA, MLT
 * converts the frame to RGBA for them whatever the requested format.
 */

#include "effect.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace gpufx {

namespace {

/** process_frei0r_item() splits the frame in horizontal slices processed by
 *  separate plugin instances when the "threads" property asks for it. The
 *  blurs then stop at the slice borders, which the GPU version does not do,
 *  so only the default single slice is handled. */
bool slicedByMlt(const EffectContext &ctx)
{
    if (!mlt_properties_get(ctx.properties, "threads")) {
        return false;
    }
    return mlt_properties_get_int(ctx.properties, "threads") != 1 && mlt_slices_count_normal() > 1;
}

/** A double parameter as process_frei0r_item() reads it: by index, then by
 *  name, then through the name map kept for old projects, and scaled when
 *  MLT lists it as resolution dependent. Returns false when it is not set:
 *  the plugin then keeps the value it has, its default. */
bool frei0rDouble(const EffectContext &ctx, int index, const char *name, double &value)
{
    char key[16];
    snprintf(key, sizeof(key), "%d", index);
    const char *property = key;
    if (!mlt_properties_get(ctx.properties, property)) {
        property = name;
    }
    if (!mlt_properties_get(ctx.properties, property)) {
        property = nullptr;
        auto map = static_cast<mlt_properties>(mlt_properties_get_data(ctx.properties, "_param_name_map", nullptr));
        const int count = map ? mlt_properties_count(map) : 0;
        for (int i = 0; i < count && !property; ++i) {
            const char *mapped = mlt_properties_get_value(map, i);
            const char *oldName = mlt_properties_get_name(map, i);
            if (mapped && !strcmp(mapped, key) && mlt_properties_get(ctx.properties, oldName)) {
                property = oldName;
            }
        }
        if (!property) {
            return false;
        }
    }
    double t = mlt_properties_anim_get_double(ctx.properties, property, ctx.position, ctx.length);
    const double scale = mlt_profile_scale_width(mlt_service_profile(MLT_FILTER_SERVICE(ctx.filter)), ctx.width);
    if (scale != 1.0) {
        auto scaleMap = static_cast<mlt_properties>(mlt_properties_get_data(ctx.properties, "_resolution_scale", nullptr));
        const double scale2 = mlt_properties_get_double(scaleMap, property);
        if (scale2 != 0.0) {
            t *= scale * scale2;
        }
    }
    value = t;
    return true;
}

// Output modes of gpufx_square_blur_v
enum SquareBlurMode { Plain = 0, Glow = 1, SoftGlowScreen = 2, SoftGlowOverlay = 3, SoftGlowAdd = 4 };

class SquareBlurJob final : public EffectJob
{
public:
    double kernel = 0;    // blur.h: the box size relative to the largest image side
    int mode = Plain;     // what is done with the blurred image
    uint8_t curve[256]{}; // Soft Glow: transfer applied to the lightness before the blur

    CUresult run(GpuScope &scope, int width, int height, CUdeviceptr in, CUdeviceptr out) override
    {
        // blur_update(): the kernel size comes from the size of the plugin instance, i.e. the image
        const unsigned int largest = unsigned(std::max(width, height));
        const double size = kernel * largest / 2.0;
        if (!(size > -1.0 && size < double(1 << 29))) {
            // Undefined conversion in the CPU code, let it deal with the frame
            return CUDA_ERROR_INVALID_VALUE;
        }
        // A box wider than the image covers it entirely, same result
        const int k = int(std::min(static_cast<unsigned int>(size), largest));
        const size_t pixels = size_t(width) * size_t(height);
        if (mode == Plain && k == 0) {
            return scope.copy(out, in, pixels * 4);
        }
        CUresult res = CUDA_SUCCESS;
        CUdeviceptr blurSource = in;
        DeviceBuffer highlights;
        DeviceBuffer table;
        if (mode >= SoftGlowScreen) {
            res = highlights.allocate(pixels * 4);
            if (res == CUDA_SUCCESS) {
                res = table.allocate(sizeof(curve));
            }
            if (res == CUDA_SUCCESS) {
                res = scope.upload(table.ptr(), curve, sizeof(curve));
            }
            if (res == CUDA_SUCCESS) {
                blurSource = highlights.ptr();
                const int count = int(pixels);
                res = scope.launch1D(KernelId::gpufx_softglow_highlights, count, in, blurSource, count, table.ptr());
            }
        }
        DeviceBuffer sums;
        if (res == CUDA_SUCCESS) {
            res = sums.allocate(pixels * 4 * sizeof(uint32_t));
        }
        // Longer runs per thread amortize the initial sum of large boxes
        int run = 32;
        while (run < 256 && run < 2 * k + 1) {
            run *= 2;
        }
        if (res == CUDA_SUCCESS) {
            res = scope.launch2D(KernelId::gpufx_square_blur_h, (width + run - 1) / run, height, 8, 16, blurSource, sums.ptr(), width, height, k, run);
        }
        if (res == CUDA_SUCCESS) {
            res = scope.launch2D(KernelId::gpufx_square_blur_v, width, (height + run - 1) / run, 32, 4, sums.ptr(), in, out, width, height, k, run, mode);
        }
        return res;
    }
};

/** Shared checks of the square blur based plans. kernel is the value given to
 *  blur_set_param_value(). */
EffectPlan squareBlurPlan(const EffectContext &ctx, double kernel, std::unique_ptr<SquareBlurJob> job)
{
    // Negative or huge sizes are undefined in the CPU code
    if (!(kernel >= 0 && kernel * std::max(ctx.width, ctx.height) < double(1 << 28))) {
        return EffectPlan::cpu();
    }
    if (job->mode == Plain && kernel == 0) {
        return EffectPlan::passThrough();
    }
    job->kernel = kernel;
    return EffectPlan::gpu(std::move(job));
}

EffectPlan planSquareBlur(const EffectContext &ctx)
{
    if (slicedByMlt(ctx)) {
        return EffectPlan::cpu();
    }
    double kernel = 0.0; // squareblur_instance_t default
    frei0rDouble(ctx, 0, "Kernel size", kernel);
    auto job = std::make_unique<SquareBlurJob>();
    job->mode = Plain;
    return squareBlurPlan(ctx, kernel, std::move(job));
}

EffectPlan planGlow(const EffectContext &ctx)
{
    if (slicedByMlt(ctx)) {
        return EffectPlan::cpu();
    }
    double blur = 0.0; // glow_instance_t default
    if (frei0rDouble(ctx, 0, "Blur", blur)) {
        blur = blur / 20.0;
    }
    auto job = std::make_unique<SquareBlurJob>();
    job->mode = Glow;
    return squareBlurPlan(ctx, blur, std::move(job));
}

EffectPlan planSoftGlow(const EffectContext &ctx)
{
    if (slicedByMlt(ctx)) {
        return EffectPlan::cpu();
    }
    // softglow_instance_t defaults. The blur only reaches the blur instance
    // when it is set, the box is empty until then.
    double blur = 0.0;
    double brightness = 0.75;
    double sharpness = 0.85;
    double blendtype = 0.0;
    frei0rDouble(ctx, 0, "blur", blur);
    frei0rDouble(ctx, 1, "brightness", brightness);
    frei0rDouble(ctx, 2, "sharpness", sharpness);
    frei0rDouble(ctx, 3, "blurblend", blendtype);
    if (!std::isfinite(brightness) || !std::isfinite(sharpness) || std::isnan(blendtype)) {
        return EffectPlan::cpu();
    }
    auto job = std::make_unique<SquareBlurJob>();
    // f0r_update(): sigmoidal transfer of the lightness, SIGMOIDAL_BASE 2 and SIGMOIDAL_RANGE 20
    for (int l = 0; l < 256; ++l) {
        double val = l / 255.0;
        val = 255.0 / (1 + exp(-(2 + (sharpness * 20)) * (val - 0.5)));
        val = val * brightness;
        if (std::isnan(val)) {
            return EffectPlan::cpu();
        }
        job->curve[l] = static_cast<uint8_t>(val < 0 ? 0 : (val > 255 ? 255 : val));
    }
    if (blendtype <= 0.33) {
        job->mode = SoftGlowScreen;
    } else if (blendtype <= 0.66) {
        job->mode = SoftGlowOverlay;
    } else {
        job->mode = SoftGlowAdd;
    }
    return squareBlurPlan(ctx, blur, std::move(job));
}

class UnsharpJob final : public EffectJob
{
public:
    int steps = 1;  // half the matrix size
    int amount = 0; // 16.16 fixed point

    CUresult run(GpuScope &scope, int width, int height, CUdeviceptr in, CUdeviceptr out) override
    {
        DeviceBuffer sums;
        CUresult res = sums.allocate(size_t(width) * size_t(height) * 4 * sizeof(uint32_t));
        if (res == CUDA_SUCCESS) {
            res = scope.launch2D(KernelId::gpufx_unsharp_h, width, height, 32, 8, in, sums.ptr(), width, height, steps);
        }
        if (res == CUDA_SUCCESS) {
            res = scope.launch2D(KernelId::gpufx_unsharp_v, width, height, 32, 8, sums.ptr(), in, out, width, height, steps, amount);
        }
        return res;
    }
};

// sharpness.c: stretch [0...1] to [min...max], in float
float mapValueForward(double v, float min, float max)
{
    return min + (max - min) * v;
}

EffectPlan planSharpness(const EffectContext &ctx)
{
    if (slicedByMlt(ctx)) {
        return EffectPlan::cpu();
    }
    // f0r_construct() defaults
    double amount = 0.0;
    int size = 3;
    double value = 0;
    if (frei0rDouble(ctx, 0, "Amount", value)) {
        amount = mapValueForward(value, -1.5, 3.5);
    }
    if (frei0rDouble(ctx, 1, "Size", value)) {
        const float mapped = mapValueForward(value, 3.0, 11.0);
        // Out of range sizes overflow the plugin's buffers or its sums
        if (!(mapped > -1.f && mapped < 64.f)) {
            return EffectPlan::cpu();
        }
        size = int(mapped);
    }
    // unsharp() copies the image when the amount is 0
    if (amount == 0) {
        return EffectPlan::passThrough();
    }
    const int steps = size / 2;
    // The plugin's 32-bit sums only hold up to 6 steps; larger amounts overflow its int math
    if (steps < 1 || steps > 6 || !(std::fabs(amount) < 100.0)) {
        return EffectPlan::cpu();
    }
    auto job = std::make_unique<UnsharpJob>();
    job->steps = steps;
    job->amount = int(amount * 65536.0);
    if (job->amount == 0) {
        return EffectPlan::passThrough();
    }
    return EffectPlan::gpu(std::move(job));
}

} // namespace

const EffectDescriptor &frei0rSquareBlurEffect()
{
    static const EffectDescriptor descriptor{"frei0r.squareblur", false, planSquareBlur};
    return descriptor;
}

const EffectDescriptor &frei0rGlowEffect()
{
    static const EffectDescriptor descriptor{"frei0r.glow", false, planGlow};
    return descriptor;
}

const EffectDescriptor &frei0rSoftGlowEffect()
{
    static const EffectDescriptor descriptor{"frei0r.softglow", false, planSoftGlow};
    return descriptor;
}

const EffectDescriptor &frei0rSharpnessEffect()
{
    static const EffectDescriptor descriptor{"frei0r.sharpness", false, planSharpness};
    return descriptor;
}

} // namespace gpufx
