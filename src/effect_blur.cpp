/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

/*
 * GPU versions of the box blurs:
 *  - "box_blur" (MLT core, Kdenlive's "BoxBlur"): separable moving average
 *  - "boxblur" (MLT kdenlive module, Kdenlive's "Box Blur"): summed area table
 */

#include "effect.h"

#include <cmath>

namespace gpufx {

namespace {

constexpr int kRun = 32; // pixels per thread, GPUFX_BOX_RUN in the kernels

int runs(int pixels)
{
    return (pixels + kRun - 1) / kRun;
}

class BoxBlurJob final : public EffectJob
{
public:
    int hRadius = 0;
    int vRadius = 0;
    int preserveAlpha = 0;

    CUresult run(GpuScope &scope, int width, int height, CUdeviceptr in, CUdeviceptr out) override
    {
        DeviceBuffer tmp;
        CUresult res = tmp.allocate(size_t(width) * size_t(height) * 4);
        if (res != CUDA_SUCCESS) {
            return res;
        }
        const CUdeviceptr tmpPtr = tmp.ptr();
        // Like mlt_image_box_blur(), both passes always run (a radius of 0 is a copy)
        res = scope.launch2D(KernelId::gpufx_box_blur_h, runs(width), height, 8, 16, in, tmpPtr, width, height, hRadius, preserveAlpha);
        if (res == CUDA_SUCCESS) {
            res = scope.launch2D(KernelId::gpufx_box_blur_v, width, runs(height), 32, 4, tmpPtr, out, width, height, vRadius, preserveAlpha);
        }
        return res;
    }
};

EffectPlan planBoxBlur(const EffectContext &ctx)
{
    mlt_profile profile = mlt_service_profile(MLT_FILTER_SERVICE(ctx.filter));
    if (!profile) {
        return EffectPlan::cpu();
    }
    double hradius = mlt_properties_anim_get_double(ctx.properties, "hradius", ctx.position, ctx.length);
    double vradius = mlt_properties_anim_get_double(ctx.properties, "vradius", ctx.position, ctx.length);
    const int preserveAlpha = mlt_properties_get_int(ctx.properties, "preserve_alpha");
    // Same conversion as filter_box_blur.c: percent of 10% of the width, using the requested width
    const double pixelScale = double(profile->width) * mlt_profile_scale_width(profile, ctx.width) / 1000.0;
    hradius = std::max(std::round(hradius * pixelScale), 0.0);
    vradius = std::max(std::round(vradius * pixelScale), 0.0);
    if (!std::isfinite(hradius) || !std::isfinite(vradius) || hradius > 1e6 || vradius > 1e6) {
        return EffectPlan::cpu();
    }
    if (hradius == 0 && vradius == 0) {
        return EffectPlan::passThrough();
    }
    if (ctx.format == mlt_image_rgba64) {
        return EffectPlan::cpu();
    }
    auto job = std::make_unique<BoxBlurJob>();
    job->hRadius = int(hradius);
    job->vRadius = int(vradius);
    job->preserveAlpha = preserveAlpha ? 1 : 0;
    return EffectPlan::gpu(std::move(job));
}

class RectBlurJob final : public EffectJob
{
public:
    unsigned int boxWidth = 0;
    unsigned int boxHeight = 0;
    mlt_profile profile = nullptr;

    CUresult run(GpuScope &scope, int width, int height, CUdeviceptr in, CUdeviceptr out) override
    {
        // filter_boxblur.c scales the box with the size of the fetched image
        unsigned int bw = boxWidth;
        unsigned int bh = boxHeight;
        bw = (unsigned int)(bw * mlt_profile_scale_width(profile, width));
        bh = (unsigned int)(bh * mlt_profile_scale_height(profile, height));
        if (bw == 0 && bh == 0) {
            return scope.copy(out, in, size_t(width) * size_t(height) * 4);
        }
        bw = std::max(1u, bw);
        bh = std::max(1u, bh);
        const float mul = 1.f / float((bw * 2) * (bh * 2));
        DeviceBuffer sums;
        CUresult res = sums.allocate(size_t(width) * size_t(height) * 4 * sizeof(int));
        if (res != CUDA_SUCCESS) {
            return res;
        }
        const CUdeviceptr sumsPtr = sums.ptr();
        const int boxW = int(std::min(bw, 1u << 30));
        const int boxH = int(std::min(bh, 1u << 30));
        res = scope.launch2D(KernelId::gpufx_rect_blur_h, runs(width), height, 8, 16, in, sumsPtr, width, height, boxW);
        if (res == CUDA_SUCCESS) {
            res = scope.launch2D(KernelId::gpufx_rect_blur_v, width, runs(height), 32, 4, sumsPtr, out, width, height, boxH, mul);
        }
        return res;
    }
};

EffectPlan planRectBlur(const EffectContext &ctx)
{
    mlt_profile profile = mlt_service_profile(MLT_FILTER_SERVICE(ctx.filter));
    if (!profile) {
        return EffectPlan::cpu();
    }
    const double hori = mlt_properties_anim_get_double(ctx.properties, "hori", ctx.position, ctx.length);
    const double vert = mlt_properties_anim_get_double(ctx.properties, "vert", ctx.position, ctx.length);
    double factor = mlt_properties_get_int(ctx.properties, "start");
    if (mlt_properties_get(ctx.properties, "end")) {
        const double end = double(mlt_properties_get_int(ctx.properties, "end"));
        factor += (end - factor) * mlt_filter_get_progress(ctx.filter, ctx.frame);
    }
    if (mlt_properties_get(ctx.properties, "blur")) {
        factor = mlt_properties_anim_get_double(ctx.properties, "blur", ctx.position, ctx.length);
    }
    const double w = factor * hori;
    const double h = factor * vert;
    // Negative or huge values are undefined behaviour in the CPU code, leave them to it
    if (!(w >= 0 && h >= 0 && w < 1e9 && h < 1e9)) {
        return EffectPlan::cpu();
    }
    auto job = std::make_unique<RectBlurJob>();
    job->boxWidth = (unsigned int)w;
    job->boxHeight = (unsigned int)h;
    if (job->boxWidth == 0 && job->boxHeight == 0) {
        return EffectPlan::passThrough();
    }
    job->profile = profile;
    return EffectPlan::gpu(std::move(job));
}

} // namespace

const EffectDescriptor &boxBlurEffect()
{
    static const EffectDescriptor descriptor{"box_blur", false, planBoxBlur};
    return descriptor;
}

const EffectDescriptor &rectBlurEffect()
{
    static const EffectDescriptor descriptor{"boxblur", false, planRectBlur};
    return descriptor;
}

} // namespace gpufx
