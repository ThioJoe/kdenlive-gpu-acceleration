/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

/*
 * GPU version of MLT's "luma" transition (core module), which Kdenlive uses
 * for its dissolves, same-track mixes and wipes. Depending on its properties
 * and on the format asked for, transition_get_image() takes one of several
 * code paths; these run on the GPU with the same arithmetic:
 *  - without a luma map and with alpha_over: dissolve_rgba32()
 *  - without a luma map, without alpha_over (Kdenlive's default): both images
 *    converted to YUV 4:2:2, dissolve_yuv422() with composite_line_yuv(),
 *    and the result converted back to RGBA for the caller
 *  - with a luma map (wipes): luma_composite_rgba32(), progressive frames
 * The other cases (interlaced or inverted wipes, luma maps from video
 * clips, other formats than RGBA) stay on the CPU.
 */

#include "bridge.h"
#include "convert.h"
#include "effect.h"
#include "kernel_math.h"
#include "sws_port.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace gpufx {

namespace {

enum class LumaMode { DissolveRgba, DissolveYuv, Wipe };

class LumaJob final : public TransitionJob
{
public:
    LumaMode mode = LumaMode::DissolveRgba;
    // Dissolves: the weight of the top image
    float weight = 0.0f;
    // Wipes: the luma map and the position of the wipe
    std::vector<uint16_t> map;
    int mapWidth = 0;
    int mapHeight = 0;
    float position = 0.0f;
    float softness = 0.0f;
    gpufx_u32 fixedPosition = 0;
    gpufx_u32 fixedSoftness = 0;
    bool addAlpha = false;
    // YUV dissolves: the colorspaces both images are converted to
    int bottomColorspace = 0;
    int topColorspace = 0;

    void beforeFetch(mlt_frame aFrame, mlt_frame bFrame) override
    {
        if (const char *distort = mlt_properties_get(MLT_FRAME_PROPERTIES(aFrame), "distort")) {
            mlt_properties_set(MLT_FRAME_PROPERTIES(bFrame), "distort", distort);
        }
    }

    // The top image is never returned as it is
    void passFrameProperties(mlt_frame, mlt_frame) override { }
    int passThrough(mlt_frame, mlt_frame, uint8_t *, mlt_image_format, int, int, uint8_t **, mlt_image_format *, int *, int *) override { return 1; }

    bool prepare(int topWidth, int topHeight, int bottomWidth, int bottomHeight) override
    {
        // The CPU transition fails on images of different sizes, or blends
        // the part they share
        if (topWidth != bottomWidth || topHeight != bottomHeight || bottomWidth <= 0 || bottomHeight <= 0) {
            return false;
        }
        m_width = bottomWidth;
        m_height = bottomHeight;
        return mode != LumaMode::DissolveYuv || !(m_width & 1);
    }

    CUresult run(GpuScope &scope, CUdeviceptr top, CUdeviceptr bottom, CUdeviceptr out) override
    {
        switch (mode) {
        case LumaMode::DissolveRgba:
            return scope.launch1D(KernelId::gpufx_luma_dissolve_rgba, m_width * m_height, top, bottom, out, m_width * m_height, weight);
        case LumaMode::DissolveYuv:
            return runDissolveYuv(scope, top, bottom, out);
        case LumaMode::Wipe:
            return runWipe(scope, top, bottom, out);
        }
        return CUDA_ERROR_INVALID_VALUE;
    }

    void finishFrame(mlt_frame aFrame) override
    {
        if (mode == LumaMode::DissolveYuv) {
            // The frame properties set by the conversion back to RGBA
            mlt_properties_set_int(MLT_FRAME_PROPERTIES(aFrame), "colorspace", mlt_colorspace_rgb);
            mlt_properties_set_int(MLT_FRAME_PROPERTIES(aFrame), "full_range", 1);
        }
    }

private:
    CUresult runDissolveYuv(GpuScope &scope, CUdeviceptr top, CUdeviceptr bottom, CUdeviceptr out)
    {
        const size_t pixels = size_t(m_width) * size_t(m_height);
        std::vector<int> columns;
        sws::chroma422Columns(m_width, columns);
        const size_t tableBytes = columns.size() * sizeof(int);
        const size_t yuvOffset = (tableBytes + 255) & ~size_t(255);
        const size_t alphaOffset = yuvOffset + ((pixels * 2 + 255) & ~size_t(255));
        const sws::RgbToYuv a = sws::rgbToYuv(sws::colorspaceTable(bottomColorspace));
        const sws::RgbToYuv b = sws::rgbToYuv(sws::colorspaceTable(topColorspace));
        // Converted back from the bottom image's YUV, limited range
        const sws::YuvToRgb back = sws::yuvToRgb(sws::colorspaceTable(bottomColorspace), false);
        // composite_line_yuv() blends lines of 8 pixels or more with SSE2,
        // except for the last width % 8 pixels
        const int sseWidth = m_width > 7 ? m_width - m_width % 8 : 0;
        const int fixedWeight = int(weight * (1 << 16));

        DeviceBuffer work;
        CUresult res = work.allocate(alphaOffset + pixels);
        if (res == CUDA_SUCCESS) {
            res = scope.upload(work.ptr(), columns.data(), tableBytes);
        }
        if (res == CUDA_SUCCESS) {
            res = scope.launch2D(KernelId::gpufx_luma_dissolve_yuyv, m_width / 2, m_height, 32, 8, top, bottom, work.ptr() + yuvOffset,
                                 work.ptr() + alphaOffset, m_width, m_height, fixedWeight, sseWidth, a.ry, a.gy, a.by, a.ru, a.gu, a.bu, a.rv, a.gv, a.bv, b.ry,
                                 b.gy, b.by, b.ru, b.gu, b.bu, b.rv, b.gv, b.bv);
        }
        if (res == CUDA_SUCCESS) {
            res = scope.launch2D(KernelId::gpufx_yuyv422_to_rgba, m_width, m_height, 32, 8, work.ptr() + yuvOffset, work.ptr(), bottom,
                                 work.ptr() + alphaOffset, out, m_width, m_height, back.yOffset, back.yCoeff, back.v2r, back.v2g, back.u2g, back.u2b);
        }
        return res;
    }

    CUresult runWipe(GpuScope &scope, CUdeviceptr top, CUdeviceptr bottom, CUdeviceptr out)
    {
        const size_t mapBytes = map.size() * sizeof(uint16_t);
        const size_t flagOffset = (mapBytes + 255) & ~size_t(255);
        const int count = m_width * m_height;
        const int xDiff = (mapWidth << 16) / m_width;
        const int yDiff = (mapHeight << 16) / m_height;
        DeviceBuffer work;
        CUresult res = work.allocate(flagOffset + sizeof(int));
        if (res == CUDA_SUCCESS) {
            res = scope.upload(work.ptr(), map.data(), mapBytes);
        }
        // Whether either image has transparency (mlt_image_is_opaque())
        if (res == CUDA_SUCCESS) {
            res = scope.fill(work.ptr() + flagOffset, 0, sizeof(int));
        }
        if (res == CUDA_SUCCESS) {
            res = scope.launch1D(KernelId::gpufx_alpha_check, count, top, count, work.ptr() + flagOffset);
        }
        if (res == CUDA_SUCCESS) {
            res = scope.launch1D(KernelId::gpufx_alpha_check, count, bottom, count, work.ptr() + flagOffset);
        }
        if (res == CUDA_SUCCESS) {
            res = scope.launch2D(KernelId::gpufx_luma_wipe_rgba, m_width, m_height, 32, 8, top, bottom, out, m_width, m_height, work.ptr(), mapWidth, xDiff,
                                 yDiff, position, softness, fixedPosition, fixedSoftness, work.ptr() + flagOffset, addAlpha ? 1 : 0);
        }
        return res;
    }

    int m_width = 0;
    int m_height = 0;
};

/** The colorspace avcolour_space converts a frame's RGBA image to YUV in */
int yuvColorspace(mlt_frame frame)
{
    mlt_producer producer = mlt_frame_get_original_producer(frame);
    mlt_profile profile = producer ? mlt_service_profile(MLT_PRODUCER_SERVICE(producer)) : nullptr;
    return profile ? profile->colorspace : mlt_colorspace_bt601;
}

/** Follows transition_get_image() up to the compositing */
TransitionPlan planLuma(const TransitionContext &ctx)
{
    // The CPU transition dissolves other formats in YUV
    if (ctx.format != mlt_image_rgba) {
        return TransitionPlan::cpu();
    }
    mlt_properties properties = ctx.properties;
    mlt_properties aProperties = MLT_FRAME_PROPERTIES(ctx.aFrame);
    mlt_properties bProperties = MLT_FRAME_PROPERTIES(ctx.bFrame);
    auto job = std::make_unique<LumaJob>();

    // The luma map, as the CPU transition finds it
    mlt_service service = MLT_TRANSITION_SERVICE(ctx.transition);
    mlt_service_lock(service);
    int mapWidth = mlt_properties_get_int(properties, "width");
    int mapHeight = mlt_properties_get_int(properties, "height");
    auto *bitmap = static_cast<const uint16_t *>(mlt_properties_get_data(properties, "bitmap", nullptr));
    const char *current = mlt_properties_get(properties, "_resource");
    const char *resource = mlt_properties_get(properties, "resource");
    // A video clip as luma map
    bool cpu = mlt_properties_get_data(properties, "producer", nullptr) != nullptr;
    if (mapWidth == 0 || mapHeight == 0) {
        mapWidth = ctx.width;
        mapHeight = ctx.height;
    }
    if (!cpu && resource && (!current || strcmp(resource, current))) {
        if (*resource) {
            // Loaded by the CPU transition
            cpu = true;
        } else {
            // Dropped by the CPU transition
            bitmap = nullptr;
        }
    }
    const bool wipe = bitmap && mapWidth > 0 && mapHeight > 0;
    if (!cpu && wipe && mapWidth < 32768 && mapHeight < 32768) {
        job->map.assign(bitmap, bitmap + size_t(mapWidth) * size_t(mapHeight));
    }
    mlt_service_unlock(service);
    if (cpu || (wipe && job->map.empty())) {
        return TransitionPlan::cpu();
    }

    float mix = mlt_transition_get_progress(ctx.transition, ctx.aFrame);
    float frameDelta = mlt_transition_get_progress_delta(ctx.transition, ctx.aFrame);
    const float softness = mlt_properties_get_double(properties, "softness");
    const int progressive = mlt_properties_get_int(aProperties, "consumer.progressive") || mlt_properties_get_int(properties, "progressive") ||
                            mlt_properties_get_int(bProperties, "luma.progressive");
    const int reverse = mlt_properties_get_int(properties, "reverse");
    const int invert = mlt_properties_get_int(properties, "invert");
    const int alphaOver = mlt_properties_get_int(properties, "alpha_over");
    if (mix >= 1.0) {
        mix -= floor(mix);
    }
    if (mlt_properties_get(properties, "fixed")) {
        mix = mlt_properties_get_double(properties, "fixed");
    }
    if (wipe) {
        // With invert, the result is the top frame's image. Fields are wiped
        // one after the other.
        if (invert || !progressive) {
            return TransitionPlan::cpu();
        }
        mix = reverse ? 1 - mix : mix;
        frameDelta *= reverse ? -1.0 : 1.0;
        const int fieldOrder = -1;
        const float position = (mix + ((fieldOrder == 0 ? 1 : 0) * frameDelta * 0.5f)) * (1.f + softness);
        // Keeps the fixed point conversions in range
        if (!(softness >= 0.0f && softness <= 1.0f && position >= 0.0f && position <= 2.0f)) {
            return TransitionPlan::cpu();
        }
        // MLT 7.42 changed the alpha of translucent images without alpha_over,
        // 7.41 development versions may have either
        const int version = mlt_version_get_int();
        if (!alphaOver && version >= 0x072900 && version < 0x072a00) {
            return TransitionPlan::cpu();
        }
        job->mode = LumaMode::Wipe;
        job->mapWidth = mapWidth;
        job->mapHeight = mapHeight;
        job->position = position;
        job->softness = softness;
        job->fixedPosition = uint32_t((1 << 16) * position);
        job->fixedSoftness = uint32_t(softness * (1 << 16));
        job->addAlpha = !alphaOver && version >= 0x072a00;
    } else {
        if (mlt_color_trc_linear == mlt_image_color_trc_id(mlt_properties_get(aProperties, "consumer.mlt_color_trc"))) {
            // Make progress appear linear using a gamma curve similar to sRGB or Rec. 709
            if (reverse || invert) {
                mix = 1.0 - pow(1.0 - mix, 2.0);
            } else {
                mix = pow(mix, 2.0);
            }
        }
        mix = (reverse || invert) ? 1 - mix : mix;
        job->weight = mix;
        if (alphaOver) {
            job->mode = LumaMode::DissolveRgba;
        } else {
            // Both images are taken in RGBA and converted to YUV by
            // avcolour_space when fix_background_alpha is set, the result is
            // converted back to RGBA for the caller
            if (!mlt_properties_get_int(properties, "fix_background_alpha") || !(mix >= 0.0f && mix <= 1.0f)) {
                return TransitionPlan::cpu();
            }
            if (!convertsWithAvcolourSpace(ctx.aFrame) || !convertsWithAvcolourSpace(ctx.bFrame) ||
                mlt_image_full_range(mlt_properties_get(aProperties, "consumer.color_range")) ||
                mlt_image_full_range(mlt_properties_get(bProperties, "consumer.color_range"))) {
                return TransitionPlan::cpu();
            }
            job->mode = LumaMode::DissolveYuv;
            job->bottomColorspace = yuvColorspace(ctx.aFrame);
            job->topColorspace = yuvColorspace(ctx.bFrame);
        }
    }
    // Copying both images to the GPU and the result back costs about as much
    // as blending them in RGBA on the CPU: those only run on the GPU when an
    // image is there anyway. The YUV dissolve also saves three conversions.
    if (job->mode != LumaMode::DissolveYuv && !ctx.topOnGpu && !ctx.bottomOnGpu && !ctx.callerOnGpu) {
        return TransitionPlan::cpu();
    }
    job->topFormat = mlt_image_rgba;
    job->topWidth = ctx.width;
    job->topHeight = ctx.height;
    job->bottomWritable = 1;
    return TransitionPlan::gpu(std::move(job));
}

} // namespace

const TransitionDescriptor &lumaTransition()
{
    static const TransitionDescriptor descriptor{"luma", planLuma};
    return descriptor;
}

} // namespace gpufx
