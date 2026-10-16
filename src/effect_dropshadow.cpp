/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

/*
 * GPU version of MLT's "dropshadow" filter (qt module).
 *
 * The CPU filter wraps the frame in a QGraphicsPixmapItem with a
 * QGraphicsDropShadowEffect and renders the scene back onto the frame. That
 * boils down to:
 *  1. pad the item to the effect's bounding rect, which grows the image by the
 *     shadow offset and the blur radius (QGraphicsItemEffectSourcePrivate)
 *  2. draw the padded image shifted by the rounded offset into a temporary
 *     image (QPixmapDropShadowFilter::draw)
 *  3. qt_blurImage() on its alpha: when the radius is >= 4, halve the image
 *     and the radius, run the exponential blur, then draw it back scaled 2x
 *     with smooth filtering
 *  4. fill with the shadow color in SourceIn mode
 *  5. draw the shadow over the frame, then the frame over the result
 * The kernels reproduce each step with Qt's integer arithmetic, so the output
 * matches the CPU filter.
 */

#include "effect.h"
#include "kernel_math.h"

#include <cmath>

namespace gpufx {

namespace {

class DropShadowJob final : public EffectJob
{
public:
    double radius = 1.0;
    double xOffset = 8.0;
    double yOffset = 8.0;
    gpufx_u32 colorPremultiplied = 0;

    CUresult run(GpuScope &scope, int width, int height, CUdeviceptr in, CUdeviceptr out) override
    {
        // QGraphicsDropShadowEffect::draw() only draws the source in that case
        if (radius <= 0 && xOffset == 0 && yOffset == 0) {
            const int count = width * height;
            return scope.launch1D(KernelId::gpufx_over_self, count, in, out, count);
        }

        // Effective bounding rect: source rect united with the shadow rect
        // (QPixmapDropShadowFilter::boundingRectFor), then QRectF::toAlignedRect()
        const double shadowX = xOffset - radius;
        const double shadowY = yOffset - radius;
        const double shadowW = width + 2 * radius;
        const double shadowH = height + 2 * radius;
        const double left = std::min(0.0, shadowX);
        const double top = std::min(0.0, shadowY);
        const double right = std::max(double(width), shadowX + shadowW);
        const double bottom = std::max(double(height), shadowY + shadowH);
        const int padLeft = int(std::floor(left));
        const int padTop = int(std::floor(top));
        const int padWidth = int(std::ceil(left + (right - left))) - padLeft;
        const int padHeight = int(std::ceil(top + (bottom - top))) - padTop;
        if (padWidth <= 0 || padHeight <= 0) {
            return scope.copy(out, in, size_t(width) * size_t(height) * 4);
        }

        // The padded image is drawn at the offset, rounded by QPainter
        const int shiftX = qtRound(xOffset);
        const int shiftY = qtRound(yOffset);

        // qt_blurImage()
        const bool half = radius >= 4 && padWidth >= 2 && padHeight >= 2;
        const double blurRadius = half ? radius * 0.5 : radius;
        const int blurWidth = half ? padWidth / 2 : padWidth;
        const int blurHeight = half ? padHeight / 2 : padHeight;
        const int alpha = blurRadius <= 1e-5 ? 4095 : qtRound((1 << 12) * (1 - std::pow(2 * (1 / 255.0), 1 / blurRadius)));

        DeviceBuffer blur;
        CUresult res = blur.allocate(size_t(blurWidth) * size_t(blurHeight));
        if (res != CUDA_SUCCESS) {
            return res;
        }
        const CUdeviceptr blurPtr = blur.ptr();
        res = scope.launch2D(KernelId::gpufx_dropshadow_alpha, blurWidth, blurHeight, 32, 8, in, width, height, blurPtr, blurWidth, blurHeight,
                             padLeft - shiftX, padTop - shiftY, half ? 1 : 0);
        if (res == CUDA_SUCCESS) {
            res = scope.launch1D(KernelId::gpufx_expblur_rows, blurHeight, blurPtr, blurWidth, blurHeight, alpha);
        }
        if (res == CUDA_SUCCESS) {
            res = scope.launch1D(KernelId::gpufx_expblur_cols, blurWidth, blurPtr, blurWidth, blurHeight, alpha);
        }
        if (res == CUDA_SUCCESS) {
            res = scope.launch2D(KernelId::gpufx_dropshadow_composite, width, height, 32, 8, in, out, width, height, blurPtr, blurWidth, blurHeight, padLeft,
                                 padTop, half ? 1 : 0, colorPremultiplied);
        }
        return res;
    }
};

EffectPlan plan(const EffectContext &ctx)
{
    // The CPU filter keeps 16-bit images in 16 bits, only 8-bit RGBA is done here
    if (ctx.format == mlt_image_rgba64) {
        return EffectPlan::cpu();
    }
    auto job = std::make_unique<DropShadowJob>();
    job->radius = mlt_properties_anim_get_double(ctx.properties, "radius", ctx.position, ctx.length);
    job->xOffset = mlt_properties_anim_get_double(ctx.properties, "x", ctx.position, ctx.length);
    job->yOffset = mlt_properties_anim_get_double(ctx.properties, "y", ctx.position, ctx.length);
    // Negative radius and absurd offsets are left to the CPU filter
    const double limit = 65536;
    if (!std::isfinite(job->radius) || job->radius < 0 || job->radius > limit || !std::isfinite(job->xOffset) || !std::isfinite(job->yOffset) ||
        std::fabs(job->xOffset) > limit || std::fabs(job->yOffset) > limit) {
        return EffectPlan::cpu();
    }
    mlt_color color = mlt_properties_anim_get_color(ctx.properties, "color", ctx.position, ctx.length);
    color = mlt_color_convert_trc(color, mlt_properties_get(MLT_FRAME_PROPERTIES(ctx.frame), "color_trc"));
    job->colorPremultiplied = qtPremultipliedColor(color.r, color.g, color.b, color.a);
    return EffectPlan::gpu(std::move(job));
}

} // namespace

const EffectDescriptor &dropShadowEffect()
{
    static const EffectDescriptor descriptor{"dropshadow", false, plan};
    return descriptor;
}

} // namespace gpufx
