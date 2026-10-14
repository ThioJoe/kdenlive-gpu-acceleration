/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

/*
 * GPU version of MLT's "qtblend" filter, Kdenlive's Transform effect.
 *
 * qtblend draws the frame with QPainter onto a transparent image: a transform
 * built from the effect rectangle and rotation, and, when the consumer scales
 * with interpolation ("consumer.rescale" other than nearest), bilinear
 * sampling and antialiased edges. The geometry below follows
 * filter_qtblend.cpp of MLT 7.38 and of MLT 7.40 and later, which scales the
 * source differently. The drawing follows Qt's raster engine (qt_raster.cpp
 * for the coverage and the span handling, the gpufx_qt_draw_image kernel for
 * the pixels). Only the default composition mode (alpha blend) and 8-bit
 * processing are handled, everything else stays on the CPU.
 */

#include "effect.h"
#include "gpufx_log.h"
#include "mlt_compat.h"
#include "qt_draw.h"
#include "qt_raster.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gpufx {

namespace {

constexpr int kMaxDimension = 16000; // MLT_QTBLEND_MAX_DIMENSION
constexpr double kMipmapStep = 0.85; // MLT_QT_MIPMAP_STEP

int qRound(double d)
{
    return qt::round(d);
}

template <typename T> T qBound(T min, T value, T max)
{
    return std::max(min, std::min(max, value));
}

/** Frame property changed by the plan, with its previous value */
struct SavedProperty
{
    const char *name;
    bool existed;
    double value;
};

class TransformJob final : public EffectJob
{
public:
    // Decided by the plan, before the source image is fetched
    bool scaleWithQPainter = true; // MLT >= 7.40: the producer scales close to the target, QPainter does the rest
    mlt_rect rect{};
    double consumerAr = 1;
    double sourceAr = 1;
    double sourceDar = 1;
    bool distort = false;
    qt::Transform transform; // translation and rotation
    double opacity = 1;
    bool smooth = false;
    uint32_t background = 0;
    bool setAspectRatio = false;
    bool storeScaling = false; // first qtblend of the frame: it records the consumer scaling
    double scaleX = 1;
    double scaleY = 1;
    std::vector<SavedProperty> saved;

    // Computed by prepare()
    QtImageDrawing drawing;

    bool prepare(int inputWidth, int inputHeight) override;

    CUresult run(GpuScope &, int, int, CUdeviceptr, CUdeviceptr) override { return CUDA_ERROR_INVALID_VALUE; }

    CUresult runResized(GpuScope &scope, int, int, CUdeviceptr in, int, int, CUdeviceptr out) override { return drawing.draw(scope, in, 0, background, out); }

    void beforeFetch(mlt_frame frame) override
    {
        // Same frame changes as the CPU filter, before the source is fetched
        mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
        if (setAspectRatio) {
            mlt_frame_set_aspect_ratio(frame, consumerAr);
        }
        if (storeScaling) {
            for (const char *name : {"qtblend_scalingx", "qtblend_scalingy"}) {
                saved.push_back({name, mlt_properties_get(properties, name) != nullptr, mlt_properties_get_double(properties, name)});
            }
            mlt_properties_set_double(properties, "qtblend_scalingx", scaleX);
            mlt_properties_set_double(properties, "qtblend_scalingy", scaleY);
        }
    }

    void restoreFrame(mlt_frame frame) override
    {
        mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
        for (const SavedProperty &property : saved) {
            if (property.existed) {
                mlt_properties_set_double(properties, property.name, property.value);
            } else {
                mlt_properties_set(properties, property.name, nullptr);
            }
        }
        saved.clear();
    }

    int outputWidth = 0;
    int outputHeight = 0;
};

/** The rest of filter_get_image() once the source is fetched: scaling of the
 *  source into the rectangle. Returns false where MLT 7.38 scales the image
 *  with QImage::scaled() instead of QPainter. */
bool TransformJob::prepare(int bWidth, int bHeight)
{
    qt::Transform matrix = transform;
    if (scaleWithQPainter) {
        if (distort) {
            if (rect.w != bWidth || rect.h != bHeight) {
                matrix.scale(rect.w / bWidth, rect.h / bHeight);
            }
        } else {
            double scale;
            const double resizeDar = rect.w * consumerAr / rect.h;
            if (sourceDar >= resizeDar) {
                scale = rect.w / bWidth;
            } else {
                scale = rect.h / bHeight;
            }
            // Center image in rect
            matrix.translate((rect.w - (bWidth * scale)) / 2.0, (rect.h - (bHeight * scale)) / 2.0);
            matrix.scale(scale, scale);
        }
    } else {
        if (distort) {
            if (rect.w != bWidth || rect.h != bHeight) {
                if (rect.w < bWidth || rect.h < bHeight) {
                    return false;
                }
                matrix.scale(rect.w / bWidth, rect.h / bHeight);
            }
        } else {
            double scale;
            const double resizeDar = rect.w * consumerAr / rect.h;
            if (sourceDar >= resizeDar) {
                scale = rect.w / bWidth;
            } else {
                scale = rect.h / bHeight * sourceAr;
            }
            if (scale < 1.) {
                return false;
            }
            matrix.translate((rect.w - (bWidth * scale)) / 2.0, (rect.h - (bHeight * scale)) / 2.0);
            if (scale > 1.) {
                matrix.scale(scale, scale);
            }
        }
    }
    return drawing.prepare(bWidth, bHeight, outputWidth, outputHeight, matrix, opacity, smooth);
}

/** filter_get_image() of filter_qtblend.cpp up to the fetch of the source */
EffectPlan planTransform(const EffectContext &ctx)
{
    const int version = mlt_version_get_int();
    // The geometry of MLT 7.38 to 7.44, newer versions are left to the CPU until checked
    static const bool forced = getenv("MLT_GPUFX_QTBLEND") && !strcmp(getenv("MLT_GPUFX_QTBLEND"), "force");
    if (version < 0x072600 || (version >= 0x072d00 && !forced)) {
        return EffectPlan::cpu();
    }
    // choose_image_format(): 16-bit processing paints in RGBA64
    if (ctx.format == mlt_image_rgba64) {
        return EffectPlan::cpu();
    }
    mlt_properties properties = ctx.properties;
    // Other composition modes than alpha blend
    if (mlt_properties_get_int(properties, "compositing") != 0) {
        return EffectPlan::cpu();
    }
    mlt_profile profile = mlt_service_profile(MLT_FILTER_SERVICE(ctx.filter));
    if (!profile || profile->width <= 0 || profile->height <= 0) {
        return EffectPlan::cpu();
    }
    mlt_properties frameProperties = MLT_FRAME_PROPERTIES(ctx.frame);
    int width = ctx.width;
    int height = ctx.height;

    auto job = std::make_unique<TransformJob>();
    job->scaleWithQPainter = version >= 0x072800;

    const int normalizedWidth = profile->width;
    const int normalizedHeight = profile->height;
    const double consumerAr = mlt_profile_sar(profile);

    mlt_rect rect = {0, 0, double(normalizedWidth), double(normalizedHeight), 1.0};
    int bWidth = mlt_properties_get_int(frameProperties, "meta.media.width");
    int bHeight = mlt_properties_get_int(frameProperties, "meta.media.height");
    const bool distort = mlt_properties_get_int(properties, "distort");

    if (bHeight == 0) {
        bWidth = normalizedWidth;
        bHeight = normalizedHeight;
    }
    // Special case - aspect_ratio = 0
    double bAr = mlt_frame_get_aspect_ratio(ctx.frame);
    const bool setAspectRatio = bAr == 0;
    if (setAspectRatio) {
        bAr = consumerAr;
    }
    const double bDar = bAr * bWidth / bHeight;

    // A qtblend filter below this one already set the consumer scaling
    const double qtblendScaleX = std::min(1., mlt_properties_get_double(frameProperties, "qtblend_scalingx"));
    if (mlt_properties_get(properties, "rect")) {
        rect = mlt_properties_anim_get_rect(properties, "rect", ctx.position, ctx.length);
        if (strchr(mlt_properties_get(properties, "rect"), '%')) {
            rect.x *= normalizedWidth;
            rect.y *= normalizedHeight;
            rect.w *= normalizedWidth;
            rect.h *= normalizedHeight;
        }
    }
    if (qtblendScaleX > 0.) {
        const double qtblendScaleY = std::min(1., mlt_properties_get_double(frameProperties, "qtblend_scalingy"));
        int tmpWidth = bWidth;
        int tmpHeight = bHeight;
        const double scaleFactor = std::max(width / rect.w, height / rect.h);
        if (scaleFactor > 1.) {
            tmpWidth *= scaleFactor;
            tmpHeight *= scaleFactor;
        }
        if (consumerAr * normalizedHeight / normalizedWidth < 1.) {
            width = qBound(qRound(normalizedWidth * qtblendScaleX), tmpWidth, kMaxDimension);
            height = qRound(width * consumerAr * normalizedHeight / normalizedWidth);
        } else {
            height = qBound(qRound(normalizedHeight * qtblendScaleY), tmpHeight, kMaxDimension);
            width = qRound(height * normalizedWidth / normalizedHeight / consumerAr);
        }
        double scale = double(width) / normalizedWidth;
        if (scale != 1.0) {
            rect.x *= scale;
            rect.w *= scale;
        }
        scale = double(height) / normalizedHeight;
        if (scale != 1.0) {
            rect.y *= scale;
            rect.h *= scale;
        }
    } else {
        // First instance of a qtblend filter: check if the requested frame size is scaled
        double scalex = mlt_profile_scale_width(profile, width);
        const double scaley = mlt_profile_scale_height(profile, height);
        job->storeScaling = true;
        job->scaleX = scalex;
        job->scaleY = scaley;
        if (scalex != 1.0) {
            rect.x *= scalex;
            rect.w *= scalex;
            if (bWidth < normalizedWidth) {
                // Adjust scale so that we don't request too small images
                scalex = qBound(scalex, normalizedWidth * scalex / bWidth, 1.);
            }
            // Apply consumer scaling to the source image
            if (scalex < 1.) {
                bWidth *= scalex;
                bHeight *= scalex;
            }
        }
        if (scaley != 1.0) {
            rect.y *= scaley;
            rect.h *= scaley;
        }
    }

    if (job->scaleWithQPainter) {
        // normalize_mlt_source_size()
        if (bAr > 0.0 && consumerAr > 0.0 && bAr != consumerAr && bHeight > 0) {
            bWidth = qRound(bWidth * bAr / consumerAr);
        }
        bWidth = std::max(1, bWidth);
        // Let the producer downscale (adjust_mlt_mipmap_size())
        if (rect.w > 0 && rect.h > 0 && bWidth > 0 && bHeight > 0) {
            double scaleTarget;
            if (distort) {
                scaleTarget = std::max(rect.w / bWidth, rect.h / bHeight);
            } else {
                const double resizeDar = rect.w * consumerAr / rect.h;
                if (bDar >= resizeDar) {
                    scaleTarget = rect.w / bWidth;
                } else {
                    scaleTarget = rect.h / bHeight;
                }
            }
            double mipmapScale = 1.0;
            while (mipmapScale * kMipmapStep >= scaleTarget && mipmapScale > 0.001) {
                mipmapScale *= kMipmapStep;
            }
            if (mipmapScale < 1.0) {
                bWidth = std::max(1, qRound(bWidth * mipmapScale));
                bHeight = std::max(1, qRound(bHeight * mipmapScale));
            }
        }
    }

    job->transform.translate(rect.x, rect.y);
    bool hasAlpha = rect.o < 1 || rect.x != 0 || rect.y != 0 || rect.w != width || rect.h != height || rect.w / bDar < height || rect.h * bDar < width ||
                    bWidth != width || bHeight != height;
    if (mlt_properties_get(properties, "rotation")) {
        const double angle = mlt_properties_anim_get_double(properties, "rotation", ctx.position, ctx.length);
        if (angle != 0.0) {
            if (mlt_properties_get(properties, "rotate_anchor")) {
                const mlt_rect anchor = mlt_properties_anim_get_rect(properties, "rotate_anchor", ctx.position, ctx.length);
                const double anchorX = anchor.x * rect.w;
                const double anchorY = anchor.y * rect.h;
                job->transform.translate(anchorX, anchorY);
                job->transform.rotate(angle);
                job->transform.translate(-anchorX, -anchorY);
            } else if (mlt_properties_get_int(properties, "rotate_center")) {
                job->transform.translate(rect.w / 2.0, rect.h / 2.0);
                job->transform.rotate(angle);
                job->transform.translate(-rect.w / 2.0, -rect.h / 2.0);
            } else {
                job->transform.rotate(angle);
            }
            hasAlpha = true;
        }
    }
    // Nothing to paint: the CPU filter passes the image through or paints it as is
    if (!hasAlpha) {
        return EffectPlan::cpu();
    }
    if (!(rect.w > 0 && rect.h > 0) || !std::isfinite(rect.x) || !std::isfinite(rect.y) || !std::isfinite(rect.w) || !std::isfinite(rect.h) || bWidth <= 0 ||
        bHeight <= 0 || width <= 0 || height <= 0) {
        return EffectPlan::cpu();
    }

    const char *interpolation = mlt_properties_get(frameProperties, "consumer.rescale");
    job->smooth = interpolation && strcmp(interpolation, "nearest") && strcmp(interpolation, "neighbor");
    job->rect = rect;
    job->consumerAr = consumerAr;
    job->sourceAr = bAr;
    job->sourceDar = bDar;
    job->distort = distort;
    job->opacity = rect.o;
    job->background = uint32_t(mlt_properties_get_int(properties, "background_color"));
    job->outputWidth = width;
    job->outputHeight = height;

    job->setAspectRatio = setAspectRatio;

    EffectPlan plan = EffectPlan::gpu(std::move(job));
    plan.inputWidth = bWidth;
    plan.inputHeight = bHeight;
    plan.outputWidth = width;
    plan.outputHeight = height;
    return plan;
}

/* ==========================================================================
 * Track compositing: MLT's "qtblend" transition (transition_qtblend.cpp)
 * draws the top frame onto the bottom frame with QPainter.
 * ========================================================================== */

bool isRgbaFormat(mlt_image_format format)
{
    return format == mlt_image_rgba || format == mlt_image_rgba64;
}

class CompositeJob final : public TransitionJob
{
public:
    // Decided by the plan, before the top image is fetched
    int version = 0;
    mlt_rect rect{};
    double consumerAr = 1;
    double sourceDar = 1;
    double geometryDar = 1;
    double transformScale = 1;
    bool distort = false;
    qt::Transform transform; // translation and rotation
    double opacity = 1;
    bool smooth = false;
    bool setAspectRatio = false;
    int requestWidth = 0;
    int requestHeight = 0;

    // Computed by prepare()
    QtImageDrawing drawing;

    void beforeFetch(mlt_frame, mlt_frame bFrame) override
    {
        // Same frame changes as the CPU transition, before the top image is fetched
        if (setAspectRatio) {
            mlt_frame_set_aspect_ratio(bFrame, consumerAr);
        }
        // This is not a field-aware transform.
        mlt_properties_set_int(MLT_FRAME_PROPERTIES(bFrame), "consumer.progressive", 1);
    }

    void passFrameProperties(mlt_frame aFrame, mlt_frame bFrame) override
    {
        mlt_properties_pass_list(MLT_FRAME_PROPERTIES(aFrame), MLT_FRAME_PROPERTIES(bFrame),
                                 "progressive,distort,colorspace,full_range,force_full_luma,top_field_first,color_trc");
    }

    int passThrough(mlt_frame aFrame, mlt_frame bFrame, uint8_t *top, mlt_image_format topFormat, int topWidth, int topHeight, uint8_t **image,
                    mlt_image_format *format, int *width, int *height) override
    {
        passFrameProperties(aFrame, bFrame);
        *format = topFormat;
        // Prepare output image
        if (hasConvertImage(bFrame) && (topWidth != requestWidth || topHeight != requestHeight)) {
            mlt_properties bProperties = MLT_FRAME_PROPERTIES(bFrame);
            mlt_properties_set_int(bProperties, "convert_image_width", requestWidth);
            mlt_properties_set_int(bProperties, "convert_image_height", requestHeight);
            convertImage(bFrame, &top, format, *format);
            *width = requestWidth;
            *height = requestHeight;
        } else {
            *width = topWidth;
            *height = topHeight;
        }
        *image = top;
        return 0;
    }

    bool prepare(int bWidth, int bHeight, int bottomWidth, int bottomHeight) override
    {
        // MLT 7.42 converts a bottom image of another size than requested.
        // The 7.41 development versions do either.
        if (version >= 0x072900 && (bottomWidth != requestWidth || bottomHeight != requestHeight)) {
            return false;
        }
        qt::Transform matrix = transform;
        if (version >= 0x072800) {
            if (distort) {
                if (bWidth != 0 && bHeight != 0) {
                    matrix.scale(rect.w / bWidth, rect.h / bHeight);
                }
            } else if (rect.w > 0 && rect.h > 0) {
                const double resizeDar = rect.w * consumerAr / rect.h;
                double scale;
                if (sourceDar >= resizeDar) {
                    scale = rect.w / bWidth;
                    if (sourceDar < geometryDar) {
                        scale *= transformScale;
                    }
                } else {
                    scale = rect.h / bHeight;
                    if (sourceDar > geometryDar) {
                        scale *= transformScale;
                    }
                }
                // Center image in rect
                matrix.translate((rect.w - (bWidth * scale)) / 2.0, (rect.h - (bHeight * scale)) / 2.0);
                // Use QPainter scaling for everything
                matrix.scale(scale, scale);
            }
        } else {
            // MLT 7.38 scales down with QImage::scaled(), left to the CPU
            if (distort) {
                if (bWidth != 0 && bHeight != 0) {
                    if (rect.w < bWidth || rect.h < bHeight) {
                        return false;
                    }
                    matrix.scale(rect.w / bWidth, rect.h / bHeight);
                }
            } else if (rect.w > 0 && rect.h > 0) {
                const double resizeDar = rect.w * consumerAr / rect.h;
                double scale;
                if (sourceDar > resizeDar) {
                    scale = rect.w / bWidth;
                    if (sourceDar < geometryDar) {
                        scale *= transformScale;
                    }
                } else {
                    scale = rect.h / bHeight;
                    if (sourceDar > geometryDar) {
                        scale *= transformScale;
                    }
                }
                if (scale < 1.) {
                    return false;
                }
                matrix.translate((rect.w - (bWidth * scale)) / 2.0, (rect.h - (bHeight * scale)) / 2.0);
                if (scale > 1.) {
                    matrix.scale(scale, scale);
                }
            }
        }
        return drawing.prepare(bWidth, bHeight, bottomWidth, bottomHeight, matrix, opacity, smooth);
    }

    CUresult run(GpuScope &scope, CUdeviceptr top, CUdeviceptr bottom, CUdeviceptr out) override { return drawing.draw(scope, top, bottom, 0, out); }
};

/** get_image() of transition_qtblend.cpp up to the fetch of the top image */
TransitionPlan planComposite(const TransitionContext &ctx)
{
    const int version = mlt_version_get_int();
    static const bool forced = getenv("MLT_GPUFX_QTBLEND") && !strcmp(getenv("MLT_GPUFX_QTBLEND"), "force");
    if (version < 0x072600 || (version >= 0x072d00 && !forced)) {
        return TransitionPlan::cpu();
    }
    // Copying images to the GPU and back costs about as much as compositing
    // them on the CPU: only worth it when an image is on the GPU anyway, or
    // when the GPU also saves the conversion of the result
    if (!ctx.topOnGpu && !ctx.bottomOnGpu && !ctx.callerOnGpu && !ctx.convertsOnGpu) {
        return TransitionPlan::cpu();
    }
    mlt_image_format format = ctx.format;
    if (format == mlt_image_rgba64) {
        return TransitionPlan::cpu();
    }
    const bool v740 = version >= 0x072800;
    const bool v741 = version >= 0x072900;
    const bool v742 = version >= 0x072a00;
    mlt_properties properties = ctx.properties;
    mlt_properties bProperties = MLT_FRAME_PROPERTIES(ctx.bFrame);
    const int width = ctx.width;
    const int height = ctx.height;

    // hasAlpha indicates whether the source material has an alpha channel
    bool hasAlpha = isRgbaFormat(format);
    int mode = mlt_properties_get_int(properties, "compositing");
    if (mlt_properties_exists(bProperties, "qtblend.mode")) {
        // Override from qtblend_mode filter
        mode = mlt_properties_get_int(bProperties, "qtblend.mode");
    }
    if (mode != 0) {
        return TransitionPlan::cpu();
    }
    mlt_profile profile = mlt_service_profile(MLT_TRANSITION_SERVICE(ctx.transition));
    if (!profile || profile->width <= 0 || profile->height <= 0 || width <= 0 || height <= 0) {
        return TransitionPlan::cpu();
    }
    int bWidth = mlt_properties_get_int(bProperties, "meta.media.width");
    int bHeight = mlt_properties_get_int(bProperties, "meta.media.height");
    const int normalizedWidth = profile->width;
    const int normalizedHeight = profile->height;
    bool parAlreadyNormalized = false;

    mlt_rect rect = {0, 0, double(normalizedWidth), double(normalizedHeight), 1.0};
    const bool distort = mlt_properties_get_int(properties, "distort");
    const double consumerAr = mlt_profile_sar(profile);

    // Check the producer's native format before fetching image
    const auto sourceFormat = mlt_image_format(mlt_properties_get_int(bProperties, "format"));
    if (isRgbaFormat(sourceFormat) || sourceFormat == mlt_image_rgb) {
        hasAlpha = isRgbaFormat(sourceFormat);
        format = mlt_image_rgba;
    }
    if (bHeight == 0) {
        bWidth = width;
        bHeight = height;
        parAlreadyNormalized = true;
    }
    // Special case - aspect_ratio = 0. MLT 7.38 fixes it after computing the source aspect.
    const double frameAr = mlt_frame_get_aspect_ratio(ctx.bFrame);
    const bool setAspectRatio = frameAr == 0;
    const double bAr = (setAspectRatio && v740) ? consumerAr : frameAr;
    const double bDar = bAr * bWidth / bHeight;
    const double geometryDar = consumerAr * normalizedWidth / normalizedHeight;
    double transformScale = 1.;
    // forceAlpha is true if some operation makes it mandatory to perform the alpha compositing, like padding or scaling
    bool forceAlpha = bDar != geometryDar;
    if (forceAlpha) {
        // Ensure we ask for an image with same dar as consumer
        bWidth = geometryDar * bHeight;
    }
    if (!distort && (bHeight < height || bWidth < width)) {
        // Source image is smaller than profile, request full frame
        if (bDar > geometryDar) {
            transformScale = bDar / geometryDar;
        } else {
            transformScale = geometryDar / bDar;
        }
        bWidth = width;
        bHeight = height;
        parAlreadyNormalized = true;
    }
    if (mlt_properties_get(properties, "rect")) {
        rect = mlt_properties_anim_get_rect(properties, "rect", ctx.position, ctx.length);
        if (strchr(mlt_properties_get(properties, "rect"), '%')) {
            // We have percentage values, scale to frame size
            rect.x *= normalizedWidth;
            rect.y *= normalizedHeight;
            rect.w *= normalizedWidth;
            rect.h *= normalizedHeight;
        }
    } else {
        // Optimization, request profile sized image
        bWidth = normalizedWidth;
        bHeight = normalizedHeight;
        parAlreadyNormalized = true;
    }
    const double scalex = mlt_profile_scale_width(profile, width);
    const double scaley = mlt_profile_scale_height(profile, height);
    // Adjust to preview scaling
    if (scalex != 1.0) {
        rect.x *= scalex;
        rect.w *= scalex;
        if (scalex < 1.) {
            bWidth *= scalex;
        }
    }
    if (scaley != 1.0) {
        rect.y *= scaley;
        rect.h *= scaley;
        if (scaley < 1.) {
            bHeight *= scaley;
        }
    }
    auto job = std::make_unique<CompositeJob>();
    job->transform.translate(rect.x, rect.y);
    const double opacity = rect.o;
    if (!forceAlpha && (opacity < 1 || rect.x != 0 || rect.y != 0 || (rect.x + rect.w != width) || (rect.y + rect.h != height))) {
        // we will process operations on top frame, so also process b_frame
        forceAlpha = true;
    }
    // MLT 7.42 avoids normalizing the pixel aspect ratio twice (parFlag).
    // The 7.41 development versions may or may not do it: when that makes a
    // difference, the CPU draws.
    auto finishSize = [&](bool parFlag, int &w, int &h) {
        bool normalized = parAlreadyNormalized;
        if (parFlag && (w <= 0 || h <= 0)) {
            w = normalizedWidth;
            h = normalizedHeight;
            normalized = true;
        }
        w = std::max(1, w);
        h = std::max(1, h);
        if (v740 && (!parFlag || !normalized)) {
            // normalize_mlt_source_size()
            if (bAr > 0.0 && consumerAr > 0.0 && bAr != consumerAr && h > 0) {
                w = qRound(w * bAr / consumerAr);
            }
            w = std::max(1, w);
        }
    };
    int flagWidth = bWidth;
    int flagHeight = bHeight;
    finishSize(true, flagWidth, flagHeight);
    int noFlagWidth = bWidth;
    int noFlagHeight = bHeight;
    finishSize(false, noFlagWidth, noFlagHeight);
    if (v741 && !v742 && (flagWidth != noFlagWidth || flagHeight != noFlagHeight)) {
        return TransitionPlan::cpu();
    }
    bWidth = v742 ? flagWidth : noFlagWidth;
    bHeight = v742 ? flagHeight : noFlagHeight;
    if (v740 && rect.w > 0 && rect.h > 0 && bWidth > 0 && bHeight > 0) {
        // adjust_mlt_mipmap_size()
        double scaleTarget;
        if (distort) {
            scaleTarget = std::max(rect.w / bWidth, rect.h / bHeight);
        } else {
            const double resizeDar = rect.w * consumerAr / rect.h;
            if (bDar >= resizeDar) {
                scaleTarget = rect.w / bWidth;
                if (bDar < geometryDar) {
                    scaleTarget *= transformScale;
                }
            } else {
                scaleTarget = rect.h / bHeight;
                if (bDar > geometryDar) {
                    scaleTarget *= transformScale;
                }
            }
        }
        double mipmapScale = 1.0;
        while (mipmapScale * kMipmapStep >= scaleTarget && mipmapScale > 0.001) {
            mipmapScale *= kMipmapStep;
        }
        if (mipmapScale < 1.0) {
            bWidth = std::max(1, qRound(bWidth * mipmapScale));
            bHeight = std::max(1, qRound(bHeight * mipmapScale));
        }
    }
    if (mlt_properties_get(properties, "rotation")) {
        const double angle = mlt_properties_anim_get_double(properties, "rotation", ctx.position, ctx.length);
        if (angle != 0.0) {
            if (mlt_properties_get_int(properties, "rotate_center")) {
                job->transform.translate(rect.w / 2.0, rect.h / 2.0);
                job->transform.rotate(angle);
                job->transform.translate(-rect.w / 2.0, -rect.h / 2.0);
            } else {
                job->transform.rotate(angle);
            }
            forceAlpha = true;
        }
    }
    // Check profile dar vs image dar image
    if (!forceAlpha && rect.w == -1 && bDar != mlt_profile_dar(profile)) {
        forceAlpha = true;
    }
    if (!forceAlpha && (mode != 0 || bWidth < width || bHeight < height)) {
        forceAlpha = true;
    }
    if (!std::isfinite(rect.x) || !std::isfinite(rect.y) || !std::isfinite(rect.w) || !std::isfinite(rect.h)) {
        return TransitionPlan::cpu();
    }

    // The CPU transition first fetches the top image in the requested format
    // when it may not need to draw anything
    job->passOpaqueTop = !forceAlpha;
    job->topAlpha = hasAlpha;
    job->topFormat = forceAlpha ? mlt_image_rgba : format;
    job->topWidth = bWidth;
    job->topHeight = bHeight;
    job->bottomWritable = v741 ? 1 : 0;
    job->version = version;
    job->rect = rect;
    job->consumerAr = consumerAr;
    job->sourceDar = bDar;
    job->geometryDar = geometryDar;
    job->transformScale = transformScale;
    job->distort = distort;
    job->opacity = opacity;
    job->setAspectRatio = setAspectRatio;
    job->requestWidth = width;
    job->requestHeight = height;
    // We don't do subpixel smoothing for nearest neighbour interpolation
    const char *interpolation = mlt_properties_get(MLT_FRAME_PROPERTIES(ctx.aFrame), "consumer.rescale");
    job->smooth = interpolation && strcmp(interpolation, "nearest") && strcmp(interpolation, "neighbor");
    return TransitionPlan::gpu(std::move(job));
}

} // namespace

const EffectDescriptor &transformEffect()
{
    static const EffectDescriptor descriptor{"qtblend", false, planTransform};
    return descriptor;
}

const TransitionDescriptor &compositeTransition()
{
    static const TransitionDescriptor descriptor{"qtblend", planComposite};
    return descriptor;
}

} // namespace gpufx
