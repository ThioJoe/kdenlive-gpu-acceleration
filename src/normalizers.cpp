/*
    SPDX-FileCopyrightText: Meltytech, LLC
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: LGPL-2.1-or-later
*/

/*
 * The normalizing filters of MLT that the bridge lets an image through on the
 * GPU (see normalizer.h). What they do to the frame and its image is that of
 * MLT's code (filter_color_transform.c, filter_fieldorder.c, filter_crop.c,
 * filter_rescale.c, filter_resize.c of src/modules/core), which is the same
 * from MLT 7.40 to 7.43 for everything used here.
 */

#include "convert.h"
#include "normalizer.h"

#include <cmath>
#include <cstring>

namespace gpufx {

namespace {

/* ==========================================================================
 * color_transform: converts the transfer characteristic when the consumer
 * asks for one (consumer.mlt_color_trc, e.g. for HDR), otherwise returns
 * the image below without any change.
 * ========================================================================== */

bool colorTransformBefore(NormalizerCall &call)
{
    auto filter = static_cast<mlt_filter>(call.entries[0]);
    if (mlt_properties_get(MLT_FILTER_PROPERTIES(filter), "force_trc") || mlt_properties_get(call.properties, "consumer.mlt_color_trc")) {
        return false;
    }
    call.fetchWidth = call.width;
    call.fetchHeight = call.height;
    return true;
}

bool passFetched(NormalizerCall &call, int fetchedWidth, int fetchedHeight)
{
    call.outWidth = fetchedWidth;
    call.outHeight = fetchedHeight;
    return true;
}

/* ==========================================================================
 * fieldorder: swaps or shifts the fields of interlaced frames to the field
 * order of the consumer, and sets the frame's field order.
 * ========================================================================== */

bool fieldOrderBefore(NormalizerCall &call)
{
    call.fetchWidth = call.width;
    call.fetchHeight = call.height;
    return true;
}

bool fieldOrderAfter(NormalizerCall &call, int fetchedWidth, int fetchedHeight)
{
    mlt_properties properties = call.properties;
    const int tff = mlt_properties_get_int(properties, "consumer.top_field_first");
    int topFieldFirst = mlt_properties_get_int(properties, "top_field_first");
    if (mlt_properties_get(properties, "meta.top_field_first")) {
        topFieldFirst = mlt_properties_get_int(properties, "meta.top_field_first");
    }
    const bool interlaced = mlt_properties_get(properties, "progressive") && mlt_properties_get_int(properties, "progressive") == 0;
    if (interlaced && (mlt_properties_get_int(properties, "meta.swap_fields") || (tff != -1 && topFieldFirst != tff))) {
        return false;
    }
    if (mlt_properties_get(properties, "meta.top_field_first")) {
        mlt_properties_set_int(properties, "top_field_first", mlt_properties_get_int(properties, "meta.top_field_first"));
    }
    mlt_properties_set_int(properties, "top_field_first", tff);
    mlt_properties_set_int(properties, "meta.top_field_first", tff);
    call.outWidth = fetchedWidth;
    call.outHeight = fetchedHeight;
    return true;
}

/** get_image() of filter_fieldorder.c after the fetch, for RGBA */
int fieldOrderWork(NormalizerCall &call, uint8_t **image, int *width, int *height)
{
    mlt_properties properties = call.properties;
    const int tff = mlt_properties_get_int(properties, "consumer.top_field_first");
    if (mlt_properties_get(properties, "meta.top_field_first")) {
        mlt_properties_set_int(properties, "top_field_first", mlt_properties_get_int(properties, "meta.top_field_first"));
    }
    const int stride = *width * 4;
    // mlt_image_format_size() of RGBA
    const int size = *width * *height * 4;
    if (mlt_properties_get_int(properties, "meta.swap_fields") && mlt_properties_get(properties, "progressive") &&
        mlt_properties_get_int(properties, "progressive") == 0) {
        auto *newImage = static_cast<uint8_t *>(mlt_pool_alloc(size));
        uint8_t *src = *image;
        uint8_t *dst = newImage;
        mlt_frame_set_image(call.frame, newImage, size, mlt_pool_release);
        *image = newImage;
        for (int i = *height; i > 0; --i) {
            memcpy(dst, src + stride * !(i % 2), size_t(stride));
            dst += stride;
            src += stride * (i % 2) * 2;
        }
    }
    if (tff != -1 && mlt_properties_get_int(properties, "top_field_first") != tff && mlt_properties_get(properties, "progressive") &&
        mlt_properties_get_int(properties, "progressive") == 0) {
        // Shift the image down by one line
        auto *newImage = static_cast<uint8_t *>(mlt_pool_alloc(size));
        memcpy(newImage, *image, size_t(stride));
        memcpy(newImage + stride, *image, size_t(stride) * size_t(*height - 1));
        mlt_frame_set_image(call.frame, newImage, size, mlt_pool_release);
        *image = newImage;
    }
    mlt_properties_set_int(properties, "top_field_first", tff);
    mlt_properties_set_int(properties, "meta.top_field_first", tff);
    return 0;
}

/* ==========================================================================
 * crop (active): crops by the frame's crop.* properties, which the crop
 * effect sets. Without them it returns the image below as it is.
 * ========================================================================== */

bool cropBefore(NormalizerCall &call)
{
    auto profile = static_cast<mlt_profile>(call.entries[0]);
    mlt_properties properties = call.properties;
    if (mlt_properties_get_int(properties, "crop.left") || mlt_properties_get_int(properties, "crop.right") || mlt_properties_get_int(properties, "crop.top") ||
        mlt_properties_get_int(properties, "crop.bottom") || !profile) {
        return false;
    }
    call.fetchWidth = call.width;
    call.fetchHeight = call.height;
    if (call.fetchWidth == 0 || call.fetchHeight == 0) {
        call.fetchWidth = profile->width;
        call.fetchHeight = profile->height;
    }
    return true;
}

/* ==========================================================================
 * rescale with swscale's scaler: asks the image below for its own size and
 * scales it to the size asked for.
 * ========================================================================== */

using ImageScaler = int (*)(mlt_frame frame, uint8_t **image, mlt_image_format *format, int iwidth, int iheight, int owidth, int oheight);

bool rescaleBefore(NormalizerCall &call)
{
    auto filter = static_cast<mlt_filter>(call.entries[0]);
    mlt_properties filterProperties = MLT_FILTER_PROPERTIES(filter);
    mlt_properties properties = call.properties;
    // The rescale filter's own scaler works on YUV 4:2:2 only
    const char *service = mlt_properties_get(filterProperties, "mlt_service");
    if (!service || strcmp(service, "swscale") || !mlt_properties_get_data(filterProperties, "method", nullptr)) {
        return false;
    }
    int width = call.width;
    int height = call.height;
    if (width == 0 || height == 0) {
        mlt_profile profile = mlt_service_profile(MLT_FILTER_SERVICE(filter));
        width = profile->width;
        height = profile->height;
    }
    if (width < 6 || height < 6) {
        return false;
    }
    const char *interps = mlt_properties_get(properties, "consumer.rescale");
    if (!interps && !mlt_properties_get(filterProperties, "interpolation")) {
        return false;
    }
    int iwidth = width;
    int iheight = height;
    int owidth = width;
    int oheight = height;
    if (mlt_properties_get(filterProperties, "factor")) {
        const double factor = mlt_properties_get_double(filterProperties, "factor");
        owidth *= factor;
        oheight *= factor;
    }
    if (!interps) {
        interps = mlt_properties_get(filterProperties, "interpolation");
        mlt_properties_set(properties, "consumer.rescale", interps);
    }
    if (mlt_properties_get_int(properties, "meta.media.width")) {
        iwidth = mlt_properties_get_int(properties, "meta.media.width");
        iheight = mlt_properties_get_int(properties, "meta.media.height");
    }
    if (strcmp(interps, "none")) {
        mlt_properties_set_int(properties, "rescale_width", width);
        mlt_properties_set_int(properties, "rescale_height", height);
    } else {
        mlt_properties_set_int(properties, "rescale_width", iwidth);
        mlt_properties_set_int(properties, "rescale_height", iheight);
    }
    if (iheight != oheight && (strcmp(interps, "nearest") || (iheight % oheight != 0))) {
        mlt_properties_set_int(properties, "consumer.progressive", 1);
    }
    call.fetchWidth = iwidth;
    call.fetchHeight = iheight;
    call.targetWidth = owidth;
    call.targetHeight = oheight;
    return true;
}

bool rescaleAfter(NormalizerCall &call, int fetchedWidth, int fetchedHeight)
{
    const char *interps = mlt_properties_get(call.properties, "consumer.rescale");
    if (interps && strcmp(interps, "none") && (fetchedWidth != call.targetWidth || fetchedHeight != call.targetHeight)) {
        return false;
    }
    call.outWidth = fetchedWidth;
    call.outHeight = fetchedHeight;
    return true;
}

/** scale_alpha() of filter_rescale.c */
void scaleAlpha(mlt_frame frame, int iwidth, int iheight, int owidth, int oheight)
{
    uint8_t *input = mlt_frame_get_alpha(frame);
    if (!input) {
        return;
    }
    const int ox = (iwidth << 16) / owidth;
    const int oy = (iheight << 16) / oheight;
    auto *output = static_cast<uint8_t *>(mlt_pool_alloc(owidth * oheight));
    uint8_t *outLine = output;
    int y = oy >> 1;
    for (int i = 0; i < oheight; ++i, y += oy) {
        const uint8_t *inLine = &input[(y >> 16) * iwidth];
        int x = ox >> 1;
        for (int j = 0; j < owidth; ++j, x += ox) {
            *outLine++ = inLine[x >> 16];
        }
    }
    mlt_frame_set_alpha(frame, output, owidth * oheight, mlt_pool_release);
}

/** filter_scale() of filter_swscale.c for RGBA, on the GPU: the scaling
 *  methods Kdenlive offers, for frames without a separate alpha plane */
bool rescaleGpuWork(NormalizerCall &call, GpuScope &scope, CUdeviceptr in, int inWidth, int inHeight, DeviceBuffer &out, CUresult &result)
{
    mlt_properties properties = call.properties;
    int alphaSize = 0;
    if (mlt_frame_get_alpha_size(call.frame, &alphaSize) && alphaSize > 0) {
        return false;
    }
    // The interpolation as filter_scale() maps it to libswscale's flags
    const char *interps = mlt_properties_get(properties, "consumer.rescale");
    sws::ScaleMethod luma = sws::ScaleBilinear;
    sws::ScaleMethod chroma = sws::ScaleBilinear;
    if (strcmp(interps, "nearest") == 0 || strcmp(interps, "neighbor") == 0) {
        luma = chroma = sws::ScalePoint;
    } else if (strcmp(interps, "tiles") == 0 || strcmp(interps, "fast_bilinear") == 0) {
        return false;
    } else if (strcmp(interps, "bilinear") == 0) {
        luma = chroma = sws::ScaleBilinear;
    } else if (strcmp(interps, "bicubic") == 0) {
        luma = chroma = sws::ScaleBicubic;
    } else if (strcmp(interps, "bicublin") == 0) {
        luma = sws::ScaleBicubic;
        chroma = sws::ScaleBilinear;
    } else if (strcmp(interps, "gauss") == 0 || strcmp(interps, "sinc") == 0 || strcmp(interps, "spline") == 0) {
        return false;
    } else if (strcmp(interps, "hyper") == 0 || strcmp(interps, "lanczos") == 0) {
        luma = chroma = sws::ScaleLanczos;
    }
    // The colorspaces given to mlt_set_luma_transfer()
    mlt_profile profile = mlt_service_profile(MLT_PRODUCER_SERVICE(mlt_frame_get_original_producer(call.frame)));
    const int dstColorspace = profile ? profile->colorspace : mlt_colorspace_bt601;
    const int srcColorspace = mlt_image_colorspace_id(mlt_properties_get(properties, "colorspace"));
    result = scaleRgba(scope, in, inWidth, inHeight, call.targetWidth, call.targetHeight, luma, chroma, srcColorspace, dstColorspace, out);
    if (result == CUDA_ERROR_NOT_SUPPORTED) {
        result = CUDA_SUCCESS;
        return false;
    }
    mlt_properties_set_int(properties, "full_range", mlt_image_full_range(mlt_properties_get(properties, "consumer.color_range")));
    call.outWidth = call.targetWidth;
    call.outHeight = call.targetHeight;
    return true;
}

/** filter_get_image() of filter_rescale.c after the fetch, with the scaler
 *  of the filter (swscale's) */
int rescaleWork(NormalizerCall &call, uint8_t **image, int *width, int *height)
{
    auto filter = static_cast<mlt_filter>(call.entries[0]);
    auto scaler = reinterpret_cast<ImageScaler>(mlt_properties_get_data(MLT_FILTER_PROPERTIES(filter), "method", nullptr));
    const int iwidth = *width;
    const int iheight = *height;
    const int owidth = call.targetWidth;
    const int oheight = call.targetHeight;
    mlt_image_format format = mlt_image_rgba;
    scaler(call.frame, image, &format, iwidth, iheight, owidth, oheight);
    *width = owidth;
    *height = oheight;
    int alphaSize = 0;
    mlt_frame_get_alpha_size(call.frame, &alphaSize);
    if (alphaSize > 0 && alphaSize != owidth * oheight && alphaSize != owidth * (oheight + 1)) {
        scaleAlpha(call.frame, iwidth, iheight, owidth, oheight);
    }
    return 0;
}

/* ==========================================================================
 * resize: asks the image below for the size that keeps its aspect ratio
 * (unless the frame is to be distorted) and pads it to the size asked for.
 * ========================================================================== */

bool resizeBefore(NormalizerCall &call)
{
    static_assert(sizeof(void *) == sizeof(double), "the aspect ratio is read from a stack entry");
    auto filter = static_cast<mlt_filter>(call.entries[0]);
    double aspectRatio = 0.0;
    memcpy(&aspectRatio, &call.entries[1], sizeof(double));
    mlt_properties properties = call.properties;
    mlt_profile profile = mlt_service_profile(MLT_FILTER_SERVICE(filter));
    if (!profile) {
        return false;
    }
    const double consumerAspect = mlt_profile_sar(profile);
    int width = call.width;
    int height = call.height;
    if (width == 0 || height == 0) {
        width = profile->width;
        height = profile->height;
    }
    int owidth = width;
    int oheight = height;
    if (aspectRatio == 0.0) {
        aspectRatio = consumerAspect;
    }
    mlt_properties_set_double(properties, "aspect_ratio", aspectRatio);
    const char *rescale = mlt_properties_get(properties, "consumer.rescale");
    if (rescale && !strcmp(rescale, "none")) {
        call.fetchWidth = width;
        call.fetchHeight = height;
        call.passAll = true;
        return true;
    }
    if (mlt_properties_get_int(properties, "distort") == 0) {
        const int normalizedWidth = profile->width;
        const int normalizedHeight = profile->height;
        int realWidth = mlt_properties_get_int(properties, "meta.media.width");
        int realHeight = mlt_properties_get_int(properties, "meta.media.height");
        if (realWidth == 0) {
            realWidth = mlt_properties_get_int(properties, "width");
        }
        if (realHeight == 0) {
            realHeight = mlt_properties_get_int(properties, "height");
        }
        const double inputAr = aspectRatio * realWidth / realHeight;
        const double outputAr = consumerAspect * owidth / oheight;
        int scaledWidth = rint((inputAr * normalizedWidth) / outputAr);
        int scaledHeight = normalizedHeight;
        if (scaledWidth > normalizedWidth) {
            scaledWidth = normalizedWidth;
            scaledHeight = rint((outputAr * normalizedHeight) / inputAr);
        }
        owidth = rint(scaledWidth * owidth / normalizedWidth);
        oheight = rint(scaledHeight * oheight / normalizedHeight);
        mlt_frame_set_aspect_ratio(call.frame, consumerAspect);
    }
    mlt_properties_set_int(properties, "distort", 0);
    mlt_properties_set_int(properties, "resize_width", width);
    mlt_properties_set_int(properties, "resize_height", height);
    call.fetchWidth = owidth;
    call.fetchHeight = oheight;
    call.targetWidth = width;
    call.targetHeight = height;
    return true;
}

bool resizeAfter(NormalizerCall &call, int fetchedWidth, int fetchedHeight)
{
    if (call.passAll) {
        call.outWidth = fetchedWidth;
        call.outHeight = fetchedHeight;
        return true;
    }
    // The image is padded when smaller, and returned as the size asked for
    // in any case
    if (fetchedWidth != call.targetWidth || fetchedHeight != call.targetHeight) {
        return false;
    }
    call.outWidth = fetchedWidth;
    call.outHeight = fetchedHeight;
    return true;
}

/** Centered in the image of the size asked for, on transparent black (or the
 *  frame's resize_alpha), as resize_image() of filter_resize.c does. */
bool resizeGpuWork(NormalizerCall &call, GpuScope &scope, CUdeviceptr in, int inWidth, int inHeight, DeviceBuffer &out, CUresult &result)
{
    const int owidth = call.targetWidth;
    const int oheight = call.targetHeight;
    if (inWidth > owidth || inHeight > oheight || owidth <= 6 || oheight <= 6 || inWidth <= 6 || inHeight <= 6) {
        return false;
    }
    const unsigned alpha = uint8_t(mlt_properties_get_int(call.properties, "resize_alpha"));
    const unsigned fill = alpha << 24;
    const int offsetX = (owidth - inWidth) / 2;
    const int offsetY = (oheight - inHeight) / 2;
    result = out.allocate(size_t(owidth) * size_t(oheight) * 4);
    if (result == CUDA_SUCCESS) {
        result = scope.launch2D(KernelId::gpufx_pad_rgba, owidth, oheight, 32, 8, in, inWidth, inHeight, out.ptr(), owidth, oheight, offsetX, offsetY, fill);
    }
    call.outWidth = owidth;
    call.outHeight = oheight;
    return true;
}

/** frame_resize_image() of filter_resize.c, for RGBA. With an image larger
 *  than asked for in one direction and smaller in the other, MLT copies rows
 *  out of the image's bounds: only the part that fits is copied here. */
int resizeWork(NormalizerCall &call, uint8_t **image, int *width, int *height)
{
    const int owidth = call.targetWidth;
    const int oheight = call.targetHeight;
    const int iwidth = *width;
    const int iheight = *height;
    if (iwidth < owidth || iheight < oheight) {
        const uint8_t alphaValue = uint8_t(mlt_properties_get_int(call.properties, "resize_alpha"));
        const int bpp = 4;
        const int size = owidth * (oheight + 1) * bpp;
        auto *output = static_cast<uint8_t *>(mlt_pool_alloc(size));
        if (output && owidth > 6 && oheight > 6 && iwidth > 6 && iheight > 6) {
            uint8_t *p = output;
            memset(p, 0, size_t(owidth) * size_t(oheight) * bpp);
            if (alphaValue) {
                for (int n = owidth * oheight; n--; p += 4) {
                    p[3] = alphaValue;
                }
            }
            const int offsetX = (owidth - iwidth) / 2;
            const int offsetY = (oheight - iheight) / 2;
            for (int y = 0; y < iheight; ++y) {
                const int oy = offsetY + y;
                if (oy < 0 || oy >= oheight) {
                    continue;
                }
                for (int x = 0; x < iwidth; ++x) {
                    const int ox = offsetX + x;
                    if (ox >= 0 && ox < owidth) {
                        memcpy(output + (size_t(oy) * owidth + ox) * bpp, *image + (size_t(y) * iwidth + x) * bpp, bpp);
                    }
                }
            }
        }
        mlt_frame_set_image(call.frame, output, size, mlt_pool_release);
        *image = output;
    }
    *width = owidth;
    *height = oheight;
    return 0;
}

const NormalizerDescriptor kNormalizers[] = {
    {"color_transform", nullptr, 1, colorTransformBefore, passFetched, nullptr, nullptr},
    {"fieldorder", nullptr, 0, fieldOrderBefore, fieldOrderAfter, nullptr, fieldOrderWork},
    {"crop", nullptr, 1, cropBefore, passFetched, nullptr, nullptr},
    {"rescale", "swscale", 1, rescaleBefore, rescaleAfter, rescaleGpuWork, rescaleWork},
    {"resize", nullptr, 2, resizeBefore, resizeAfter, resizeGpuWork, resizeWork},
};

} // namespace

const NormalizerDescriptor *findNormalizer(const char *service)
{
    if (!service) {
        return nullptr;
    }
    for (const NormalizerDescriptor &descriptor : kNormalizers) {
        if (!strcmp(descriptor.service, service) || (descriptor.alias && !strcmp(descriptor.alias, service))) {
            return &descriptor;
        }
    }
    return nullptr;
}

} // namespace gpufx
