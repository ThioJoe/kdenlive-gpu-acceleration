/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

/*
 * The bridge between MLT's image stack and the GPU.
 *
 * Supported filters keep their own process() function: it pushes the
 * filter and its CPU get_image onto the frame. wrappedProcess() then slides
 * bridgeGetImage() on top of them, so that for every frame the bridge can
 * decide to call the CPU get_image unchanged or to do the work on the GPU.
 *
 * Frames move between the host and the GPU per effect, so any mix of
 * accelerated and CPU-only effects works. When two accelerated effects follow
 * each other in the stack the image stays in device memory between them:
 * the upper one sets kWantDevice on the frame before fetching its input, and
 * the lower one then attaches its device result to the frame (kDeviceImage)
 * instead of downloading it. The host pointer returned along with it is only
 * a tag in that case, its content is stale. This also works across the
 * boundary of a tractor (a Kdenlive track), whose frame hands the image of
 * the track's frame up unchanged.
 */

#include "bridge.h"
#include "convert.h"
#include "gpufx_log.h"
#include "mlt_compat.h"

#include <cstdio>
#include <cstring>
#include <memory>

namespace gpufx {

namespace {

const char *const kWrapProperty = "_gpufx.wrap";
const char *const kWantDevice = "_gpufx.want_device";
// Set with a request for YUV 4:2:2 by a caller that converts the image
// straight back to RGBA (see deliverYuv422Later())
const char *const kYuvRoundTrip = "_gpufx.yuv_round_trip";
const char *const kDeviceImage = "_gpufx.device_image";
const char *const kForceCpu = "_gpufx.disable";

struct WrapInfo
{
    mlt_frame (*originalProcess)(mlt_filter, mlt_frame);
    const EffectDescriptor *effect;
};

void destroyWrapInfo(void *data)
{
    delete static_cast<WrapInfo *>(data);
}

struct DeviceImage
{
    DeviceBuffer buffer;
    int width = 0;
    int height = 0;
    const uint8_t *tag = nullptr;
    /** The RGBA image stands for its conversion to YUV 4:2:2 in this
     *  colorspace, not done yet: the tag is a YUV 4:2:2 image */
    bool pendingYuv422 = false;
    int colorspace = 0;
};

struct DeviceImageBox
{
    std::unique_ptr<DeviceImage> image;
};

void destroyDeviceImageBox(void *data)
{
    delete static_cast<DeviceImageBox *>(data);
}

int bridgeGetImage(mlt_frame frame, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable);
int bridgeTransitionGetImage(mlt_frame frame, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable);
int decodeBridgeGetImage(mlt_frame frame, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable);
int normalizerGetImage(mlt_frame frame, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable);

bool isBridge(void *getImage)
{
    return getImage == reinterpret_cast<void *>(&bridgeGetImage) || getImage == reinterpret_cast<void *>(&bridgeTransitionGetImage) ||
           getImage == reinterpret_cast<void *>(&decodeBridgeGetImage);
}

/** The get_image a tractor puts on its frame to fetch the image of a track's
 *  frame, the entry below it: [track frame, get_image]. It hands that image up
 *  unchanged, in the same buffer. It is not exported, so it is found once in
 *  the frame of a probe tractor. */
void *tractorGetImage()
{
    static void *const getImage = []() -> void * {
        void *result = nullptr;
        mlt_profile profile = mlt_profile_init(nullptr);
        mlt_producer producer = profile ? mlt_factory_producer(profile, "color", "black") : nullptr;
        mlt_tractor tractor = producer ? mlt_tractor_new() : nullptr;
        mlt_frame frame = nullptr;
        if (tractor && !mlt_tractor_set_track(tractor, producer, 0) && !mlt_service_get_frame(MLT_TRACTOR_SERVICE(tractor), &frame, 0) && frame) {
            mlt_deque stack = MLT_FRAME_IMAGE_STACK(frame);
            void *track = mlt_deque_count(stack) == 2 ? mlt_deque_peek(stack, 0) : nullptr;
            // The tractor keeps the track's frame in a property of its own frame
            mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
            for (int i = 0; track && i < mlt_properties_count(properties); ++i) {
                if (mlt_properties_get_data_at(properties, i, nullptr) == track) {
                    result = mlt_deque_peek(stack, 1);
                    break;
                }
            }
        }
        mlt_frame_close(frame);
        mlt_tractor_close(tractor);
        mlt_producer_close(producer);
        mlt_profile_close(profile);
        if (!result) {
            logInfo("images cannot stay on the GPU across tracks with this version of MLT");
        }
        return result;
    }();
    return getImage;
}

/** The frame on which to request a GPU image, for the image that the
 *  get_image at depth (0 = top) of frame's stack delivers: frame itself if
 *  that get_image is a bridge, or normalizing filters that let the image of
 *  a bridge through, or the track frame below the get_image of one or more
 *  tractors if its next get_image is one. Null if the image comes from the
 *  CPU. */
mlt_frame gpuSource(mlt_frame frame, int depth)
{
    bool throughNormalizer = false;
    for (int level = 0; level < 16; ++level) {
        mlt_deque stack = MLT_FRAME_IMAGE_STACK(frame);
        const int count = mlt_deque_count(stack);
        void *getImage = count > depth ? mlt_deque_peek(stack, count - 1 - depth) : nullptr;
        if (!getImage) {
            return nullptr;
        }
        if (isBridge(getImage)) {
            return frame;
        }
        if (getImage == reinterpret_cast<void *>(&normalizerGetImage) && count > depth + 2) {
            // [CPU entries, CPU get_image, entry count, wrap info, normalizerGetImage]:
            // the image comes from the next get_image below
            depth += 4 + int(intptr_t(mlt_deque_peek(stack, count - 3 - depth)));
            throughNormalizer = true;
            continue;
        }
        // The request goes to the frame of the top normalizer, which passes it on
        if (throughNormalizer || count < depth + 2 || getImage != tractorGetImage()) {
            return nullptr;
        }
        frame = static_cast<mlt_frame>(mlt_deque_peek(stack, count - 2 - depth));
        depth = 0;
    }
    return nullptr;
}

/** The same for the image of a transition's frame, where depth is the first
 *  entry below the transition's own. MLT's tractor puts a hook there that
 *  only sets some frame properties: [transition, get_image_a] on the bottom
 *  frame, [bottom frame, transition, get_image_b] on the top frame. The image
 *  passes through it unchanged, so the bridge below it can be used. */
mlt_frame gpuSourceBelowTransition(mlt_frame frame, int depth, mlt_transition transition, mlt_frame hookFrame)
{
    mlt_deque stack = MLT_FRAME_IMAGE_STACK(frame);
    const int count = mlt_deque_count(stack);
    auto entry = [&](int d) { return count > d ? mlt_deque_peek(stack, count - 1 - d) : nullptr; };
    if (entry(depth + 1) == transition) {
        depth += 2;
        if (hookFrame && entry(depth) == hookFrame) {
            depth += 1;
        }
    }
    return gpuSource(frame, depth);
}

bool takeFlag(mlt_properties properties, const char *name)
{
    if (mlt_properties_get_int(properties, name)) {
        mlt_properties_set_int(properties, name, 0);
        return true;
    }
    return false;
}

/** mlt_destructor of the memory of allocateImage() */
void releaseImage(void *ptr)
{
    if (ptr && !Runtime::instance().recycleHost(ptr)) {
        mlt_pool_release(ptr);
    }
}

/** Memory for an image that the GPU writes and MLT then keeps: page-locked
 *  while the pool has room, so that the image is downloaded straight into it.
 *  Must be called with the context current, released by releaseImage(). */
uint8_t *allocateImage(size_t bytes)
{
    // Room past the end, as MLT's pool gives, for code that reads a little beyond
    void *pinned = Runtime::instance().allocateHost(bytes + 64);
    return static_cast<uint8_t *>(pinned ? pinned : mlt_pool_alloc(int(bytes)));
}

void attachDeviceImage(mlt_properties properties, std::unique_ptr<DeviceImage> image)
{
    auto *box = new DeviceImageBox{std::move(image)};
    mlt_properties_set_data(properties, kDeviceImage, box, 0, destroyDeviceImageBox, nullptr);
}

/** mlt_frame_get_image() of frame, asking for the image on the GPU from the
 *  bridge of source (see gpuSource()) when it is not null. A GPU image ends up
 *  on frame, for takeDeviceImage(). With yuvRoundTrip set, a request for YUV
 *  4:2:2 may come back as an RGBA image on the GPU whose conversion to YUV is
 *  pending (DeviceImage::pendingYuv422). */
int fetchImage(mlt_frame frame, mlt_frame source, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable, bool yuvRoundTrip = false)
{
    const char *flag = kWantDevice;
    if (*format != mlt_image_rgba) {
        // The host image standing for a GPU image is not filled, and MLT
        // would convert it to the format asked for (the qtblend transition
        // asks for the consumer's format when it may only pass the top image
        // on): the bridge delivers a real image in that format instead.
        if (yuvRoundTrip && *format == mlt_image_yuv422) {
            flag = kYuvRoundTrip;
        } else {
            source = nullptr;
        }
    }
    if (source) {
        mlt_properties_set_int(MLT_FRAME_PROPERTIES(source), flag, 1);
    }
    const int error = mlt_frame_get_image(frame, image, format, width, height, writable);
    if (source) {
        mlt_properties sourceProperties = MLT_FRAME_PROPERTIES(source);
        mlt_properties_set_int(sourceProperties, flag, 0);
        auto *box = source != frame ? static_cast<DeviceImageBox *>(mlt_properties_get_data(sourceProperties, kDeviceImage, nullptr)) : nullptr;
        if (box) {
            // The tractors passed the image up, the GPU image goes along
            std::unique_ptr<DeviceImage> device = std::move(box->image);
            mlt_properties_set_data(sourceProperties, kDeviceImage, nullptr, 0, nullptr, nullptr);
            if (device) {
                attachDeviceImage(MLT_FRAME_PROPERTIES(frame), std::move(device));
            }
        }
    }
    return error;
}

std::unique_ptr<DeviceImage> takeDeviceImage(mlt_properties properties, const uint8_t *tag, int width, int height, mlt_image_format format)
{
    auto *box = static_cast<DeviceImageBox *>(mlt_properties_get_data(properties, kDeviceImage, nullptr));
    if (!box) {
        return nullptr;
    }
    std::unique_ptr<DeviceImage> image = std::move(box->image);
    // Deletes the now empty box
    mlt_properties_set_data(properties, kDeviceImage, nullptr, 0, nullptr, nullptr);
    if (!image) {
        return nullptr;
    }
    if (image->tag != tag || image->width != width || image->height != height || format != (image->pendingYuv422 ? mlt_image_yuv422 : mlt_image_rgba)) {
        logWarning("ignoring a GPU image that does not match the frame image");
        return nullptr;
    }
    return image;
}

/** Runs the original CPU get_image after the input image has already been
 *  fetched, e.g. when a CUDA call failed. The image stack is empty at this
 *  point, so the CPU filter gets the frame's image back from its properties. */
int cpuFallback(mlt_frame frame, mlt_filter filter, mlt_get_image cpuGetImage, uint8_t *input, std::unique_ptr<uint8_t, void (*)(void *)> hostCopy,
                int inputWidth, int inputHeight, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    if (hostCopy) {
        // Pixels downloaded from the GPU, they become the frame image
        const int size = inputWidth * inputHeight * 4;
        mlt_frame_set_image(frame, hostCopy.release(), size, mlt_pool_release);
        mlt_properties_set_int(properties, "format", mlt_image_rgba);
        mlt_properties_set_int(properties, "width", inputWidth);
        mlt_properties_set_int(properties, "height", inputHeight);
    } else if (input && mlt_properties_get_data(properties, "image", nullptr) != input) {
        // Make sure the CPU filter finds this image on the frame
        const int size = inputWidth * inputHeight * 4;
        auto *copy = static_cast<uint8_t *>(mlt_pool_alloc(size));
        if (!copy) {
            return 1;
        }
        memcpy(copy, input, size_t(size));
        mlt_frame_set_image(frame, copy, size, mlt_pool_release);
        mlt_properties_set_int(properties, "format", mlt_image_rgba);
        mlt_properties_set_int(properties, "width", inputWidth);
        mlt_properties_set_int(properties, "height", inputHeight);
    }
    Runtime::instance().countFrame(false);
    mlt_frame_push_service(frame, filter);
    return cpuGetImage(frame, image, format, width, height, writable);
}

/** Downloads an input image that only exists on the GPU, so that the CPU
 *  filter can process it. Returns nullptr if that fails. */
std::unique_ptr<uint8_t, void (*)(void *)> rescueInput(GpuScope &scope, const DeviceBuffer &in, size_t bytes)
{
    std::unique_ptr<uint8_t, void (*)(void *)> copy(static_cast<uint8_t *>(mlt_pool_alloc(int(bytes))), mlt_pool_release);
    CUresult res = copy ? scope.download(copy.get(), in.ptr(), bytes) : CUDA_ERROR_OUT_OF_MEMORY;
    if (res == CUDA_SUCCESS) {
        res = scope.synchronize();
    }
    if (res != CUDA_SUCCESS) {
        copy.reset();
    }
    return copy;
}

/* ==========================================================================
 * Conversion of the result for the caller
 *
 * When the caller asks for packed 4:2:2 YUV, as MLT's avformat consumer and
 * Kdenlive's monitor do, mlt_frame_get_image() converts the RGBA result with
 * the frame's image converter. If that converter is avcolour_space, whose
 * arithmetic the GPU reproduces, the result is converted on the GPU instead
 * and only the smaller YUV image is downloaded.
 * ========================================================================== */

/** The converter mlt_frame_convert_image() tries first: MLT 7.42 keeps a list
 *  of them, older versions a single function. */
void *frameConverter(mlt_frame frame)
{
    auto list = static_cast<mlt_deque>(mlt_properties_get_data(MLT_FRAME_PROPERTIES(frame), "_convert_image_callbacks", nullptr));
    if (list) {
        return mlt_deque_count(list) > 0 ? mlt_deque_peek(list, 0) : nullptr;
    }
    return reinterpret_cast<void *>(frame->convert_image);
}

/** The convert_image function of MLT's avcolour_space filter, found by
 *  letting an instance process a frame. */
void *avcolourConverter()
{
    static void *const converter = []() -> void * {
        void *result = nullptr;
        mlt_profile profile = mlt_profile_init(nullptr);
        mlt_filter filter = profile ? mlt_factory_filter(profile, "avcolor_space", nullptr) : nullptr;
        mlt_frame frame = filter ? mlt_frame_init(nullptr) : nullptr;
        if (frame) {
            void *before = frameConverter(frame);
            mlt_filter_process(filter, frame);
            result = frameConverter(frame);
            if (result == before) {
                result = nullptr;
            }
            mlt_frame_close(frame);
        }
        mlt_filter_close(filter);
        mlt_profile_close(profile);
        if (!result) {
            logInfo("avcolour_space is not available, RGBA results are converted on the CPU");
        }
        return result;
    }();
    return converter;
}

/** True if the RGBA result can be delivered as the YUV 4:2:2 image the caller
 *  asked for, converted on the GPU. */
bool deliversYuv422(mlt_frame frame, mlt_image_format requested, int width, int height)
{
    if (requested != mlt_image_yuv422 || width <= 0 || height <= 0 || (width & 1) || !serviceEnabled("conversion", "avcolor_space")) {
        return false;
    }
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    // A conversion that also scales the image
    if (mlt_properties_get_int(properties, "convert_image_width") || mlt_properties_get_int(properties, "convert_image_height")) {
        return false;
    }
    // FFmpeg 8.0 changed the rounding of libswscale's conversion to full
    // range YUV, and which FFmpeg MLT uses is not known here: those stay on
    // the CPU (limited range is the default)
    if (mlt_image_full_range(mlt_properties_get(properties, "consumer.color_range"))) {
        return false;
    }
    void *converter = frameConverter(frame);
    return converter && converter == avcolourConverter();
}

/** Converts the RGBA result to YUV 4:2:2 on the GPU and makes it the frame
 *  image, with the frame properties avcolour_space would set. */
CUresult deliverYuv422(GpuScope &scope, CUdeviceptr rgba, mlt_frame frame, int width, int height, uint8_t **image, mlt_image_format *format, int *outWidth,
                       int *outHeight)
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    mlt_producer producer = mlt_frame_get_original_producer(frame);
    mlt_profile profile = producer ? mlt_service_profile(MLT_PRODUCER_SERVICE(producer)) : nullptr;
    const int colorspace = profile ? profile->colorspace : mlt_colorspace_bt601;
    const int fullRange = 0; // see deliversYuv422()
    const int size = width * height * 2;
    const int alphaSize = width * height;
    auto *yuv = allocateImage(size_t(size));
    auto *alpha = allocateImage(size_t(alphaSize));
    CUresult res = yuv && alpha ? convertRgbaToYuv422(scope, rgba, width, height, colorspace, fullRange, yuv, alpha) : CUDA_ERROR_OUT_OF_MEMORY;
    if (res != CUDA_SUCCESS) {
        releaseImage(yuv);
        releaseImage(alpha);
        return res;
    }
    mlt_frame_set_alpha(frame, alpha, alphaSize, releaseImage);
    mlt_frame_set_image(frame, yuv, size, releaseImage);
    mlt_properties_set_int(properties, "colorspace", colorspace);
    mlt_properties_set_int(properties, "full_range", fullRange);
    mlt_properties_set_int(properties, "format", mlt_image_yuv422);
    mlt_properties_set_int(properties, "width", width);
    mlt_properties_set_int(properties, "height", height);
    *image = yuv;
    *format = mlt_image_yuv422;
    *outWidth = width;
    *outHeight = height;
    return CUDA_SUCCESS;
}

/** deliverYuv422() for a caller that converts the YUV image straight back to
 *  RGBA (kYuvRoundTrip), as MLT's qtblend transition does with a top image
 *  that has an alpha plane: the RGBA result stays on the GPU and the caller
 *  does both conversions there. The host image is a YUV 4:2:2 image that only
 *  stands for the GPU one, with the frame properties of deliverYuv422(). */
CUresult deliverYuv422Later(GpuScope &scope, DeviceBuffer &rgba, mlt_frame frame, int width, int height, uint8_t **image, mlt_image_format *format,
                            int *outWidth, int *outHeight)
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    mlt_producer producer = mlt_frame_get_original_producer(frame);
    mlt_profile profile = producer ? mlt_service_profile(MLT_PRODUCER_SERVICE(producer)) : nullptr;
    const int colorspace = profile ? profile->colorspace : mlt_colorspace_bt601;
    const int size = width * height * 2;
    CUresult res = scope.synchronize();
    auto *tag = res == CUDA_SUCCESS ? static_cast<uint8_t *>(mlt_pool_alloc(size)) : nullptr;
    if (!tag) {
        return res == CUDA_SUCCESS ? CUDA_ERROR_OUT_OF_MEMORY : res;
    }
    mlt_frame_set_image(frame, tag, size, mlt_pool_release);
    auto handoff = std::make_unique<DeviceImage>();
    handoff->buffer = std::move(rgba);
    handoff->width = width;
    handoff->height = height;
    handoff->tag = tag;
    handoff->pendingYuv422 = true;
    handoff->colorspace = colorspace;
    attachDeviceImage(properties, std::move(handoff));
    mlt_properties_set_int(properties, "colorspace", colorspace);
    mlt_properties_set_int(properties, "full_range", 0);
    mlt_properties_set_int(properties, "format", mlt_image_yuv422);
    mlt_properties_set_int(properties, "width", width);
    mlt_properties_set_int(properties, "height", height);
    *image = tag;
    *format = mlt_image_yuv422;
    *outWidth = width;
    *outHeight = height;
    return CUDA_SUCCESS;
}

int runOnGpu(mlt_frame frame, mlt_filter filter, mlt_get_image cpuGetImage, const EffectDescriptor &effect, EffectPlan &plan, bool callerWantsDevice,
             bool yuvRoundTrip, mlt_frame lowerSource, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    Runtime &runtime = Runtime::instance();
    EffectJob &job = *plan.job;
    const mlt_image_format requestedFormat = *format;

    // Our filter entry, the stack now starts with the image source
    mlt_frame_pop_service(frame);

    uint8_t *input = nullptr;
    mlt_image_format inputFormat = mlt_image_rgba;
    int w = plan.inputWidth > 0 ? plan.inputWidth : *width;
    int h = plan.inputHeight > 0 ? plan.inputHeight : *height;
    job.beforeFetch(frame);
    const int error = fetchImage(frame, lowerSource, &input, &inputFormat, &w, &h, 0);
    std::unique_ptr<DeviceImage> deviceInput = takeDeviceImage(properties, input, w, h, inputFormat);
    if (error || !input) {
        *image = input;
        *format = inputFormat;
        *width = w;
        *height = h;
        return error ? error : 1;
    }
    const size_t bytes = size_t(w) * size_t(h) * 4;
    const bool inputOnDevice = deviceInput != nullptr;
    std::unique_ptr<uint8_t, void (*)(void *)> rescued(nullptr, mlt_pool_release);
    if (inputFormat != mlt_image_rgba || w <= 0 || h <= 0 || !job.prepare(w, h)) {
        if (inputOnDevice) {
            GpuScope scope;
            if (scope.ok()) {
                rescued = rescueInput(scope, deviceInput->buffer, bytes);
            }
            if (!rescued) {
                logError("%s: lost the GPU image", effect.service);
                return 1;
            }
        }
        job.restoreFrame(frame);
        return cpuFallback(frame, filter, cpuGetImage, input, std::move(rescued), w, h, image, format, width, height, writable);
    }

    const int outW = plan.outputWidth > 0 ? plan.outputWidth : w;
    const int outH = plan.outputHeight > 0 ? plan.outputHeight : h;
    const size_t outBytes = size_t(outW) * size_t(outH) * 4;
    CUresult res = CUDA_ERROR_NOT_INITIALIZED;
    {
        GpuScope scope;
        if (scope.ok()) {
            DeviceBuffer in;
            DeviceBuffer out;
            if (inputOnDevice) {
                in = std::move(deviceInput->buffer);
                res = CUDA_SUCCESS;
            } else {
                res = in.allocate(bytes);
                if (res == CUDA_SUCCESS) {
                    res = scope.upload(in.ptr(), input, bytes);
                }
            }
            if (res == CUDA_SUCCESS) {
                res = out.allocate(outBytes);
            }
            if (res == CUDA_SUCCESS) {
                res = job.runResized(scope, w, h, in.ptr(), outW, outH, out.ptr());
            }
            if (res == CUDA_SUCCESS) {
                if (callerWantsDevice) {
                    // The caller continues on the GPU: keep the result there
                    res = scope.synchronize();
                    uint8_t *tag = input;
                    if (res == CUDA_SUCCESS && outBytes != bytes) {
                        // The host image stands for the GPU one, give it the right size
                        tag = static_cast<uint8_t *>(mlt_pool_alloc(int(outBytes)));
                        if (tag) {
                            mlt_frame_set_image(frame, tag, int(outBytes), mlt_pool_release);
                        } else {
                            res = CUDA_ERROR_OUT_OF_MEMORY;
                        }
                    }
                    if (res == CUDA_SUCCESS) {
                        auto handoff = std::make_unique<DeviceImage>();
                        handoff->buffer = std::move(out);
                        handoff->width = outW;
                        handoff->height = outH;
                        handoff->tag = tag;
                        attachDeviceImage(properties, std::move(handoff));
                        *image = tag;
                        *format = mlt_image_rgba;
                        *width = outW;
                        *height = outH;
                        runtime.countFrame(true);
                        logDebug("%s: %dx%d on GPU, kept on GPU", effect.service, outW, outH);
                        return 0;
                    }
                } else if (deliversYuv422(frame, requestedFormat, outW, outH)) {
                    res = yuvRoundTrip ? deliverYuv422Later(scope, out, frame, outW, outH, image, format, width, height)
                                       : deliverYuv422(scope, out.ptr(), frame, outW, outH, image, format, width, height);
                    if (res == CUDA_SUCCESS) {
                        runtime.countFrame(true);
                        logDebug("%s: %dx%d on GPU, %s", effect.service, outW, outH,
                                 yuvRoundTrip ? "kept on GPU, its conversion to YUV and back to RGBA left to the caller" : "converted to YUV 4:2:2 on GPU");
                        return 0;
                    }
                } else {
                    auto *result = allocateImage(outBytes);
                    res = result ? scope.download(result, out.ptr(), outBytes) : CUDA_ERROR_OUT_OF_MEMORY;
                    if (res == CUDA_SUCCESS) {
                        res = scope.synchronize();
                    }
                    if (res == CUDA_SUCCESS) {
                        mlt_frame_set_image(frame, result, int(outBytes), releaseImage);
                        *image = result;
                        *format = mlt_image_rgba;
                        *width = outW;
                        *height = outH;
                        runtime.countFrame(true);
                        logDebug("%s: %dx%d on GPU", effect.service, outW, outH);
                        return 0;
                    }
                    if (result) {
                        releaseImage(result);
                    }
                }
            }
            runtime.handleError(res, effect.service);
            if (inputOnDevice && in.valid()) {
                // The host image is stale: get the pixels back for the CPU filter
                rescued = rescueInput(scope, in, bytes);
            }
        } else if (inputOnDevice) {
            // No usable context, the input only exists on the GPU
            res = CUDA_ERROR_NOT_INITIALIZED;
        }
    }
    if (inputOnDevice && !rescued) {
        logError("%s: lost the GPU image after a CUDA error", effect.service);
        return 1;
    }
    logDebug("%s: falling back to the CPU", effect.service);
    job.restoreFrame(frame);
    return cpuFallback(frame, filter, cpuGetImage, input, std::move(rescued), w, h, image, format, width, height, writable);
}

int bridgeGetImage(mlt_frame frame, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    auto cpuGetImage = reinterpret_cast<mlt_get_image>(mlt_frame_pop_service(frame));
    // Always consume the flags, they are only meant for the get_image right below their setter
    const bool callerWantsDevice = takeFlag(properties, kWantDevice);
    const bool yuvRoundTrip = takeFlag(properties, kYuvRoundTrip);
    auto filter = static_cast<mlt_filter>(mlt_deque_peek_back(MLT_FRAME_IMAGE_STACK(frame)));
    if (!cpuGetImage || !filter) {
        return 1;
    }
    mlt_properties filterProperties = MLT_FILTER_PROPERTIES(filter);
    auto *info = static_cast<WrapInfo *>(mlt_properties_get_data(filterProperties, kWrapProperty, nullptr));
    if (!info || !moduleEnabled() || !effectEnabled(*info->effect) || mlt_properties_get_int(filterProperties, kForceCpu)) {
        return cpuGetImage(frame, image, format, width, height, writable);
    }
    const EffectDescriptor &effect = *info->effect;

    EffectContext ctx;
    ctx.filter = filter;
    ctx.frame = frame;
    ctx.properties = filterProperties;
    ctx.format = *format;
    ctx.width = *width;
    ctx.height = *height;
    mlt_service_lock(MLT_FILTER_SERVICE(filter));
    ctx.position = mlt_filter_get_position(filter, frame);
    ctx.length = mlt_filter_get_length2(filter, frame);
    EffectPlan plan = effect.plan(ctx);
    mlt_service_unlock(MLT_FILTER_SERVICE(filter));

    mlt_frame lowerSource = gpuSource(frame, 1);
    const bool lowerOnGpu = lowerSource != nullptr;
    switch (plan.decision) {
    case Decision::PassThrough:
        if (callerWantsDevice && lowerOnGpu && *format == mlt_image_rgba) {
            // Nothing to do: let the image travel through on the GPU
            mlt_frame_pop_service(frame);
            return fetchImage(frame, lowerSource, image, format, width, height, writable);
        }
        return cpuGetImage(frame, image, format, width, height, writable);
    case Decision::Cpu:
        Runtime::instance().countFrame(false);
        return cpuGetImage(frame, image, format, width, height, writable);
    case Decision::Gpu:
        break;
    }
    if (!plan.job) {
        return cpuGetImage(frame, image, format, width, height, writable);
    }
    // A cheap effect on its own would spend more time copying the frame to
    // the GPU and back than the CPU needs to process it, unless the GPU also
    // converts the result to the format the caller asked for
    if (effect.lightweight && !callerWantsDevice && !(lowerOnGpu && *format == mlt_image_rgba) && !deliversYuv422(frame, *format, *width, *height)) {
        return cpuGetImage(frame, image, format, width, height, writable);
    }
    if (!Runtime::instance().ensureInitialized()) {
        return cpuGetImage(frame, image, format, width, height, writable);
    }
    return runOnGpu(frame, filter, cpuGetImage, effect, plan, callerWantsDevice, yuvRoundTrip, lowerSource, image, format, width, height, writable);
}

/* ==========================================================================
 * Transitions
 * ========================================================================== */

struct TransitionWrapInfo
{
    mlt_frame (*originalProcess)(mlt_transition, mlt_frame, mlt_frame);
    const TransitionDescriptor *descriptor;
};

void destroyTransitionWrapInfo(void *data)
{
    delete static_cast<TransitionWrapInfo *>(data);
}

bool isRgba(mlt_image_format format)
{
    return format == mlt_image_rgba;
}

/** An image fetched by the transition: in host memory, or on the GPU with a
 *  stale host buffer standing for it. */
struct FetchedImage
{
    uint8_t *host = nullptr;
    mlt_image_format format = mlt_image_rgba;
    int width = 0;
    int height = 0;
    std::unique_ptr<DeviceImage> device;
};

/** Brings an image that only exists on the GPU back to its frame, for the CPU. */
bool ensureHost(mlt_frame frame, FetchedImage &fetched)
{
    if (!fetched.device) {
        return true;
    }
    const size_t bytes = size_t(fetched.width) * size_t(fetched.height) * 4;
    std::unique_ptr<uint8_t, void (*)(void *)> copy(nullptr, mlt_pool_release);
    {
        GpuScope scope;
        if (scope.ok()) {
            copy = rescueInput(scope, fetched.device->buffer, bytes);
        }
    }
    fetched.device.reset();
    if (!copy) {
        return false;
    }
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    fetched.host = copy.release();
    mlt_frame_set_image(frame, fetched.host, int(bytes), mlt_pool_release);
    mlt_properties_set_int(properties, "format", mlt_image_rgba);
    mlt_properties_set_int(properties, "width", fetched.width);
    mlt_properties_set_int(properties, "height", fetched.height);
    return true;
}

/** Downloads an image of the GPU to its frame (the host buffer was stale). */
bool restoreToFrame(GpuScope &scope, const DeviceBuffer &buffer, mlt_frame frame, FetchedImage &fetched)
{
    const size_t bytes = size_t(fetched.width) * size_t(fetched.height) * 4;
    std::unique_ptr<uint8_t, void (*)(void *)> copy =
        buffer.valid() ? rescueInput(scope, buffer, bytes) : std::unique_ptr<uint8_t, void (*)(void *)>(nullptr, mlt_pool_release);
    if (!copy) {
        return false;
    }
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    fetched.host = copy.release();
    mlt_frame_set_image(frame, fetched.host, int(bytes), mlt_pool_release);
    mlt_properties_set_int(properties, "format", mlt_image_rgba);
    mlt_properties_set_int(properties, "width", fetched.width);
    mlt_properties_set_int(properties, "height", fetched.height);
    return true;
}

/** Hands the frame to the CPU transition after images were fetched: they are
 *  on their frames, where the CPU transition finds them again. */
int transitionCpuFallback(mlt_frame aFrame, mlt_transition transition, mlt_frame bFrame, mlt_get_image cpuGetImage, FetchedImage &top, FetchedImage &bottom,
                          uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    if (!ensureHost(bFrame, top) || !ensureHost(aFrame, bottom)) {
        logError("lost a GPU image after a CUDA error");
        return 1;
    }
    Runtime::instance().countFrame(false);
    mlt_frame_push_service(aFrame, transition);
    mlt_frame_push_frame(aFrame, bFrame);
    return cpuGetImage(aFrame, image, format, width, height, writable);
}

/** mlt_image_is_opaque() of an RGBA image on the GPU. Returns a CUDA error code. */
CUresult deviceImageOpaque(const DeviceImage &image, bool &opaque)
{
    GpuScope scope;
    if (!scope.ok()) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    DeviceBuffer flag;
    CUresult res = flag.allocate(sizeof(int));
    if (res == CUDA_SUCCESS) {
        res = scope.fill(flag.ptr(), 0, sizeof(int));
    }
    const int count = image.width * image.height;
    if (res == CUDA_SUCCESS) {
        res = scope.launch1D(KernelId::gpufx_alpha_check, count, image.buffer.ptr(), count, flag.ptr());
    }
    int transparent = 0;
    if (res == CUDA_SUCCESS) {
        res = scope.download(&transparent, flag.ptr(), sizeof(int));
    }
    if (res == CUDA_SUCCESS) {
        res = scope.synchronize();
    }
    opaque = transparent == 0;
    return res;
}

/** Does the conversion of the top image to YUV 4:2:2 that its bridge left
 *  pending (DeviceImage::pendingYuv422), and the conversion back to RGBA that
 *  MLT's transition does next. When avcolour_space would convert back from
 *  the image's own colorspace, both are done on the GPU and the RGBA image
 *  stays there. Otherwise only the YUV image is made, for MLT's conversion on
 *  the CPU. Returns a CUDA error code. */
CUresult convertTopBack(mlt_frame bFrame, FetchedImage &top)
{
    GpuScope scope;
    if (!scope.ok()) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    DeviceImage &device = *top.device;
    mlt_properties properties = MLT_FRAME_PROPERTIES(bFrame);
    if (deliversYuv422(bFrame, mlt_image_yuv422, top.width, top.height) && mlt_properties_get_int(properties, "colorspace") == device.colorspace &&
        !mlt_properties_get_int(properties, "full_range")) {
        DeviceBuffer rgba;
        CUresult res = convertRgbaThroughYuv422(scope, device.buffer.ptr(), top.width, top.height, device.colorspace, device.colorspace, false, rgba);
        if (res == CUDA_SUCCESS) {
            res = scope.synchronize();
        }
        if (res == CUDA_SUCCESS) {
            device.buffer = std::move(rgba);
            device.pendingYuv422 = false;
            top.format = mlt_image_rgba;
        }
        return res;
    }
    const int pixels = top.width * top.height;
    auto *yuv = static_cast<uint8_t *>(mlt_pool_alloc(pixels * 2));
    auto *alpha = static_cast<uint8_t *>(mlt_pool_alloc(pixels));
    CUresult res = yuv && alpha ? convertRgbaToYuv422(scope, device.buffer.ptr(), top.width, top.height, device.colorspace, false, yuv, alpha)
                                : CUDA_ERROR_OUT_OF_MEMORY;
    if (res != CUDA_SUCCESS) {
        mlt_pool_release(yuv);
        mlt_pool_release(alpha);
        return res;
    }
    mlt_frame_set_alpha(bFrame, alpha, pixels, mlt_pool_release);
    mlt_frame_set_image(bFrame, yuv, pixels * 2, mlt_pool_release);
    top.host = yuv;
    top.device.reset();
    return CUDA_SUCCESS;
}

int runTransitionOnGpu(mlt_frame aFrame, mlt_get_image cpuGetImage, const TransitionDescriptor &descriptor, TransitionPlan &plan, const TransitionContext &ctx,
                       mlt_frame topSource, mlt_frame bottomSource, bool callerWantsDevice, uint8_t **image, mlt_image_format *format, int *width,
                       int *height, int writable)
{
    TransitionJob &job = *plan.job;
    Runtime &runtime = Runtime::instance();
    const mlt_image_format requestedFormat = *format;
    mlt_frame bFrame = mlt_frame_pop_frame(aFrame);
    auto transition = static_cast<mlt_transition>(mlt_frame_pop_service(aFrame));
    mlt_properties aProperties = MLT_FRAME_PROPERTIES(aFrame);
    mlt_properties bProperties = MLT_FRAME_PROPERTIES(bFrame);
    job.beforeFetch(aFrame, bFrame);

    // Top image
    FetchedImage top;
    top.format = job.topFormat;
    top.width = job.topWidth;
    top.height = job.topHeight;
    int error = fetchImage(bFrame, topSource, &top.host, &top.format, &top.width, &top.height, 0, job.passOpaqueTop);
    top.device = takeDeviceImage(bProperties, top.host, top.width, top.height, top.format);
    FetchedImage bottom;
    if (error || !top.host) {
        return transitionCpuFallback(aFrame, transition, bFrame, cpuGetImage, top, bottom, image, format, width, height, writable);
    }

    if (job.passOpaqueTop) {
        // The top image is used as is if it has no transparency
        bool hasAlpha = job.topAlpha;
        // A pending conversion to YUV 4:2:2 also takes the alpha plane out
        const bool pendingYuv422 = top.device && top.device->pendingYuv422;
        if (!hasAlpha && (isRgba(top.format) || mlt_frame_get_alpha(bFrame) || pendingYuv422)) {
            hasAlpha = true;
        }
        if (hasAlpha && isRgba(top.format)) {
            bool opaque = false;
            if (top.device) {
                const CUresult res = deviceImageOpaque(*top.device, opaque);
                if (res != CUDA_SUCCESS) {
                    runtime.handleError(res, descriptor.service);
                    return transitionCpuFallback(aFrame, transition, bFrame, cpuGetImage, top, bottom, image, format, width, height, writable);
                }
            } else {
                struct mlt_image_s planes;
                mlt_image_set_values(&planes, top.host, top.format, top.width, top.height);
                opaque = mlt_image_is_opaque(&planes);
            }
            hasAlpha = !opaque;
        }
        if (!hasAlpha) {
            if (top.device && callerWantsDevice && top.width == ctx.width && top.height == ctx.height) {
                // Stays on the GPU for the caller
                job.passFrameProperties(aFrame, bFrame);
                *image = top.host;
                *format = mlt_image_rgba;
                *width = top.width;
                *height = top.height;
                attachDeviceImage(aProperties, std::move(top.device));
                logDebug("%s: top image passed through on the GPU", descriptor.service);
                return 0;
            }
            if (!ensureHost(bFrame, top)) {
                logError("%s: lost the GPU image", descriptor.service);
                return 1;
            }
            return job.passThrough(aFrame, bFrame, top.host, top.format, top.width, top.height, image, format, width, height);
        }
        if (pendingYuv422) {
            const CUresult res = convertTopBack(bFrame, top);
            if (res != CUDA_SUCCESS) {
                runtime.handleError(res, descriptor.service);
                return transitionCpuFallback(aFrame, transition, bFrame, cpuGetImage, top, bottom, image, format, width, height, writable);
            }
        }
        if (!isRgba(top.format) && hasConvertImage(bFrame)) {
            convertImage(bFrame, &top.host, &top.format, mlt_image_rgba);
        }
    }
    if (!isRgba(top.format)) {
        return transitionCpuFallback(aFrame, transition, bFrame, cpuGetImage, top, bottom, image, format, width, height, writable);
    }

    // Bottom image
    bottom.format = mlt_image_rgba;
    bottom.width = ctx.width;
    bottom.height = ctx.height;
    error = fetchImage(aFrame, bottomSource, &bottom.host, &bottom.format, &bottom.width, &bottom.height, job.bottomWritable);
    bottom.device = takeDeviceImage(aProperties, bottom.host, bottom.width, bottom.height, bottom.format);
    if (error || !bottom.host) {
        return error ? error : 1;
    }
    if (!isRgba(bottom.format) || !job.prepare(top.width, top.height, bottom.width, bottom.height)) {
        return transitionCpuFallback(aFrame, transition, bFrame, cpuGetImage, top, bottom, image, format, width, height, writable);
    }

    const size_t topBytes = size_t(top.width) * size_t(top.height) * 4;
    const size_t bytes = size_t(bottom.width) * size_t(bottom.height) * 4;
    const bool topFromDevice = top.device != nullptr;
    const bool bottomFromDevice = bottom.device != nullptr;
    CUresult res = CUDA_ERROR_NOT_INITIALIZED;
    {
        GpuScope scope;
        if (scope.ok()) {
            DeviceBuffer topBuffer;
            DeviceBuffer bottomBuffer;
            DeviceBuffer out;
            res = CUDA_SUCCESS;
            if (top.device) {
                topBuffer = std::move(top.device->buffer);
                top.device.reset();
            } else {
                res = topBuffer.allocate(topBytes);
                if (res == CUDA_SUCCESS) {
                    res = scope.upload(topBuffer.ptr(), top.host, topBytes);
                }
            }
            if (res == CUDA_SUCCESS) {
                if (bottom.device) {
                    bottomBuffer = std::move(bottom.device->buffer);
                    bottom.device.reset();
                } else {
                    res = bottomBuffer.allocate(bytes);
                    if (res == CUDA_SUCCESS) {
                        res = scope.upload(bottomBuffer.ptr(), bottom.host, bytes);
                    }
                }
            }
            if (res == CUDA_SUCCESS) {
                res = out.allocate(bytes);
            }
            if (res == CUDA_SUCCESS) {
                res = job.run(scope, topBuffer.ptr(), bottomBuffer.ptr(), out.ptr());
            }
            if (res == CUDA_SUCCESS && callerWantsDevice) {
                res = scope.synchronize();
                if (res == CUDA_SUCCESS) {
                    auto handoff = std::make_unique<DeviceImage>();
                    handoff->buffer = std::move(out);
                    handoff->width = bottom.width;
                    handoff->height = bottom.height;
                    handoff->tag = bottom.host;
                    attachDeviceImage(aProperties, std::move(handoff));
                    // The CPU transition drops the top image
                    mlt_frame_set_image(bFrame, nullptr, 0, nullptr);
                    *image = bottom.host;
                    *format = mlt_image_rgba;
                    *width = bottom.width;
                    *height = bottom.height;
                    runtime.countFrame(true);
                    logDebug("%s: %dx%d on GPU, kept on GPU", descriptor.service, bottom.width, bottom.height);
                    return 0;
                }
            } else if (res == CUDA_SUCCESS && deliversYuv422(aFrame, requestedFormat, bottom.width, bottom.height)) {
                res = deliverYuv422(scope, out.ptr(), aFrame, bottom.width, bottom.height, image, format, width, height);
                if (res == CUDA_SUCCESS) {
                    mlt_frame_set_image(bFrame, nullptr, 0, nullptr);
                    runtime.countFrame(true);
                    logDebug("%s: %dx%d on GPU, converted to YUV 4:2:2 on GPU", descriptor.service, bottom.width, bottom.height);
                    return 0;
                }
            } else if (res == CUDA_SUCCESS) {
                auto *result = allocateImage(bytes);
                res = result ? scope.download(result, out.ptr(), bytes) : CUDA_ERROR_OUT_OF_MEMORY;
                if (res == CUDA_SUCCESS) {
                    res = scope.synchronize();
                }
                if (res == CUDA_SUCCESS) {
                    mlt_frame_set_image(aFrame, result, int(bytes), releaseImage);
                    mlt_frame_set_image(bFrame, nullptr, 0, nullptr);
                    *image = result;
                    *format = mlt_image_rgba;
                    *width = bottom.width;
                    *height = bottom.height;
                    runtime.countFrame(true);
                    logDebug("%s: %dx%d on GPU", descriptor.service, bottom.width, bottom.height);
                    return 0;
                }
                if (result) {
                    releaseImage(result);
                }
            }
            runtime.handleError(res, descriptor.service);
            // Images that only existed on the GPU go back to their frames for the CPU
            if ((topFromDevice && !restoreToFrame(scope, topBuffer, bFrame, top)) ||
                (bottomFromDevice && !restoreToFrame(scope, bottomBuffer, aFrame, bottom))) {
                logError("%s: lost a GPU image after a CUDA error", descriptor.service);
                return 1;
            }
        }
    }
    logDebug("%s: falling back to the CPU", descriptor.service);
    return transitionCpuFallback(aFrame, transition, bFrame, cpuGetImage, top, bottom, image, format, width, height, writable);
}

int bridgeTransitionGetImage(mlt_frame aFrame, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    mlt_properties aProperties = MLT_FRAME_PROPERTIES(aFrame);
    auto cpuGetImage = reinterpret_cast<mlt_get_image>(mlt_frame_pop_service(aFrame));
    const bool callerWantsDevice = takeFlag(aProperties, kWantDevice);
    // A request for YUV is answered with a real YUV image here
    takeFlag(aProperties, kYuvRoundTrip);
    // The CPU transition entries: [transition, b_frame]
    mlt_deque stack = MLT_FRAME_IMAGE_STACK(aFrame);
    const int count = mlt_deque_count(stack);
    if (!cpuGetImage || count < 2) {
        return 1;
    }
    auto bFrame = static_cast<mlt_frame>(mlt_deque_peek(stack, count - 1));
    auto transition = static_cast<mlt_transition>(mlt_deque_peek(stack, count - 2));
    mlt_properties properties = MLT_TRANSITION_PROPERTIES(transition);
    auto *info = static_cast<TransitionWrapInfo *>(mlt_properties_get_data(properties, kWrapProperty, nullptr));
    if (!info || !moduleEnabled() || !serviceEnabled("transition", info->descriptor->service) || mlt_properties_get_int(properties, kForceCpu)) {
        return cpuGetImage(aFrame, image, format, width, height, writable);
    }
    const TransitionDescriptor &descriptor = *info->descriptor;

    TransitionContext ctx;
    ctx.transition = transition;
    ctx.aFrame = aFrame;
    ctx.bFrame = bFrame;
    ctx.properties = properties;
    ctx.position = mlt_transition_get_position(transition, aFrame);
    ctx.length = mlt_transition_get_length(transition);
    ctx.format = *format;
    ctx.width = *width;
    ctx.height = *height;
    mlt_frame topSource = gpuSourceBelowTransition(bFrame, 0, transition, aFrame);
    mlt_frame bottomSource = gpuSourceBelowTransition(aFrame, 2, transition, nullptr);
    ctx.topOnGpu = topSource != nullptr;
    ctx.bottomOnGpu = bottomSource != nullptr;
    ctx.callerOnGpu = callerWantsDevice;
    ctx.convertsOnGpu = !callerWantsDevice && deliversYuv422(aFrame, *format, *width, *height);
    TransitionPlan plan = descriptor.plan(ctx);
    if (plan.decision != Decision::Gpu || !plan.job || !Runtime::instance().ensureInitialized()) {
        if (plan.decision == Decision::Gpu) {
            Runtime::instance().countFrame(false);
        }
        return cpuGetImage(aFrame, image, format, width, height, writable);
    }
    return runTransitionOnGpu(aFrame, cpuGetImage, descriptor, plan, ctx, topSource, bottomSource, callerWantsDevice, image, format, width, height, writable);
}

mlt_frame wrappedTransitionProcess(mlt_transition transition, mlt_frame aFrame, mlt_frame bFrame)
{
    auto *info = static_cast<TransitionWrapInfo *>(mlt_properties_get_data(MLT_TRANSITION_PROPERTIES(transition), kWrapProperty, nullptr));
    if (!info) {
        return aFrame;
    }
    mlt_frame result = info->originalProcess(transition, aFrame, bFrame);
    if (result != aFrame || !moduleEnabled()) {
        return result;
    }
    // The CPU transition pushed [transition, b_frame, get_image]: insert the bridge on top
    mlt_deque stack = MLT_FRAME_IMAGE_STACK(aFrame);
    const int count = mlt_deque_count(stack);
    if (count >= 3 && mlt_deque_peek(stack, count - 2) == bFrame && mlt_deque_peek(stack, count - 3) == transition) {
        void *cpuGetImage = mlt_deque_pop_back(stack);
        mlt_frame_push_service(aFrame, cpuGetImage);
        mlt_frame_push_get_image(aFrame, bridgeTransitionGetImage);
    }
    return result;
}

mlt_frame wrappedProcess(mlt_filter filter, mlt_frame frame)
{
    auto *info = static_cast<WrapInfo *>(mlt_properties_get_data(MLT_FILTER_PROPERTIES(filter), kWrapProperty, nullptr));
    if (!info) {
        return frame;
    }
    mlt_frame result = info->originalProcess(filter, frame);
    if (result != frame || !moduleEnabled()) {
        return result;
    }
    // The CPU filter pushed itself and its get_image: insert the bridge on top
    mlt_deque stack = MLT_FRAME_IMAGE_STACK(frame);
    const int count = mlt_deque_count(stack);
    if (count >= 2 && mlt_deque_peek(stack, count - 2) == filter) {
        void *cpuGetImage = mlt_deque_pop_back(stack);
        mlt_frame_push_service(frame, cpuGetImage);
        mlt_frame_push_get_image(frame, bridgeGetImage);
    }
    return result;
}

/* ==========================================================================
 * Normalizing filters (see normalizer.h)
 * ========================================================================== */

struct NormalizerWrapInfo
{
    mlt_frame (*originalProcess)(mlt_filter, mlt_frame);
    const NormalizerDescriptor *descriptor;
};

void destroyNormalizerWrapInfo(void *data)
{
    delete static_cast<NormalizerWrapInfo *>(data);
}

int normalizerGetImage(mlt_frame frame, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    auto *info = static_cast<NormalizerWrapInfo *>(mlt_frame_pop_service(frame));
    const int entries = int(intptr_t(mlt_frame_pop_service(frame)));
    auto cpuGetImage = reinterpret_cast<mlt_get_image>(mlt_frame_pop_service(frame));
    const bool callerWantsDevice = takeFlag(properties, kWantDevice);
    takeFlag(properties, kYuvRoundTrip);
    if (!info || !cpuGetImage) {
        return 1;
    }
    const NormalizerDescriptor &descriptor = *info->descriptor;
    // Only worth it when the image below is on the GPU and stays there
    mlt_frame lowerSource = callerWantsDevice && *format == mlt_image_rgba && moduleEnabled() && serviceEnabled("conversion", "normalize")
                                ? gpuSource(frame, entries)
                                : nullptr;
    if (!lowerSource) {
        return cpuGetImage(frame, image, format, width, height, writable);
    }
    NormalizerCall call;
    call.frame = frame;
    call.properties = properties;
    mlt_deque stack = MLT_FRAME_IMAGE_STACK(frame);
    const int count = mlt_deque_count(stack);
    for (int i = 0; i < entries && i < 2; ++i) {
        call.entries[i] = mlt_deque_peek(stack, count - 1 - i);
    }
    call.width = *width;
    call.height = *height;
    if (!descriptor.before(call)) {
        return cpuGetImage(frame, image, format, width, height, writable);
    }
    // The CPU filter's own entries
    for (int i = 0; i < entries; ++i) {
        mlt_frame_pop_service(frame);
    }

    uint8_t *fetched = nullptr;
    mlt_image_format fetchedFormat = mlt_image_rgba;
    int w = call.fetchWidth;
    int h = call.fetchHeight;
    const int error = fetchImage(frame, lowerSource, &fetched, &fetchedFormat, &w, &h, writable);
    if (error || !fetched || fetchedFormat != mlt_image_rgba) {
        *image = fetched;
        *format = fetchedFormat;
        *width = w;
        *height = h;
        return error ? error : (fetched ? 0 : 1);
    }
    if (descriptor.after(call, w, h)) {
        // Unchanged: a GPU image goes on to the caller as it is
        *image = fetched;
        *format = mlt_image_rgba;
        *width = call.outWidth;
        *height = call.outHeight;
        return 0;
    }

    FetchedImage input;
    input.host = fetched;
    input.width = w;
    input.height = h;
    input.device = takeDeviceImage(properties, fetched, w, h, mlt_image_rgba);
    if (input.device && descriptor.gpuWork) {
        Runtime &runtime = Runtime::instance();
        GpuScope scope;
        DeviceBuffer out;
        CUresult res = scope.ok() ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED;
        if (res == CUDA_SUCCESS && descriptor.gpuWork(call, scope, input.device->buffer.ptr(), w, h, out, res)) {
            if (res == CUDA_SUCCESS) {
                res = scope.synchronize();
            }
            const int outBytes = call.outWidth * call.outHeight * 4;
            auto *tag = res == CUDA_SUCCESS ? static_cast<uint8_t *>(mlt_pool_alloc(outBytes)) : nullptr;
            if (tag) {
                // The host image stands for the GPU one
                mlt_frame_set_image(frame, tag, outBytes, mlt_pool_release);
                auto handoff = std::make_unique<DeviceImage>();
                handoff->buffer = std::move(out);
                handoff->width = call.outWidth;
                handoff->height = call.outHeight;
                handoff->tag = tag;
                attachDeviceImage(properties, std::move(handoff));
                *image = tag;
                *format = mlt_image_rgba;
                *width = call.outWidth;
                *height = call.outHeight;
                logDebug("%s: %dx%d -> %dx%d on GPU", descriptor.service, w, h, call.outWidth, call.outHeight);
                return 0;
            }
            runtime.handleError(res == CUDA_SUCCESS ? CUDA_ERROR_OUT_OF_MEMORY : res, descriptor.service);
        }
    }
    // The CPU filter's work, on the image in memory
    if (!ensureHost(frame, input)) {
        logError("%s: lost the GPU image", descriptor.service);
        return 1;
    }
    *image = input.host;
    *format = mlt_image_rgba;
    *width = w;
    *height = h;
    if (!descriptor.work) {
        return 0;
    }
    logDebug("%s: %dx%d changed on the CPU", descriptor.service, w, h);
    return descriptor.work(call, image, width, height);
}

mlt_frame wrappedNormalizerProcess(mlt_filter filter, mlt_frame frame)
{
    auto *info = static_cast<NormalizerWrapInfo *>(mlt_properties_get_data(MLT_FILTER_PROPERTIES(filter), kWrapProperty, nullptr));
    if (!info) {
        return frame;
    }
    mlt_deque stack = MLT_FRAME_IMAGE_STACK(frame);
    const int before = mlt_deque_count(stack);
    mlt_frame result = info->originalProcess(filter, frame);
    if (result != frame || !moduleEnabled()) {
        return result;
    }
    // The CPU filter pushed its entries and its get_image: unless the layout
    // is another one, insert the bridge on top
    if (mlt_deque_count(stack) - before == info->descriptor->entries + 1) {
        void *cpuGetImage = mlt_deque_pop_back(stack);
        mlt_frame_push_service(frame, cpuGetImage);
        mlt_frame_push_service(frame, reinterpret_cast<void *>(intptr_t(info->descriptor->entries)));
        mlt_frame_push_service(frame, info);
        mlt_frame_push_get_image(frame, normalizerGetImage);
    }
    return result;
}

/* ==========================================================================
 * Decoding
 *
 * MLT's avformat producer converts each decoded picture to the format its
 * caller asks for. For RGBA, the format of most effects and of compositing,
 * converting the usual 4:2:0 YUV video costs a lot of CPU time. The
 * producer's get_image is wrapped so that for an RGBA request the picture is
 * fetched as decoded, uploaded at its smaller size and converted on the GPU
 * with libswscale's arithmetic. When the next stage up the stack is another
 * bridge the RGBA image stays on the GPU.
 * ========================================================================== */

struct ProducerWrapInfo
{
    int (*originalGetFrame)(mlt_producer, mlt_frame_ptr, int);
};

void destroyProducerWrapInfo(void *data)
{
    delete static_cast<ProducerWrapInfo *>(data);
}

/** True if the producer decodes to 4:2:0 YUV, whose conversion to RGBA can
 *  be done on the GPU. */
bool decodesOnGpu(mlt_producer producer)
{
    if (!moduleEnabled() || !serviceEnabled("conversion", "avformat")) {
        return false;
    }
    mlt_properties properties = MLT_PRODUCER_PROPERTIES(producer);
    if (mlt_properties_get_int(properties, kForceCpu)) {
        return false;
    }
    // Hardware decoding and its scaling filters change the decoded format
    const char *hwaccel = mlt_properties_get(properties, "hwaccel");
    if (hwaccel && hwaccel[0] && strcmp(hwaccel, "none") != 0) {
        return false;
    }
    char key[64];
    snprintf(key, sizeof(key), "meta.media.%d.codec.pix_fmt", mlt_properties_get_int(properties, "video_index"));
    const char *pixFmt = mlt_properties_get(properties, key);
    return pixFmt && (!strcmp(pixFmt, "yuv420p") || !strcmp(pixFmt, "yuvj420p") || !strcmp(pixFmt, "yuva420p"));
}

int decodeBridgeGetImage(mlt_frame frame, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    auto producer = static_cast<mlt_producer>(mlt_frame_pop_service(frame));
    auto decoderGetImage = reinterpret_cast<mlt_get_image>(mlt_frame_pop_service(frame));
    const bool callerWantsDevice = takeFlag(properties, kWantDevice);
    // A request for YUV is answered by the producer itself
    takeFlag(properties, kYuvRoundTrip);
    if (!producer || !decoderGetImage) {
        return 1;
    }
    if (*format != mlt_image_rgba || !decodesOnGpu(producer) || !Runtime::instance().ensureInitialized()) {
        return decoderGetImage(frame, image, format, width, height, writable);
    }
    // The producer's own entry, which its get_image consumes
    void *decoder = mlt_deque_peek_back(MLT_FRAME_IMAGE_STACK(frame));

    // mlt_image_none asks for the picture as decoded, without conversion
    uint8_t *decoded = nullptr;
    mlt_image_format decodedFormat = mlt_image_none;
    int w = *width;
    int h = *height;
    const int error = decoderGetImage(frame, &decoded, &decodedFormat, &w, &h, writable);
    if (error || !decoded || decodedFormat == mlt_image_rgba) {
        *image = decoded;
        *format = decoded ? decodedFormat : mlt_image_rgba;
        *width = w;
        *height = h;
        return error;
    }

    Yuv420Image source;
    bool convertible = decodedFormat == mlt_image_yuv420p && w > 0 && h > 0 && !(w & 1) && !(h & 1);
    if (convertible) {
        source.data = decoded;
        source.width = w;
        source.height = h;
        source.colorspace = mlt_image_colorspace_id(mlt_properties_get(properties, "colorspace"));
        source.fullRange = mlt_properties_get_int(properties, "full_range") != 0;
        // The producer converts the fields of interlaced limited range pictures separately
        source.fields = !source.fullRange && !mlt_properties_get_int(properties, "progressive");
        int alphaSize = 0;
        uint8_t *alpha = mlt_frame_get_alpha_size(frame, &alphaSize);
        if (alpha && alphaSize >= w * h) {
            source.alpha = alpha;
        }
        convertible = !source.fields || !(h & 3);
    }
    if (convertible) {
        CUresult res = CUDA_ERROR_NOT_INITIALIZED;
        {
            GpuScope scope;
            if (scope.ok()) {
                const size_t bytes = size_t(w) * size_t(h) * 4;
                DeviceBuffer rgba;
                res = convertYuv420ToRgba(scope, source, rgba);
                // Page-locked only for a download: the host image standing for a GPU image is not filled
                auto *result =
                    res != CUDA_SUCCESS ? nullptr : callerWantsDevice ? static_cast<uint8_t *>(mlt_pool_alloc(int(bytes))) : allocateImage(bytes);
                if (res == CUDA_SUCCESS && !result) {
                    res = CUDA_ERROR_OUT_OF_MEMORY;
                }
                if (res == CUDA_SUCCESS && !callerWantsDevice) {
                    res = scope.download(result, rgba.ptr(), bytes);
                }
                if (res == CUDA_SUCCESS) {
                    res = scope.synchronize();
                }
                if (res == CUDA_SUCCESS) {
                    // With the caller on the GPU the host image only stands for the GPU one
                    mlt_frame_set_image(frame, result, int(bytes), releaseImage);
                    if (source.alpha) {
                        // Now in the RGBA image
                        mlt_frame_set_alpha(frame, nullptr, 0, nullptr);
                    }
                    // As set by the producer when it converts to RGBA
                    mlt_properties_set_int(properties, "full_range", mlt_image_full_range(mlt_properties_get(properties, "consumer.color_range")));
                    mlt_properties_set_int(properties, "format", mlt_image_rgba);
                    if (callerWantsDevice) {
                        auto handoff = std::make_unique<DeviceImage>();
                        handoff->buffer = std::move(rgba);
                        handoff->width = w;
                        handoff->height = h;
                        handoff->tag = result;
                        attachDeviceImage(properties, std::move(handoff));
                    }
                    *image = result;
                    *format = mlt_image_rgba;
                    *width = w;
                    *height = h;
                    logDebug("avformat: %dx%d converted to RGBA on GPU%s", w, h, callerWantsDevice ? ", kept on GPU" : "");
                    return 0;
                }
                releaseImage(result);
            }
        }
        Runtime::instance().handleError(res, "avformat");
    }
    // The producer converts its picture itself, as without gpufx. An alpha
    // plane taken out of the YUV picture is then part of the RGBA image.
    if (mlt_frame_get_alpha(frame)) {
        mlt_frame_set_alpha(frame, nullptr, 0, nullptr);
    }
    mlt_frame_push_service(frame, decoder);
    *format = mlt_image_rgba;
    return decoderGetImage(frame, image, format, width, height, writable);
}

int wrappedProducerGetFrame(mlt_producer producer, mlt_frame_ptr frame, int index)
{
    auto *info = static_cast<ProducerWrapInfo *>(mlt_properties_get_data(MLT_PRODUCER_PROPERTIES(producer), kWrapProperty, nullptr));
    if (!info) {
        return 1;
    }
    const int result = info->originalGetFrame(producer, frame, index);
    if (result || !frame || !*frame || !moduleEnabled()) {
        return result;
    }
    // The new frame holds the producer's [service, get_image] when it has video
    mlt_deque stack = MLT_FRAME_IMAGE_STACK(*frame);
    if (mlt_deque_count(stack) == 2) {
        void *getImage = mlt_deque_pop_back(stack);
        mlt_frame_push_service(*frame, getImage);
        mlt_frame_push_service(*frame, producer);
        mlt_frame_push_get_image(*frame, decodeBridgeGetImage);
    }
    return result;
}

} // namespace

void wrapTransition(mlt_transition transition, const TransitionDescriptor &descriptor)
{
    if (!transition || !transition->process || transition->process == wrappedTransitionProcess) {
        return;
    }
    auto *info = new TransitionWrapInfo{transition->process, &descriptor};
    mlt_properties_set_data(MLT_TRANSITION_PROPERTIES(transition), kWrapProperty, info, 0, destroyTransitionWrapInfo, nullptr);
    transition->process = wrappedTransitionProcess;
}

void wrapFilter(mlt_filter filter, const EffectDescriptor &effect)
{
    if (!filter || !filter->process || filter->process == wrappedProcess) {
        return;
    }
    auto *info = new WrapInfo{filter->process, &effect};
    mlt_properties_set_data(MLT_FILTER_PROPERTIES(filter), kWrapProperty, info, 0, destroyWrapInfo, nullptr);
    filter->process = wrappedProcess;
}

void wrapNormalizer(mlt_filter filter, const NormalizerDescriptor &descriptor)
{
    if (!filter || !filter->process || filter->process == wrappedNormalizerProcess) {
        return;
    }
    auto *info = new NormalizerWrapInfo{filter->process, &descriptor};
    mlt_properties_set_data(MLT_FILTER_PROPERTIES(filter), kWrapProperty, info, 0, destroyNormalizerWrapInfo, nullptr);
    filter->process = wrappedNormalizerProcess;
}

void wrapProducer(mlt_producer producer)
{
    if (!producer || !producer->get_frame || producer->get_frame == wrappedProducerGetFrame) {
        return;
    }
    auto *info = new ProducerWrapInfo{producer->get_frame};
    mlt_properties_set_data(MLT_PRODUCER_PROPERTIES(producer), kWrapProperty, info, 0, destroyProducerWrapInfo, nullptr);
    producer->get_frame = wrappedProducerGetFrame;
}

} // namespace gpufx
