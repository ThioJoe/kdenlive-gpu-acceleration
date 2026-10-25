/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#pragma once

#include "gpu_runtime.h"
#include "sws_port.h"

#include <cstdint>

namespace gpufx {

/**
 * Image format conversions that MLT does with libswscale, done on the GPU
 * with the same results (see gpufx_kernels.cu and sws_port.cpp).
 */

/** A yuv420p image of MLT in host memory: the luma plane followed by the
 *  two chroma planes of half the width and height. */
struct Yuv420Image
{
    const uint8_t *data = nullptr;
    /** Optional alpha plane of width x height (yuva420p sources) */
    const uint8_t *alpha = nullptr;
    int width = 0;
    int height = 0;
    /** mlt_colorspace of the YUV values */
    int colorspace = 0;
    bool fullRange = false;
    /** Convert the two fields separately, as MLT does for interlaced
     *  limited range frames (the height must be a multiple of 4) */
    bool fields = false;
};

/** Uploads a yuv420p image and converts it to RGBA as MLT's avformat producer
 *  does (sws_scale() with SWS_POINT | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT).
 *  out receives width x height RGBA pixels. Width and height must be even.
 *  Does not synchronize the stream. */
CUresult convertYuv420ToRgba(GpuScope &scope, const Yuv420Image &image, DeviceBuffer &out);

/** Converts an RGBA image on the GPU to packed 4:2:2 YUV (mlt_image_yuv422)
 *  as MLT's avcolour_space does (sws_scale() with SWS_BICUBIC |
 *  SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT) and downloads it, along with the
 *  alpha channel when alpha is not null. colorspace is the mlt_colorspace of
 *  the output. Full range output follows FFmpeg 7.1 and older. The width must
 *  be even. Synchronizes the stream. */
CUresult convertRgbaToYuv422(GpuScope &scope, CUdeviceptr rgba, int width, int height, int colorspace, bool fullRange, uint8_t *yuv422, uint8_t *alpha);

/** Converts an RGBA image on the GPU to packed 4:2:2 YUV and back to RGBA, as
 *  MLT's avcolour_space does when an RGBA image is asked for in YUV 4:2:2 and
 *  then converted back: to limited range YUV in colorspace as
 *  convertRgbaToYuv422() does, back from backColorspace and backFullRange
 *  (sws_scale() with SWS_POINT | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT). The
 *  alpha channel is kept. out receives width x height RGBA pixels. The width
 *  must be even. Does not synchronize the stream. */
CUresult convertRgbaThroughYuv422(GpuScope &scope, CUdeviceptr rgba, int width, int height, int colorspace, int backColorspace, bool backFullRange,
                                  DeviceBuffer &out);

/** Scales an RGBA image on the GPU as MLT's rescale filter does with its
 *  swscale scaler (sws_scale_frame() with SWS_FULL_CHR_H_INP |
 *  SWS_FULL_CHR_H_INT | SWS_ACCURATE_RND), through YUV: converted with the
 *  coefficients of dstColorspace and back with those of srcColorspace
 *  (mlt_colorspace values, as mlt_set_luma_transfer() sets them). Luma and
 *  alpha are scaled with lumaMethod, chroma with chromaMethod. out receives
 *  dstWidth x dstHeight RGBA pixels. Returns CUDA_ERROR_NOT_SUPPORTED, doing
 *  nothing, for the scalings libswscale does in two passes or differently
 *  between versions. Does not synchronize the stream. */
CUresult scaleRgba(GpuScope &scope, CUdeviceptr rgba, int width, int height, int dstWidth, int dstHeight, sws::ScaleMethod lumaMethod,
                   sws::ScaleMethod chromaMethod, int srcColorspace, int dstColorspace, DeviceBuffer &out);

} // namespace gpufx
