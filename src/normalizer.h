/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#pragma once

#include "gpu_runtime.h"

#include <framework/mlt.h>

namespace gpufx {

/**
 * MLT puts normalizing filters between a producer and everything else: in a
 * Kdenlive clip (an MLT chain), color_transform, fieldorder, crop, rescale
 * (swscale) and resize. Most of the time they leave the image as it is,
 * apart from frame properties, but as CPU filters they need it in memory: a
 * decoded video frame left the GPU there, only to go back for the clip's
 * effects or for compositing. Each one is wrapped by the bridge
 * (wrapNormalizer() in bridge.cpp), which lets the image through on the GPU
 * when the filter would not change it, doing the filter's frame changes
 * itself, and otherwise has its work done (see normalizers.cpp).
 */

/** One frame of a normalizing filter, for an RGBA image. */
struct NormalizerCall
{
    mlt_frame frame = nullptr;
    mlt_properties properties = nullptr; // the frame's
    /** The filter's entries on the image stack below its get_image, from the
     *  top (see NormalizerDescriptor::entries) */
    void *entries[2] = {};
    /** The size asked for */
    int width = 0;
    int height = 0;
    /** Set by before(): the size to ask the image below for */
    int fetchWidth = 0;
    int fetchHeight = 0;
    /** Set by before(), for after() and the work */
    int targetWidth = 0;
    int targetHeight = 0;
    bool passAll = false;
    /** Set by after(): the size of the image returned */
    int outWidth = 0;
    int outHeight = 0;
};

struct NormalizerDescriptor
{
    /** MLT service names of the filter */
    const char *service;
    const char *alias;
    /** The number of entries the filter pushes on the image stack before its
     *  get_image. The bridge leaves frames with another layout to the filter. */
    int entries;
    /** Before the image below is fetched, for an RGBA request. Returns false,
     *  changing nothing, to leave the frame to the CPU filter. Otherwise does
     *  the frame changes the CPU filter makes before it fetches the image, and
     *  sets fetchWidth and fetchHeight. */
    bool (*before)(NormalizerCall &call);
    /** After the image below is fetched in RGBA, fetchedWidth x fetchedHeight.
     *  Returns true when the CPU filter would leave the pixels as they are,
     *  with its frame changes done and outWidth and outHeight set, false when
     *  it changes them (see gpuWork and work). */
    bool (*after)(NormalizerCall &call, int fetchedWidth, int fetchedHeight);
    /** The changes to the pixels done on the GPU, when possible: returns false
     *  if this case is left to work(). Otherwise converts in into out, sets
     *  outWidth and outHeight and does the frame changes. */
    bool (*gpuWork)(NormalizerCall &call, GpuScope &scope, CUdeviceptr in, int inWidth, int inHeight, DeviceBuffer &out, CUresult &result);
    /** The CPU filter's changes to the pixels and the frame, for the RGBA
     *  image in host memory that is the frame's image. Returns an MLT error
     *  code, with the image and its size set. */
    int (*work)(NormalizerCall &call, uint8_t **image, int *width, int *height);
};

const NormalizerDescriptor *findNormalizer(const char *service);

} // namespace gpufx
