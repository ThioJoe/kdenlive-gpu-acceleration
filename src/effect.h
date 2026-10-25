/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#pragma once

#include "gpu_runtime.h"

#include <framework/mlt.h>

#include <memory>
#include <vector>

namespace gpufx {

/** What the effect is asked to do for one frame, read before the image is fetched. */
struct EffectContext
{
    mlt_filter filter = nullptr;
    mlt_frame frame = nullptr;
    mlt_properties properties = nullptr; // the filter's properties
    mlt_position position = 0;
    mlt_position length = 0;
    // The format and size requested by the caller of get_image
    mlt_image_format format = mlt_image_none;
    int width = 0;
    int height = 0;
};

/** The GPU work for one frame, with the parameters already decoded. */
class EffectJob
{
public:
    virtual ~EffectJob() = default;
    /** Processes width x height RGBA pixels from in to out (device memory,
     *  4 bytes per pixel, not overlapping) on the scope's stream. */
    virtual CUresult run(GpuScope &scope, int width, int height, CUdeviceptr in, CUdeviceptr out) = 0;
    /** Called right before the input is fetched, for the frame changes the
     *  CPU filter makes at that point (they can matter to the filters below). */
    virtual void beforeFetch(mlt_frame frame) { (void)frame; }
    /** Called once the input is fetched, with its size. Returning false
     *  leaves the frame to the CPU filter. */
    virtual bool prepare(int inputWidth, int inputHeight)
    {
        (void)inputWidth;
        (void)inputHeight;
        return true;
    }
    /** For effects whose output size differs from the input size (see
     *  EffectPlan::outputWidth). */
    virtual CUresult runResized(GpuScope &scope, int inWidth, int inHeight, CUdeviceptr in, int outWidth, int outHeight, CUdeviceptr out)
    {
        (void)outWidth;
        (void)outHeight;
        return run(scope, inWidth, inHeight, in, out);
    }
    /** Undoes the changes of beforeFetch(), before the CPU filter processes
     *  the frame after all. */
    virtual void restoreFrame(mlt_frame frame) { (void)frame; }
};

enum class Decision {
    Cpu,        // let the original CPU filter process the frame
    Gpu,        // run the job on the GPU
    PassThrough // the filter does nothing for these parameters
};

struct EffectPlan
{
    Decision decision = Decision::Cpu;
    std::unique_ptr<EffectJob> job;
    /** For effects that do not keep the image size: the size of the input to
     *  request, and the size of the output. 0 keeps the requested size. */
    int inputWidth = 0;
    int inputHeight = 0;
    int outputWidth = 0;
    int outputHeight = 0;

    static EffectPlan cpu() { return {}; }
    static EffectPlan passThrough()
    {
        EffectPlan plan;
        plan.decision = Decision::PassThrough;
        return plan;
    }
    static EffectPlan gpu(std::unique_ptr<EffectJob> job)
    {
        EffectPlan plan;
        plan.decision = Decision::Gpu;
        plan.job = std::move(job);
        return plan;
    }
};

struct EffectDescriptor
{
    /** MLT service name of the CPU filter that gets accelerated */
    const char *service;
    /** Cheap per pixel effects: the CPU version is already fast, so they only
     *  run on the GPU when the frame is on the GPU anyway, next to another
     *  accelerated effect. */
    bool lightweight;
    /** Reads the parameters for this frame (with the filter locked). Must not
     *  fetch the image nor change the frame (see EffectJob::beforeFetch()).
     *  Only rgba output is supported on the GPU, the plan
     *  must return Decision::Cpu when the CPU filter would produce anything
     *  else for ctx.format. */
    EffectPlan (*plan)(const EffectContext &ctx);
};

/** What a transition is asked to do for one frame, read before any image is
 *  fetched. The transition draws the top frame (b) onto the bottom frame (a). */
struct TransitionContext
{
    mlt_transition transition = nullptr;
    mlt_frame aFrame = nullptr;
    mlt_frame bFrame = nullptr;
    mlt_properties properties = nullptr; // the transition's properties
    mlt_position position = 0;
    mlt_position length = 0;
    // The format and size requested by the caller of get_image
    mlt_image_format format = mlt_image_none;
    int width = 0;
    int height = 0;
    // Whether the top and bottom images, or the caller, are on the GPU anyway
    bool topOnGpu = false;
    bool bottomOnGpu = false;
    bool callerOnGpu = false;
    // Whether a result computed on the GPU would also be converted to the
    // caller's format there (see deliversYuv() in bridge.cpp)
    bool convertsOnGpu = false;
};

/** The GPU work of a transition for one frame. */
class TransitionJob
{
public:
    virtual ~TransitionJob() = default;

    /** The top image to request */
    mlt_image_format topFormat = mlt_image_rgba;
    int topWidth = 0;
    int topHeight = 0;
    /** The top image is returned unchanged, without the bottom one, when it
     *  turns out to have no transparency (MLT's qtblend does that). topAlpha
     *  tells if the top image has an alpha channel before it is fetched. */
    bool passOpaqueTop = false;
    bool topAlpha = false;
    /** Called right before the top image is fetched, for the frame changes
     *  the CPU transition makes at that point. */
    virtual void beforeFetch(mlt_frame aFrame, mlt_frame bFrame)
    {
        (void)aFrame;
        (void)bFrame;
    }
    /** The frame properties the CPU transition copies from the top frame
     *  when it returns the top image unchanged. */
    virtual void passFrameProperties(mlt_frame aFrame, mlt_frame bFrame) = 0;
    /** The CPU transition's way of returning the top image unchanged, for a
     *  top image in host memory. */
    virtual int passThrough(mlt_frame aFrame, mlt_frame bFrame, uint8_t *top, mlt_image_format topFormat, int topWidth, int topHeight, uint8_t **image,
                            mlt_image_format *format, int *width, int *height) = 0;
    /** The bottom image is fetched with this writable flag. */
    int bottomWritable = 0;
    /** The format of the result: RGBA, or YUV 4:2:2 (mlt_image_yuv422), which
     *  run() writes to out as YUYV followed by the alpha plane of the result
     *  at yuv422AlphaOffset(). */
    mlt_image_format resultFormat = mlt_image_rgba;
    /** Called once both images are fetched (the top one in RGBA). Returning
     *  false leaves the frame to the CPU transition. */
    virtual bool prepare(int topWidth, int topHeight, int bottomWidth, int bottomHeight) = 0;
    /** Draws top onto bottom into out (bottom size). */
    virtual CUresult run(GpuScope &scope, CUdeviceptr top, CUdeviceptr bottom, CUdeviceptr out) = 0;
    /** Called when the result of run() is delivered, for the frame
     *  properties the CPU transition sets. */
    virtual void finishFrame(mlt_frame aFrame) { (void)aFrame; }
};

/** Where the alpha plane follows a YUYV image of this size in GPU memory */
inline size_t yuv422AlphaOffset(int width, int height)
{
    return (size_t(width) * size_t(height) * 2 + 255) & ~size_t(255);
}

struct TransitionPlan
{
    Decision decision = Decision::Cpu;
    std::unique_ptr<TransitionJob> job;

    static TransitionPlan cpu() { return {}; }
    static TransitionPlan gpu(std::unique_ptr<TransitionJob> job)
    {
        TransitionPlan plan;
        plan.decision = Decision::Gpu;
        plan.job = std::move(job);
        return plan;
    }
};

struct TransitionDescriptor
{
    /** MLT service name of the CPU transition that gets accelerated */
    const char *service;
    /** Reads the parameters for this frame. Must not fetch images nor change
     *  the frames (see TransitionJob::beforeFetch()). */
    TransitionPlan (*plan)(const TransitionContext &ctx);
};

/** All the effects with a GPU implementation */
const std::vector<const EffectDescriptor *> &effectRegistry();
const EffectDescriptor *findEffect(const char *service);
const std::vector<const TransitionDescriptor *> &transitionRegistry();
const TransitionDescriptor *findTransition(const char *service);

// Implemented in the effect_*.cpp files
const EffectDescriptor &dropShadowEffect();
const EffectDescriptor &boxBlurEffect();
const EffectDescriptor &rectBlurEffect();
const EffectDescriptor &brightnessEffect();
const EffectDescriptor &liftGammaGainEffect();
const EffectDescriptor &frei0rSquareBlurEffect();
const EffectDescriptor &frei0rGlowEffect();
const EffectDescriptor &frei0rSoftGlowEffect();
const EffectDescriptor &frei0rSharpnessEffect();
const EffectDescriptor &frei0rScale0TiltEffect();
const EffectDescriptor &transformEffect();
const TransitionDescriptor &compositeTransition();
const TransitionDescriptor &lumaTransition();

} // namespace gpufx
