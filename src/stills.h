/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#pragma once

#include "gpu_runtime.h"

#include <framework/mlt.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gpufx {

/**
 * Producers whose picture stays the same from frame to frame: still images
 * (qimage), titles (kdenlivetitle) and color clips. Every frame, they copy
 * the picture they keep into a new image, which an effect or the
 * compositing on the GPU then uploads. The bridge wraps their get_image
 * (wrapStill() in bridge.cpp): once two calls with the same request gave
 * the same image and made the same changes to their frames, the picture is
 * kept on the GPU, and the next frames with that request get a copy of it
 * there and the same changes, without the producer.
 *
 * A request is everything the producer's get_image can read: the size
 * asked for, the frame's properties and the producer's, apart from those
 * that only tell the frame's position. Where the picture depends on the
 * position, the descriptor tells how.
 */

struct StillDescriptor
{
    /** MLT service name of the producer */
    const char *service;
    /** After the producer's get_image, with its frame: false when its
     *  picture may change with the frame's position. */
    bool (*still)(mlt_producer producer, mlt_frame frame);
    /** The part of the frame's position the picture depends on, if any (a
     *  qimage producer shows the image of floor(position / ttl) of its
     *  sequence) */
    int64_t (*period)(mlt_producer producer, mlt_frame frame);
};

const StillDescriptor *findStill(const char *service);

/** A picture of a still producer on the GPU, with what its get_image does
 *  to a frame */
struct StillImage
{
    StillImage() = default;
    StillImage(const StillImage &) = delete;
    StillImage &operator=(const StillImage &) = delete;
    ~StillImage();

    DeviceBuffer buffer;
    int width = 0;
    int height = 0;
    /** The frame properties to set */
    mlt_properties changes = nullptr;
    /** The alpha plane to set, if any */
    std::vector<uint8_t> alpha;

    /** Makes frame what the producer's get_image would make of it, apart
     *  from its image */
    void apply(mlt_frame frame) const;
};

/** The frame properties before a get_image, to find what it changed */
class FrameSnapshot
{
public:
    explicit FrameSnapshot(mlt_frame frame);
    ~FrameSnapshot();
    FrameSnapshot(const FrameSnapshot &) = delete;
    FrameSnapshot &operator=(const FrameSnapshot &) = delete;

    /** The properties of frame set since, as a new properties list, or null
     *  if other data than the image and its alpha plane changed */
    mlt_properties changes(mlt_frame frame) const;

private:
    mlt_properties m_values;
    std::vector<std::pair<std::string, void *>> m_data;
};

/** The pictures of a still producer. The producer owns it (a property), so
 *  that its pictures go with it. */
class StillProducer
{
public:
    StillProducer(mlt_producer producer, const StillDescriptor &descriptor);
    ~StillProducer();
    StillProducer(const StillProducer &) = delete;
    StillProducer &operator=(const StillProducer &) = delete;

    const StillDescriptor &descriptor() const { return m_descriptor; }
    mlt_producer producer() const { return m_producer; }
    /** False after the producer kept making different pictures for the
     *  same request */
    bool enabled() const;

    /** The request of a get_image with this frame and size */
    std::string request(mlt_frame frame, int width, int height) const;
    /** The picture for a request, when confirmed */
    std::shared_ptr<const StillImage> find(const std::string &request);
    /** Records what the producer's get_image made for a request: an RGBA
     *  image of width x height in image, the frame properties it set
     *  (changes, taken over) and its alpha plane, if any. Uploads the
     *  picture when the same request gave the same result before. */
    void record(const std::string &request, const uint8_t *image, int width, int height, mlt_properties changes, const uint8_t *alpha, int alphaSize);

private:
    struct Pending;
    mlt_producer m_producer;
    const StillDescriptor &m_descriptor;
    const uint64_t m_id;
    mutable std::mutex m_mutex;
    std::unique_ptr<Pending> m_pending;
    int m_mismatches = 0;
};

} // namespace gpufx
