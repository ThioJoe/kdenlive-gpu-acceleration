/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#include "stills.h"
#include "gpufx_log.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>

namespace gpufx {

namespace {

bool alwaysStill(mlt_producer, mlt_frame)
{
    return true;
}

int64_t noPeriod(mlt_producer, mlt_frame)
{
    return 0;
}

/** refresh_qimage() of qimage_wrapper.cpp shows image floor(position / ttl)
 *  of its sequence (of one image for a single picture) */
int64_t qimagePeriod(mlt_producer producer, mlt_frame frame)
{
    const int ttl = mlt_properties_get_int(MLT_PRODUCER_PROPERTIES(producer), "ttl");
    const mlt_position position = mlt_frame_original_position(frame) + mlt_producer_get_in(producer);
    return ttl > 0 ? int64_t(std::floor(double(position) / ttl)) : int64_t(position);
}

/** drawKdenliveTitle() of kdenlivetitle_wrapper.cpp draws the title again
 *  for every frame when it is animated: typewriter or revealed text, a
 *  moving view, or a typewriter effect giving its text to the frame */
bool titleStill(mlt_producer producer, mlt_frame frame)
{
    mlt_properties properties = MLT_PRODUCER_PROPERTIES(producer);
    return !mlt_properties_get(properties, "_animated") && !mlt_properties_get(properties, "_endrect") &&
           !mlt_properties_get(MLT_FRAME_PROPERTIES(frame), "_kdenlivetitle_typewriter_xml");
}

const StillDescriptor kStills[] = {
    {"qimage", alwaysStill, qimagePeriod},
    {"kdenlivetitle", titleStill, noPeriod},
    {"color", alwaysStill, noPeriod},
    {"colour", alwaysStill, noPeriod},
};

/** Properties that only tell the position: of a frame (MLT's producers,
 *  frames and playlists, and pos.<id> of each filter in
 *  mlt_filter_process()) and of a producer (mlt_producer_seek()). A name
 *  ending with a dot stands for all the names it starts. */
const char *const kFramePositions[] = {"_position", "original_position", "meta.playlist.clip_position", "pos.", nullptr};
const char *const kProducerPositions[] = {"_position", "_frame", nullptr};

bool listed(const char *name, const char *const *list)
{
    for (; *list; ++list) {
        const size_t length = strlen(*list);
        if ((*list)[length - 1] == '.' ? !strncmp(name, *list, length) : !strcmp(name, *list)) {
            return true;
        }
    }
    return false;
}

/** The values of the properties, one "name=value" line each, apart from
 *  data and the excluded ones */
void appendProperties(std::string &out, mlt_properties properties, const char *const *excluded)
{
    const int count = mlt_properties_count(properties);
    for (int i = 0; i < count; ++i) {
        const char *name = mlt_properties_get_name(properties, i);
        if (!name || listed(name, excluded) || mlt_properties_get_data_at(properties, i, nullptr)) {
            continue;
        }
        const char *value = mlt_properties_get_value(properties, i);
        out += name;
        out += value ? "=" : "~";
        if (value) {
            out += value;
        }
        out += '\n';
    }
}

bool sameValues(mlt_properties a, mlt_properties b)
{
    const int count = mlt_properties_count(a);
    if (count != mlt_properties_count(b)) {
        return false;
    }
    for (int i = 0; i < count; ++i) {
        const char *name = mlt_properties_get_name(a, i);
        const char *value = mlt_properties_get_value(a, i);
        const char *other = name ? mlt_properties_get(b, name) : nullptr;
        if (!name || (value == nullptr) != (other == nullptr) || (value && strcmp(value, other))) {
            return false;
        }
    }
    return true;
}

/** The first line that differs, for the log */
std::string firstDifference(const std::string &a, const std::string &b)
{
    size_t start = 0;
    while (start < a.size() && start < b.size()) {
        const size_t endA = a.find('\n', start);
        const size_t endB = b.find('\n', start);
        if (endA != endB || a.compare(start, endA - start, b, start, endB - start)) {
            return a.substr(start, endA - start) + " / " + b.substr(start, endB - start);
        }
        start = endA + 1;
    }
    return "length";
}

/**
 * The confirmed pictures of all still producers, within a budget of GPU
 * memory: the least recently used go first.
 */
class StillCache
{
public:
    static StillCache &instance()
    {
        static StillCache cache;
        return cache;
    }

    std::shared_ptr<const StillImage> find(uint64_t producer, const std::string &request)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_entries.find(Key(producer, request));
        if (it == m_entries.end()) {
            return nullptr;
        }
        m_lru.splice(m_lru.begin(), m_lru, it->second.lru);
        return it->second.image;
    }

    void insert(uint64_t producer, const std::string &request, std::shared_ptr<StillImage> image)
    {
        const size_t bytes = image->buffer.size();
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_limit) {
            size_t limit = std::min<size_t>(Runtime::instance().device().totalMemory / 8, size_t(1) << 30);
            if (const char *mb = getenv("GPUFX_STILLS_MB")) {
                limit = size_t(atoi(mb)) << 20;
            }
            m_limit = std::max<size_t>(limit, 1);
        }
        const Key key(producer, request);
        eraseLocked(key);
        while (!m_lru.empty() && m_bytes + bytes > m_limit) {
            eraseLocked(m_lru.back());
        }
        if (bytes > m_limit) {
            return;
        }
        m_lru.push_front(key);
        m_entries[key] = Entry{std::move(image), m_lru.begin()};
        m_bytes += bytes;
    }

    void purge(uint64_t producer)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto it = m_entries.lower_bound(Key(producer, std::string())); it != m_entries.end() && it->first.first == producer;) {
            m_bytes -= it->second.image->buffer.size();
            m_lru.erase(it->second.lru);
            it = m_entries.erase(it);
        }
    }

private:
    using Key = std::pair<uint64_t, std::string>;
    struct Entry
    {
        std::shared_ptr<StillImage> image;
        std::list<Key>::iterator lru;
    };

    void eraseLocked(const Key &key)
    {
        auto it = m_entries.find(key);
        if (it != m_entries.end()) {
            m_bytes -= it->second.image->buffer.size();
            m_lru.erase(it->second.lru);
            m_entries.erase(it);
        }
    }

    std::mutex m_mutex;
    std::map<Key, Entry> m_entries;
    std::list<Key> m_lru; // the most recent first
    size_t m_bytes = 0;
    size_t m_limit = 0;
};

std::atomic<uint64_t> g_nextProducerId{1};

/** Consecutive results of a producer that did not confirm the one before,
 *  after which its pictures are no longer kept */
constexpr int kMaxMismatches = 8;

} // namespace

const StillDescriptor *findStill(const char *service)
{
    for (const StillDescriptor &descriptor : kStills) {
        if (service && !strcmp(service, descriptor.service)) {
            return &descriptor;
        }
    }
    return nullptr;
}

StillImage::~StillImage()
{
    mlt_properties_close(changes);
}

void StillImage::apply(mlt_frame frame) const
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    const int count = mlt_properties_count(changes);
    for (int i = 0; i < count; ++i) {
        mlt_properties_pass_property(properties, changes, mlt_properties_get_name(changes, i));
    }
    if (!alpha.empty()) {
        auto *plane = static_cast<uint8_t *>(mlt_pool_alloc(int(alpha.size())));
        memcpy(plane, alpha.data(), alpha.size());
        mlt_frame_set_alpha(frame, plane, int(alpha.size()), mlt_pool_release);
    }
}

FrameSnapshot::FrameSnapshot(mlt_frame frame)
    : m_values(mlt_properties_new())
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    const int count = mlt_properties_count(properties);
    for (int i = 0; i < count; ++i) {
        const char *name = mlt_properties_get_name(properties, i);
        if (!name) {
            continue;
        }
        if (void *data = mlt_properties_get_data_at(properties, i, nullptr)) {
            m_data.emplace_back(name, data);
        } else {
            mlt_properties_pass_property(m_values, properties, name);
        }
    }
}

FrameSnapshot::~FrameSnapshot()
{
    mlt_properties_close(m_values);
}

mlt_properties FrameSnapshot::changes(mlt_frame frame) const
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    mlt_properties result = mlt_properties_new();
    const int count = mlt_properties_count(properties);
    for (int i = 0; i < count; ++i) {
        const char *name = mlt_properties_get_name(properties, i);
        if (!name) {
            continue;
        }
        if (void *data = mlt_properties_get_data_at(properties, i, nullptr)) {
            if (!strcmp(name, "image") || !strcmp(name, "alpha")) {
                continue;
            }
            bool unchanged = false;
            for (const auto &entry : m_data) {
                if (entry.first == name) {
                    unchanged = entry.second == data;
                    break;
                }
            }
            if (!unchanged) {
                mlt_properties_close(result);
                return nullptr;
            }
            continue;
        }
        const char *now = mlt_properties_get_value(properties, i);
        const char *before = mlt_properties_get(m_values, name);
        if ((now == nullptr) != (before == nullptr) || (now && strcmp(now, before))) {
            mlt_properties_pass_property(result, properties, name);
        }
    }
    return result;
}

struct StillProducer::Pending
{
    Pending() = default;
    Pending(const Pending &) = delete;
    Pending &operator=(const Pending &) = delete;
    ~Pending() { mlt_properties_close(changes); }

    std::string request;
    std::vector<uint8_t> image;
    int width = 0;
    int height = 0;
    mlt_properties changes = nullptr;
    std::vector<uint8_t> alpha;
};

StillProducer::StillProducer(mlt_producer producer, const StillDescriptor &descriptor)
    : m_producer(producer)
    , m_descriptor(descriptor)
    , m_id(g_nextProducerId.fetch_add(1))
{
}

StillProducer::~StillProducer()
{
    StillCache::instance().purge(m_id);
}

bool StillProducer::enabled() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_mismatches < kMaxMismatches;
}

std::string StillProducer::request(mlt_frame frame, int width, int height) const
{
    std::string result = std::to_string(width) + "x" + std::to_string(height) + "@" + std::to_string(m_descriptor.period(m_producer, frame)) + "\n";
    appendProperties(result, MLT_FRAME_PROPERTIES(frame), kFramePositions);
    result += "--\n";
    // Other threads may be getting frames of the producer meanwhile
    mlt_properties producerProperties = MLT_PRODUCER_PROPERTIES(m_producer);
    mlt_properties_lock(producerProperties);
    appendProperties(result, producerProperties, kProducerPositions);
    mlt_properties_unlock(producerProperties);
    return result;
}

std::shared_ptr<const StillImage> StillProducer::find(const std::string &request)
{
    return StillCache::instance().find(m_id, request);
}

void StillProducer::record(const std::string &request, const uint8_t *image, int width, int height, mlt_properties changes, const uint8_t *alpha,
                           int alphaSize)
{
    const size_t bytes = size_t(width) * size_t(height) * 4;
    std::unique_ptr<Pending> previous;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        previous = std::move(m_pending);
    }
    const bool confirmed = previous && previous->request == request && previous->width == width && previous->height == height &&
                           !memcmp(previous->image.data(), image, bytes) && previous->alpha.size() == size_t(std::max(alphaSize, 0)) &&
                           (previous->alpha.empty() || !memcmp(previous->alpha.data(), alpha, previous->alpha.size())) &&
                           sameValues(previous->changes, changes) && sameValues(changes, previous->changes);
    if (confirmed) {
        mlt_properties_close(changes);
        auto picture = std::make_shared<StillImage>();
        CUresult res = CUDA_ERROR_NOT_INITIALIZED;
        {
            GpuScope scope;
            if (scope.ok()) {
                res = picture->buffer.allocate(bytes);
                if (res == CUDA_SUCCESS) {
                    res = scope.upload(picture->buffer.ptr(), previous->image.data(), bytes);
                }
                if (res == CUDA_SUCCESS) {
                    res = scope.synchronize();
                }
            }
        }
        if (res != CUDA_SUCCESS) {
            Runtime::instance().handleError(res, m_descriptor.service);
            return;
        }
        picture->width = width;
        picture->height = height;
        picture->changes = previous->changes;
        previous->changes = nullptr;
        picture->alpha = std::move(previous->alpha);
        StillCache::instance().insert(m_id, request, std::move(picture));
        std::lock_guard<std::mutex> lock(m_mutex);
        m_mismatches = 0;
        logDebug("%s: %dx%d picture kept on GPU", m_descriptor.service, width, height);
        return;
    }

    auto pending = std::make_unique<Pending>();
    pending->request = request;
    pending->image.assign(image, image + bytes);
    pending->width = width;
    pending->height = height;
    pending->changes = changes;
    if (alpha && alphaSize > 0) {
        pending->alpha.assign(alpha, alpha + alphaSize);
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (previous) {
        ++m_mismatches;
        logDebug("%s: picture not confirmed (%s)%s", m_descriptor.service,
                 previous->request == request ? "different result" : firstDifference(previous->request, request).c_str(),
                 m_mismatches >= kMaxMismatches ? ", no longer kept on GPU" : "");
    }
    m_pending = std::move(pending);
}

} // namespace gpufx
