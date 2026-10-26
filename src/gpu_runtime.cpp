/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#include "gpu_runtime.h"
#include "gpufx_log.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Generated from kernels/gpufx_kernels.ptx at build time (see CMakeLists.txt)
extern const unsigned char gpufx_ptx[];
extern const size_t gpufx_ptx_size;

namespace gpufx {

namespace {

const char *const kKernelNames[] = {
#define GPUFX_KERNEL_NAME(name) #name,
    GPUFX_KERNEL_LIST(GPUFX_KERNEL_NAME)
#undef GPUFX_KERNEL_NAME
};

// Pool sizes are rounded up so that buffers whose size changes slightly from
// frame to frame (animated blur radius, ...) can be reused.
size_t roundAllocation(size_t bytes)
{
    const size_t granule = bytes >= (size_t(1) << 20) ? (size_t(1) << 20) : (size_t(1) << 16);
    if (bytes == 0) {
        bytes = 1;
    }
    return (bytes + granule - 1) / granule * granule;
}

bool isRecoverable(CUresult result)
{
    return result == CUDA_ERROR_OUT_OF_MEMORY || result == CUDA_ERROR_INVALID_VALUE;
}

// Uploads from this size on go through page-locked memory
constexpr size_t kStagingBytes = size_t(256) << 10;

thread_local GpuScope *t_scope = nullptr;

} // namespace

Runtime &Runtime::instance()
{
    // Never destroyed: CUDA must not be called from static destructors.
    static Runtime *runtime = new Runtime();
    return *runtime;
}

bool Runtime::ensureInitialized()
{
    State state = m_state.load();
    if (state == State::Ready) {
        return true;
    }
    if (state != State::Uninitialized) {
        return false;
    }
    std::lock_guard<std::mutex> lock(m_initMutex);
    if (m_state.load() == State::Uninitialized) {
        const bool ok = initialize();
        m_state.store(ok ? State::Ready : State::Unavailable);
    }
    return m_state.load() == State::Ready;
}

bool Runtime::initialize()
{
    std::string error;
    if (!m_cu.load(error)) {
        m_status = error;
        logInfo("GPU effects unavailable: %s", error.c_str());
        return false;
    }
    CUresult res = m_cu.Init(0);
    if (res != CUDA_SUCCESS) {
        m_status = "CUDA initialization failed: " + m_cu.errorText(res);
        logInfo("GPU effects unavailable: %s", m_status.c_str());
        return false;
    }
    m_cu.DriverGetVersion(&m_device.driverVersion);
    int count = 0;
    res = m_cu.DeviceGetCount(&count);
    if (res != CUDA_SUCCESS || count <= 0) {
        m_status = "No CUDA capable GPU found";
        logInfo("GPU effects unavailable: %s", m_status.c_str());
        return false;
    }
    int ordinal = 0;
    if (const char *wanted = getenv("GPUFX_CUDA_DEVICE")) {
        ordinal = atoi(wanted);
        if (ordinal < 0 || ordinal >= count) {
            ordinal = 0;
        }
    }
    CUdevice device = 0;
    res = m_cu.DeviceGet(&device, ordinal);
    if (res != CUDA_SUCCESS) {
        m_status = "Cannot open CUDA device: " + m_cu.errorText(res);
        logInfo("GPU effects unavailable: %s", m_status.c_str());
        return false;
    }
    char name[256] = {0};
    m_cu.DeviceGetName(name, int(sizeof(name)) - 1, device);
    m_device.name = name;
    m_device.ordinal = ordinal;
    m_cu.DeviceGetAttribute(&m_device.computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    m_cu.DeviceGetAttribute(&m_device.computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    m_cu.DeviceGetAttribute(&m_device.multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    m_cu.DeviceTotalMem(&m_device.totalMemory, device);
    if (m_device.computeMajor < 5) {
        m_status = "GPU " + m_device.name + " is too old (compute capability 5.0 or newer is needed)";
        logInfo("GPU effects unavailable: %s", m_status.c_str());
        return false;
    }

    res = m_cu.DevicePrimaryCtxRetain(&m_context, device);
    if (res != CUDA_SUCCESS) {
        m_status = "Cannot create a CUDA context: " + m_cu.errorText(res);
        logInfo("GPU effects unavailable: %s", m_status.c_str());
        return false;
    }
    res = m_cu.CtxPushCurrent(m_context);
    if (res != CUDA_SUCCESS) {
        m_status = "Cannot use the CUDA context: " + m_cu.errorText(res);
        logInfo("GPU effects unavailable: %s", m_status.c_str());
        return false;
    }

    // The PTX is compiled for the GPU by the driver. The result is cached by
    // the driver, so only the very first run pays for it.
    char errorLog[4096] = {0};
    int options[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void *values[] = {errorLog, reinterpret_cast<void *>(uintptr_t(sizeof(errorLog) - 1))};
    res = m_cu.ModuleLoadDataEx(&m_module, gpufx_ptx, 2, options, values);
    bool ok = res == CUDA_SUCCESS;
    if (!ok) {
        m_status = "Cannot load the GPU effect kernels: " + m_cu.errorText(res);
        if (errorLog[0]) {
            m_status += std::string(": ") + errorLog;
        }
    }
    for (int i = 0; ok && i < int(KernelId::Count); ++i) {
        res = m_cu.ModuleGetFunction(&m_kernels[i], m_module, kKernelNames[i]);
        if (res != CUDA_SUCCESS) {
            m_status = std::string("Missing GPU kernel ") + kKernelNames[i] + ": " + m_cu.errorText(res);
            ok = false;
        }
    }
    CUcontext previous = nullptr;
    m_cu.CtxPopCurrent(&previous);
    if (!ok) {
        logError("GPU effects disabled: %s", m_status.c_str());
        return false;
    }

    size_t limit = std::min<size_t>(m_device.totalMemory / 8, size_t(1) << 30);
    if (const char *mb = getenv("GPUFX_POOL_MB")) {
        limit = size_t(atoi(mb)) << 20;
    }
    m_poolLimit = std::max<size_t>(limit, size_t(64) << 20);

    // Page-locked memory holds the frames on their way to and from the GPU,
    // including downloaded frames that wait in MLT's queues: a 4K render
    // with 8 threads uses about 700 MB
    size_t hostLimit = size_t(1024) << 20;
    if (const char *mb = getenv("GPUFX_PINNED_MB")) {
        hostLimit = size_t(std::max(0, atoi(mb))) << 20;
    }
    m_hostLimit = m_cu.MemHostAlloc ? hostLimit : 0;

    char summary[512];
    snprintf(summary, sizeof(summary), "Using %s (compute %d.%d, %zu MiB, driver %d.%d)", m_device.name.c_str(), m_device.computeMajor, m_device.computeMinor,
             m_device.totalMemory >> 20, m_device.driverVersion / 1000, (m_device.driverVersion % 1000) / 10);
    m_status = summary;
    logInfo("GPU effects %s: %s", GPUFX_VERSION, summary);
    return true;
}

std::string Runtime::status() const
{
    std::lock_guard<std::mutex> lock(m_initMutex);
    if (m_state.load() == State::Uninitialized) {
        return "Not initialized";
    }
    return m_status;
}

DeviceInfo Runtime::device() const
{
    std::lock_guard<std::mutex> lock(m_initMutex);
    return m_device;
}

void Runtime::handleError(CUresult result, const char *what)
{
    if (isRecoverable(result)) {
        logWarning("%s: %s, using the CPU for this frame", what, m_cu.errorText(result).c_str());
        return;
    }
    State expected = State::Ready;
    if (m_state.compare_exchange_strong(expected, State::Failed)) {
        std::lock_guard<std::mutex> lock(m_initMutex);
        m_status = std::string("Disabled after a GPU error in ") + what + ": " + m_cu.errorText(result);
        logError("GPU effects turned off for this session: %s", m_status.c_str());
    }
}

CUresult Runtime::allocate(size_t bytes, CUdeviceptr *ptr, size_t *allocated)
{
    const size_t size = roundAllocation(bytes);
    {
        std::lock_guard<std::mutex> lock(m_poolMutex);
        for (size_t i = 0; i < m_pool.size(); ++i) {
            if (m_pool[i].bytes == size) {
                *ptr = m_pool[i].ptr;
                *allocated = size;
                m_pooledBytes -= size;
                m_pool.erase(m_pool.begin() + std::ptrdiff_t(i));
                return CUDA_SUCCESS;
            }
        }
    }
    if (m_pooledBytes > m_poolLimit) {
        trimPool(m_poolLimit / 2);
    }
    CUresult res = m_cu.MemAlloc(ptr, size);
    if (res == CUDA_ERROR_OUT_OF_MEMORY) {
        trimPool(0);
        res = m_cu.MemAlloc(ptr, size);
    }
    if (res == CUDA_SUCCESS) {
        *allocated = size;
    }
    return res;
}

void Runtime::recycle(CUdeviceptr ptr, size_t bytes)
{
    if (!ptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_poolMutex);
    m_pool.push_back({ptr, bytes});
    m_pooledBytes += bytes;
}

void Runtime::trimPool(size_t keepBytes)
{
    std::vector<PooledBlock> released;
    {
        std::lock_guard<std::mutex> lock(m_poolMutex);
        size_t i = 0;
        while (m_pooledBytes > keepBytes && i < m_pool.size()) {
            released.push_back(m_pool[i]);
            m_pooledBytes -= m_pool[i].bytes;
            ++i;
        }
        m_pool.erase(m_pool.begin(), m_pool.begin() + std::ptrdiff_t(i));
    }
    for (const PooledBlock &block : released) {
        m_cu.MemFree(block.ptr);
    }
}

CUresult Runtime::acquireStream(CUstream *stream, CUevent *event)
{
    {
        std::lock_guard<std::mutex> lock(m_streamMutex);
        if (!m_streams.empty()) {
            *stream = m_streams.back().first;
            *event = m_streams.back().second;
            m_streams.pop_back();
            return CUDA_SUCCESS;
        }
    }
    *event = nullptr;
    const CUresult res = m_cu.StreamCreate(stream, CU_STREAM_NON_BLOCKING);
    // With the context's default scheduling a thread waiting for the stream
    // spins on a CPU core, while one waiting for this event sleeps
    if (res == CUDA_SUCCESS && m_cu.EventCreate && m_cu.EventCreate(event, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
        *event = nullptr;
    }
    return res;
}

void Runtime::recycleStream(CUstream stream, CUevent event)
{
    std::lock_guard<std::mutex> lock(m_streamMutex);
    m_streams.emplace_back(stream, event);
}

void *Runtime::allocateHost(size_t bytes)
{
    const size_t size = roundAllocation(bytes);
    std::vector<void *> released;
    bool fits = false;
    {
        std::lock_guard<std::mutex> lock(m_hostMutex);
        for (auto &block : m_hostBlocks) {
            if (!block.second.inUse && block.second.bytes == size) {
                block.second.inUse = true;
                return const_cast<void *>(block.first);
            }
        }
        // Make room with the blocks of other sizes that are no longer used,
        // e.g. after a change of resolution. Freeing and allocating
        // page-locked memory takes long and can stall the GPU: when the
        // blocks are all in use, this copy uses pageable memory instead.
        const auto now = std::chrono::steady_clock::now();
        for (auto it = m_hostBlocks.begin(); it != m_hostBlocks.end() && m_hostBytes + size > m_hostLimit;) {
            if (it->second.inUse || now - it->second.released < std::chrono::seconds(10)) {
                ++it;
                continue;
            }
            released.push_back(const_cast<void *>(it->first));
            m_hostBytes -= it->second.bytes;
            it = m_hostBlocks.erase(it);
        }
        fits = m_hostBytes + size <= m_hostLimit;
        if (fits) {
            m_hostBytes += size;
        }
    }
    for (void *ptr : released) {
        m_cu.MemFreeHost(ptr);
    }
    if (!fits) {
        return nullptr;
    }
    void *ptr = nullptr;
    const CUresult res = m_cu.MemHostAlloc(&ptr, size, 0);
    std::lock_guard<std::mutex> lock(m_hostMutex);
    if (res != CUDA_SUCCESS || !ptr) {
        m_hostBytes -= size;
        return nullptr;
    }
    m_hostBlocks[ptr] = HostBlock{size, true, {}};
    return ptr;
}

bool Runtime::recycleHost(void *ptr)
{
    std::lock_guard<std::mutex> lock(m_hostMutex);
    auto it = m_hostBlocks.find(ptr);
    if (it == m_hostBlocks.end()) {
        return false;
    }
    it->second.inUse = false;
    it->second.released = std::chrono::steady_clock::now();
    return true;
}

bool Runtime::isHost(const void *ptr)
{
    std::lock_guard<std::mutex> lock(m_hostMutex);
    auto it = m_hostBlocks.find(ptr);
    return it != m_hostBlocks.end() && it->second.inUse;
}

CUresult DeviceBuffer::allocate(size_t bytes)
{
    reset();
    size_t allocated = 0;
    CUresult res = Runtime::instance().allocate(bytes, &m_ptr, &allocated);
    if (res == CUDA_SUCCESS) {
        m_size = allocated;
    } else {
        m_ptr = 0;
        m_size = 0;
    }
    return res;
}

void DeviceBuffer::reset()
{
    if (!m_ptr) {
        return;
    }
    if (t_scope) {
        // Work using this buffer may still be queued: recycle it once the
        // scope has synchronized its stream.
        t_scope->deferRecycle(m_ptr, m_size);
    } else {
        Runtime::instance().recycle(m_ptr, m_size);
    }
    m_ptr = 0;
    m_size = 0;
}

GpuScope::GpuScope()
    : m_runtime(Runtime::instance())
    , m_previous(t_scope)
{
    t_scope = this;
    if (!m_runtime.usable()) {
        return;
    }
    CudaDriver &cu = m_runtime.cu();
    CUresult res = cu.CtxPushCurrent(m_runtime.context());
    if (res != CUDA_SUCCESS) {
        m_runtime.handleError(res, "cuCtxPushCurrent");
        return;
    }
    m_pushed = true;
    res = m_runtime.acquireStream(&m_stream, &m_event);
    if (res != CUDA_SUCCESS) {
        m_stream = nullptr;
        m_event = nullptr;
        m_runtime.handleError(res, "cuStreamCreate");
        return;
    }
    m_ok = true;
}

GpuScope::~GpuScope()
{
    CudaDriver &cu = m_runtime.cu();
    if (m_stream) {
        synchronize();
        m_runtime.recycleStream(m_stream, m_event);
    }
    t_scope = m_previous;
    for (const auto &block : m_deferred) {
        m_runtime.recycle(block.first, block.second);
    }
    for (void *staging : m_staging) {
        m_runtime.recycleHost(staging);
    }
    if (m_pushed) {
        CUcontext previous = nullptr;
        cu.CtxPopCurrent(&previous);
    }
}

void GpuScope::deferRecycle(CUdeviceptr ptr, size_t bytes)
{
    m_deferred.emplace_back(ptr, bytes);
}

CUresult GpuScope::upload(CUdeviceptr dst, const void *src, size_t bytes)
{
    // From pageable memory the driver copies the data to a buffer of its own
    // and waits for each part of it to reach the GPU
    if (bytes >= kStagingBytes && !m_runtime.isHost(src)) {
        if (void *staging = m_runtime.allocateHost(bytes)) {
            memcpy(staging, src, bytes);
            m_staging.push_back(staging);
            src = staging;
        }
    }
    return m_runtime.cu().MemcpyHtoDAsync(dst, src, bytes, m_stream);
}

CUresult GpuScope::download(void *dst, CUdeviceptr src, size_t bytes)
{
    return m_runtime.cu().MemcpyDtoHAsync(dst, src, bytes, m_stream);
}

CUresult GpuScope::copy(CUdeviceptr dst, CUdeviceptr src, size_t bytes)
{
    return m_runtime.cu().MemcpyDtoDAsync(dst, src, bytes, m_stream);
}

CUresult GpuScope::fill(CUdeviceptr dst, unsigned char value, size_t bytes)
{
    return m_runtime.cu().MemsetD8Async(dst, value, bytes, m_stream);
}

CUresult GpuScope::synchronize()
{
    CudaDriver &cu = m_runtime.cu();
    if (m_event) {
        const CUresult res = cu.EventRecord(m_event, m_stream);
        return res == CUDA_SUCCESS ? cu.EventSynchronize(m_event) : res;
    }
    return cu.StreamSynchronize(m_stream);
}

CUresult GpuScope::launchRaw(KernelId id, unsigned gridX, unsigned gridY, unsigned blockX, unsigned blockY, void **params)
{
    return m_runtime.cu().LaunchKernel(m_runtime.kernel(id), gridX, gridY, 1, blockX, blockY, 1, 0, m_stream, params, nullptr);
}

} // namespace gpufx
