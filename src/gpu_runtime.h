/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#pragma once

#include "../kernels/gpufx_kernel_list.h"
#include "cuda_driver.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace gpufx {

enum class KernelId : int {
#define GPUFX_KERNEL_ENUM(name) name,
    GPUFX_KERNEL_LIST(GPUFX_KERNEL_ENUM)
#undef GPUFX_KERNEL_ENUM
        Count
};

struct DeviceInfo
{
    std::string name;
    int ordinal = -1;
    int computeMajor = 0;
    int computeMinor = 0;
    int multiprocessors = 0;
    size_t totalMemory = 0;
    int driverVersion = 0;
};

/**
 * Process wide CUDA state: driver, primary context, kernels, and pools of
 * device memory and streams. Initialization is lazy, so processes that never
 * render a supported effect never touch the GPU.
 */
class Runtime
{
public:
    static Runtime &instance();

    /** Initializes CUDA on first call. Thread-safe. Returns true when the GPU can be used. */
    bool ensureInitialized();
    /** True when initialized and not disabled after an error. Never triggers initialization. */
    bool usable() const { return m_state.load() == State::Ready; }
    /** Human readable state, for logs and the status service */
    std::string status() const;
    DeviceInfo device() const;

    CudaDriver &cu() { return m_cu; }
    CUcontext context() const { return m_context; }
    CUfunction kernel(KernelId id) const { return m_kernels[int(id)]; }

    /** Called after a failed CUDA call. Errors that leave the context unusable
     *  turn GPU processing off for the rest of the session. */
    void handleError(CUresult result, const char *what);

    /** Device memory from the pool. Must be called with the context current. */
    CUresult allocate(size_t bytes, CUdeviceptr *ptr, size_t *allocated);
    /** Returns memory to the pool. Safe from any thread, no CUDA call. */
    void recycle(CUdeviceptr ptr, size_t bytes);

    CUresult acquireStream(CUstream *stream);
    void recycleStream(CUstream stream);

    uint64_t gpuFrames() const { return m_gpuFrames.load(); }
    uint64_t cpuFrames() const { return m_cpuFrames.load(); }
    void countFrame(bool gpu) { (gpu ? m_gpuFrames : m_cpuFrames).fetch_add(1, std::memory_order_relaxed); }

private:
    enum class State { Uninitialized, Ready, Unavailable, Failed };
    Runtime() = default;
    bool initialize();
    void trimPool(size_t keepBytes);

    struct PooledBlock
    {
        CUdeviceptr ptr;
        size_t bytes;
    };

    CudaDriver m_cu;
    std::atomic<State> m_state{State::Uninitialized};
    mutable std::mutex m_initMutex;
    std::string m_status;
    DeviceInfo m_device;
    CUcontext m_context = nullptr;
    CUmodule m_module = nullptr;
    CUfunction m_kernels[int(KernelId::Count)] = {};

    std::mutex m_poolMutex;
    std::vector<PooledBlock> m_pool;
    size_t m_pooledBytes = 0;
    size_t m_poolLimit = 0;

    std::mutex m_streamMutex;
    std::vector<CUstream> m_streams;

    std::atomic<uint64_t> m_gpuFrames{0};
    std::atomic<uint64_t> m_cpuFrames{0};
};

/** A block of pooled device memory, returned to the pool when destroyed. */
class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&other) noexcept
        : m_ptr(other.m_ptr)
        , m_size(other.m_size)
    {
        other.m_ptr = 0;
        other.m_size = 0;
    }
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this != &other) {
            reset();
            m_ptr = other.m_ptr;
            m_size = other.m_size;
            other.m_ptr = 0;
            other.m_size = 0;
        }
        return *this;
    }

    /** Allocates at least bytes. Needs a GpuScope. */
    CUresult allocate(size_t bytes);
    void reset();
    bool valid() const { return m_ptr != 0; }
    CUdeviceptr ptr() const { return m_ptr; }
    size_t size() const { return m_size; }

private:
    CUdeviceptr m_ptr = 0;
    size_t m_size = 0;
};

/**
 * Makes the gpufx context current on this thread and provides a stream for
 * the duration of one effect. Everything enqueued on the stream must be
 * synchronized before buffers are handed to another stream or to the host.
 */
class GpuScope
{
public:
    GpuScope();
    ~GpuScope();
    GpuScope(const GpuScope &) = delete;
    GpuScope &operator=(const GpuScope &) = delete;

    bool ok() const { return m_ok; }
    CUstream stream() const { return m_stream; }

    /** Buffers released while the scope is alive are recycled after the
     *  stream has been synchronized, in the destructor. */
    void deferRecycle(CUdeviceptr ptr, size_t bytes);

    CUresult upload(CUdeviceptr dst, const void *src, size_t bytes);
    CUresult download(void *dst, CUdeviceptr src, size_t bytes);
    CUresult copy(CUdeviceptr dst, CUdeviceptr src, size_t bytes);
    CUresult fill(CUdeviceptr dst, unsigned char value, size_t bytes);
    CUresult synchronize();

    /** Launches a kernel on a 2D grid covering width x height work items. */
    template <typename... Args> CUresult launch2D(KernelId id, int width, int height, unsigned blockX, unsigned blockY, Args... args)
    {
        if (width <= 0 || height <= 0) {
            return CUDA_SUCCESS;
        }
        void *params[] = {const_cast<void *>(static_cast<const void *>(&args))...};
        const unsigned gridX = (unsigned(width) + blockX - 1) / blockX;
        const unsigned gridY = (unsigned(height) + blockY - 1) / blockY;
        return launchRaw(id, gridX, gridY, blockX, blockY, params);
    }

    /** Launches a kernel on a 1D grid covering count work items. */
    template <typename... Args> CUresult launch1D(KernelId id, int count, Args... args)
    {
        if (count <= 0) {
            return CUDA_SUCCESS;
        }
        void *params[] = {const_cast<void *>(static_cast<const void *>(&args))...};
        const unsigned block = 128;
        return launchRaw(id, (unsigned(count) + block - 1) / block, 1, block, 1, params);
    }

private:
    CUresult launchRaw(KernelId id, unsigned gridX, unsigned gridY, unsigned blockX, unsigned blockY, void **params);

    Runtime &m_runtime;
    GpuScope *m_previous = nullptr;
    CUstream m_stream = nullptr;
    bool m_pushed = false;
    bool m_ok = false;
    std::vector<std::pair<CUdeviceptr, size_t>> m_deferred;
};

} // namespace gpufx
