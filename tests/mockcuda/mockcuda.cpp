/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

/*
 * A stand-in for the NVIDIA driver (libcuda) used by the gpufx tests.
 *
 * It implements the part of the CUDA driver API that the module uses on the
 * CPU: device memory is host memory, copies are memcpy, and kernel launches
 * run the kernels compiled as C++ (emulated_kernels.cpp) once per thread of
 * the grid. That exercises the module's real code paths (context handling,
 * pools, uploads, launches with their parameter arrays, hand-off between
 * effects, error handling) on machines without a GPU.
 *
 * Environment:
 *  MOCKCUDA_NO_DEVICE=1   report no CUDA device
 *  MOCKCUDA_INIT_ERROR=n  make cuInit() return n
 */

#include "mockcuda.h"
#include "../../kernels/gpufx_kernel_list.h"
#include "emulation.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

struct GpufxDim3
{
    unsigned x, y, z;
};
thread_local GpufxDim3 threadIdx;
thread_local GpufxDim3 blockIdx;
thread_local GpufxDim3 blockDim;
thread_local GpufxDim3 gridDim;

namespace {

using CUresult = int;
using CUdeviceptr = unsigned long long;

constexpr CUresult kSuccess = 0;
constexpr CUresult kInvalidValue = 1;
constexpr CUresult kOutOfMemory = 2;
constexpr CUresult kNoDevice = 100;
constexpr CUresult kInvalidContext = 201;
constexpr CUresult kInvalidPtx = 218;
constexpr CUresult kNotFound = 500;
constexpr CUresult kLaunchFailed = 719;

struct State
{
    std::mutex mutex;
    std::map<std::string, GpufxTrampoline> kernels;
    std::map<uintptr_t, size_t> allocations;
    MockCudaStats stats{};
    int failAfter[MOCKCUDA_OP_COUNT] = {0};
    CUresult failWith[MOCKCUDA_OP_COUNT] = {0};
};

State &state()
{
    static State *s = new State();
    return *s;
}

thread_local int t_contextDepth = 0;
int g_context = 0;
int g_module = 0;

bool noDevice()
{
    const char *value = getenv("MOCKCUDA_NO_DEVICE");
    return value && value[0] && strcmp(value, "0") != 0;
}

// Returns an error to inject for this operation, or kSuccess
CUresult injectedFailure(MockCudaOp op)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.failAfter[op] > 0 && --s.failAfter[op] == 0) {
        return s.failWith[op];
    }
    return kSuccess;
}

bool validRange(uintptr_t ptr, size_t bytes)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.allocations.upper_bound(ptr);
    if (it == s.allocations.begin()) {
        return false;
    }
    --it;
    return ptr >= it->first && ptr + bytes <= it->first + it->second;
}

void complain(const char *what)
{
    fprintf(stderr, "mockcuda: %s\n", what);
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.stats.misuse++;
}

} // namespace

void gpufx_emu_register(const char *name, GpufxTrampoline trampoline)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.kernels[name] = trampoline;
}

extern "C" {

MOCKCUDA_EXPORT void mockcuda_get_stats(MockCudaStats *stats)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.stats.liveAllocations = s.allocations.size();
    *stats = s.stats;
}

MOCKCUDA_EXPORT void mockcuda_reset_stats()
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.stats = MockCudaStats{};
}

MOCKCUDA_EXPORT void mockcuda_fail_after(MockCudaOp op, int count, int error)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.failAfter[op] = count;
    s.failWith[op] = error;
}

MOCKCUDA_EXPORT CUresult cuInit(unsigned int)
{
    if (const char *error = getenv("MOCKCUDA_INIT_ERROR")) {
        return atoi(error);
    }
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuDriverGetVersion(int *version)
{
    *version = 12040;
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuDeviceGetCount(int *count)
{
    *count = noDevice() ? 0 : 1;
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuDeviceGet(int *device, int ordinal)
{
    if (noDevice() || ordinal != 0) {
        return kNoDevice;
    }
    *device = 0;
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuDeviceGetName(char *name, int length, int)
{
    snprintf(name, size_t(length), "gpufx emulated GPU");
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuDeviceGetAttribute(int *value, int attribute, int)
{
    switch (attribute) {
    case 75: // compute capability major
        *value = 8;
        break;
    case 76:
        *value = 6;
        break;
    case 16: // multiprocessors
        *value = 4;
        break;
    default:
        *value = 0;
    }
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuDeviceTotalMem_v2(size_t *bytes, int)
{
    *bytes = size_t(4) << 30;
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuDevicePrimaryCtxRetain(void **context, int)
{
    *context = &g_context;
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuCtxPushCurrent_v2(void *context)
{
    if (context != &g_context) {
        return kInvalidContext;
    }
    ++t_contextDepth;
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuCtxPopCurrent_v2(void **context)
{
    if (t_contextDepth <= 0) {
        complain("cuCtxPopCurrent without context");
        return kInvalidContext;
    }
    --t_contextDepth;
    if (context) {
        *context = &g_context;
    }
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuModuleLoadDataEx(void **module, const void *image, unsigned int, int *, void **)
{
    if (t_contextDepth <= 0) {
        complain("cuModuleLoadDataEx without context");
        return kInvalidContext;
    }
    // Check that the PTX the module embeds declares every kernel we emulate
    const char *ptx = static_cast<const char *>(image);
    if (!ptx || !strstr(ptx, ".version") || !strstr(ptx, ".target")) {
        return kInvalidPtx;
    }
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (const auto &kernel : s.kernels) {
        const std::string entry = ".entry " + kernel.first + "(";
        if (!strstr(ptx, entry.c_str())) {
            fprintf(stderr, "mockcuda: kernel %s is missing from the PTX\n", kernel.first.c_str());
            return kInvalidPtx;
        }
    }
    *module = &g_module;
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuModuleGetFunction(void **function, void *module, const char *name)
{
    if (module != &g_module) {
        return kInvalidValue;
    }
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.kernels.find(name);
    if (it == s.kernels.end()) {
        return kNotFound;
    }
    *function = reinterpret_cast<void *>(it->second);
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuMemAlloc_v2(CUdeviceptr *ptr, size_t bytes)
{
    if (t_contextDepth <= 0) {
        complain("cuMemAlloc without context");
        return kInvalidContext;
    }
    if (CUresult error = injectedFailure(MOCKCUDA_ALLOC)) {
        return error;
    }
    void *memory = malloc(bytes);
    if (!memory) {
        return kOutOfMemory;
    }
    // Device memory is not initialized: make code relying on zeros fail
    memset(memory, 0xcd, bytes);
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.allocations[uintptr_t(memory)] = bytes;
    s.stats.allocations++;
    *ptr = CUdeviceptr(uintptr_t(memory));
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuMemFree_v2(CUdeviceptr ptr)
{
    State &s = state();
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        auto it = s.allocations.find(uintptr_t(ptr));
        if (it == s.allocations.end()) {
            fprintf(stderr, "mockcuda: cuMemFree of unknown pointer\n");
            s.stats.misuse++;
            return kInvalidValue;
        }
        s.allocations.erase(it);
        s.stats.frees++;
    }
    free(reinterpret_cast<void *>(uintptr_t(ptr)));
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuMemGetInfo_v2(size_t *freeBytes, size_t *totalBytes)
{
    *freeBytes = size_t(3) << 30;
    *totalBytes = size_t(4) << 30;
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t bytes, void *)
{
    if (t_contextDepth <= 0) {
        complain("cuMemcpyHtoD without context");
        return kInvalidContext;
    }
    if (CUresult error = injectedFailure(MOCKCUDA_UPLOAD)) {
        return error;
    }
    if (!validRange(uintptr_t(dst), bytes)) {
        complain("cuMemcpyHtoD out of bounds");
        return kInvalidValue;
    }
    memcpy(reinterpret_cast<void *>(uintptr_t(dst)), src, bytes);
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.stats.uploads++;
    s.stats.uploadedBytes += bytes;
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t bytes, void *)
{
    if (t_contextDepth <= 0) {
        complain("cuMemcpyDtoH without context");
        return kInvalidContext;
    }
    if (CUresult error = injectedFailure(MOCKCUDA_DOWNLOAD)) {
        return error;
    }
    if (!validRange(uintptr_t(src), bytes)) {
        complain("cuMemcpyDtoH out of bounds");
        return kInvalidValue;
    }
    memcpy(dst, reinterpret_cast<const void *>(uintptr_t(src)), bytes);
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.stats.downloads++;
    s.stats.downloadedBytes += bytes;
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes, void *)
{
    if (t_contextDepth <= 0) {
        complain("cuMemcpyDtoD without context");
        return kInvalidContext;
    }
    if (!validRange(uintptr_t(src), bytes) || !validRange(uintptr_t(dst), bytes)) {
        complain("cuMemcpyDtoD out of bounds");
        return kInvalidValue;
    }
    memmove(reinterpret_cast<void *>(uintptr_t(dst)), reinterpret_cast<const void *>(uintptr_t(src)), bytes);
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuMemsetD8Async(CUdeviceptr dst, unsigned char value, size_t count, void *)
{
    if (t_contextDepth <= 0) {
        complain("cuMemsetD8 without context");
        return kInvalidContext;
    }
    if (!validRange(uintptr_t(dst), count)) {
        complain("cuMemsetD8 out of bounds");
        return kInvalidValue;
    }
    memset(reinterpret_cast<void *>(uintptr_t(dst)), value, count);
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuStreamCreate(void **stream, unsigned int)
{
    if (t_contextDepth <= 0) {
        complain("cuStreamCreate without context");
        return kInvalidContext;
    }
    static int streams[64];
    static int next = 0;
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    *stream = &streams[next++ % 64];
    s.stats.streams++;
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuStreamSynchronize(void *)
{
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuLaunchKernel(void *function, unsigned int gridX, unsigned int gridY, unsigned int gridZ, unsigned int blockX, unsigned int blockY,
                                        unsigned int blockZ, unsigned int sharedBytes, void *, void **params, void **extra)
{
    if (t_contextDepth <= 0) {
        complain("cuLaunchKernel without context");
        return kInvalidContext;
    }
    if (!function || !params || extra || sharedBytes || blockX * blockY * blockZ == 0 || blockX * blockY * blockZ > 1024 || gridX == 0 || gridY == 0 ||
        gridZ == 0 || gridY > 65535 || gridZ > 65535 || blockZ > 64) {
        complain("invalid kernel launch");
        return kInvalidValue;
    }
    if (CUresult error = injectedFailure(MOCKCUDA_LAUNCH)) {
        return error;
    }
    auto trampoline = reinterpret_cast<GpufxTrampoline>(function);
    gridDim = {gridX, gridY, gridZ};
    blockDim = {blockX, blockY, blockZ};
    for (unsigned z = 0; z < gridZ; ++z) {
        for (unsigned y = 0; y < gridY; ++y) {
            for (unsigned x = 0; x < gridX; ++x) {
                blockIdx = {x, y, z};
                for (unsigned tz = 0; tz < blockZ; ++tz) {
                    for (unsigned ty = 0; ty < blockY; ++ty) {
                        for (unsigned tx = 0; tx < blockX; ++tx) {
                            threadIdx = {tx, ty, tz};
                            trampoline(params);
                        }
                    }
                }
            }
        }
    }
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.stats.launches++;
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuGetErrorName(CUresult error, const char **name)
{
    switch (error) {
    case kSuccess:
        *name = "CUDA_SUCCESS";
        break;
    case kOutOfMemory:
        *name = "CUDA_ERROR_OUT_OF_MEMORY";
        break;
    case kLaunchFailed:
        *name = "CUDA_ERROR_LAUNCH_FAILED";
        break;
    default:
        *name = "CUDA_ERROR_MOCK";
    }
    return kSuccess;
}

MOCKCUDA_EXPORT CUresult cuGetErrorString(CUresult, const char **text)
{
    *text = "emulated error";
    return kSuccess;
}

} // extern "C"
