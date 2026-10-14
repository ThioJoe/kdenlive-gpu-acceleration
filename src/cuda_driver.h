/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <cstddef>
#include <string>

/*
 * Minimal declarations of the CUDA driver API, loaded at runtime from the
 * NVIDIA driver (libcuda.so.1 / nvcuda.dll). Nothing here needs the CUDA
 * toolkit: the module builds everywhere and simply reports CUDA as
 * unavailable on machines without an NVIDIA driver.
 */

#if defined(_WIN32)
#define GPUFX_CUDAAPI __stdcall
#else
#define GPUFX_CUDAAPI
#endif

namespace gpufx {

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
using CUcontext = struct CUctx_st *;
using CUmodule = struct CUmod_st *;
using CUfunction = struct CUfunc_st *;
using CUstream = struct CUstream_st *;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
constexpr CUresult CUDA_ERROR_OUT_OF_MEMORY = 2;
constexpr CUresult CUDA_ERROR_NOT_INITIALIZED = 3;
constexpr CUresult CUDA_ERROR_NO_DEVICE = 100;
constexpr CUresult CUDA_ERROR_NOT_FOUND = 500;

constexpr int CU_JIT_INFO_LOG_BUFFER = 3;
constexpr int CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;
constexpr int CU_JIT_ERROR_LOG_BUFFER = 5;
constexpr int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;

constexpr int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

constexpr unsigned int CU_STREAM_NON_BLOCKING = 0x1;

struct CudaDriver
{
    CUresult(GPUFX_CUDAAPI *Init)(unsigned int flags) = nullptr;
    CUresult(GPUFX_CUDAAPI *DriverGetVersion)(int *version) = nullptr;
    CUresult(GPUFX_CUDAAPI *DeviceGetCount)(int *count) = nullptr;
    CUresult(GPUFX_CUDAAPI *DeviceGet)(CUdevice *device, int ordinal) = nullptr;
    CUresult(GPUFX_CUDAAPI *DeviceGetName)(char *name, int len, CUdevice device) = nullptr;
    CUresult(GPUFX_CUDAAPI *DeviceGetAttribute)(int *value, int attribute, CUdevice device) = nullptr;
    CUresult(GPUFX_CUDAAPI *DeviceTotalMem)(size_t *bytes, CUdevice device) = nullptr;
    CUresult(GPUFX_CUDAAPI *DevicePrimaryCtxRetain)(CUcontext *context, CUdevice device) = nullptr;
    CUresult(GPUFX_CUDAAPI *CtxPushCurrent)(CUcontext context) = nullptr;
    CUresult(GPUFX_CUDAAPI *CtxPopCurrent)(CUcontext *context) = nullptr;
    CUresult(GPUFX_CUDAAPI *ModuleLoadDataEx)(CUmodule *module, const void *image, unsigned int numOptions, int *options, void **optionValues) = nullptr;
    CUresult(GPUFX_CUDAAPI *ModuleGetFunction)(CUfunction *function, CUmodule module, const char *name) = nullptr;
    CUresult(GPUFX_CUDAAPI *MemAlloc)(CUdeviceptr *ptr, size_t bytes) = nullptr;
    CUresult(GPUFX_CUDAAPI *MemFree)(CUdeviceptr ptr) = nullptr;
    CUresult(GPUFX_CUDAAPI *MemGetInfo)(size_t *freeBytes, size_t *totalBytes) = nullptr;
    CUresult(GPUFX_CUDAAPI *MemcpyHtoDAsync)(CUdeviceptr dst, const void *src, size_t bytes, CUstream stream) = nullptr;
    CUresult(GPUFX_CUDAAPI *MemcpyDtoHAsync)(void *dst, CUdeviceptr src, size_t bytes, CUstream stream) = nullptr;
    CUresult(GPUFX_CUDAAPI *MemcpyDtoDAsync)(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream) = nullptr;
    CUresult(GPUFX_CUDAAPI *MemsetD8Async)(CUdeviceptr dst, unsigned char value, size_t count, CUstream stream) = nullptr;
    CUresult(GPUFX_CUDAAPI *StreamCreate)(CUstream *stream, unsigned int flags) = nullptr;
    CUresult(GPUFX_CUDAAPI *StreamSynchronize)(CUstream stream) = nullptr;
    CUresult(GPUFX_CUDAAPI *LaunchKernel)(CUfunction function, unsigned int gridX, unsigned int gridY, unsigned int gridZ, unsigned int blockX,
                                          unsigned int blockY, unsigned int blockZ, unsigned int sharedBytes, CUstream stream, void **params,
                                          void **extra) = nullptr;
    // Optional, only used for better error messages
    CUresult(GPUFX_CUDAAPI *GetErrorName)(CUresult error, const char **name) = nullptr;
    CUresult(GPUFX_CUDAAPI *GetErrorString)(CUresult error, const char **text) = nullptr;

    /** Loads the driver library and resolves the entry points.
     *  The library can be overridden with the GPUFX_CUDA_LIBRARY environment
     *  variable (used by the tests to load an emulated driver). */
    bool load(std::string &error);

    /** "CUDA_ERROR_xxx (description)" for log messages */
    std::string errorText(CUresult result) const;

private:
    void *m_library = nullptr;
};

} // namespace gpufx
